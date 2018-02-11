/**
 * @file
 * bp_problem.cuh
 *
 * @brief Gunrock GPU simplified implementation of the belief propagation problem
*/

#pragma once

#include <cub/cub.cuh>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace bp {

template<typename VertexId, typename SizeT, typename Value>
struct BPProblem: ProblemBase<VertexId, SizeT, Value,
        true, // MARK_PREDECESSORS
        false> // ENABLE IDEMPOTENCE
{
    static const bool MARK_PREDECESSORS = true;
    static const bool ENABLE_IDEMPOTENCE = false;
    static const int MAX_NUM_VERTEX_ASSOCIATES = 0;
    static const int MAX_NUM_VALUE__ASSOCIATES = 1;

    typedef ProblemBase <VertexId, SizeT, Value,
            MARK_PREDECESSORS, ENABLE_IDEMPOTENCE> BaseProblem;
    typedef DataSliceBase <VertexId, SizeT, Value,
            MAX_NUM_VERTEX_ASSOCIATES, MAX_NUM_VALUE__ASSOCIATES> BaseDataSlice;
    typedef unsigned char MaskT;

    struct DataSlice : BaseDataSlice
    {
        // device storage arrays
        util::Array1D<SizeT, Value> belief_curr;
        util::Array1D<SizeT, Value> belief_next;
        util::Array1D<SizeT, Value> joint_probabilities;
        Value threshold;
        Value delta;
        SizeT max_iter;


        /**
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice(),
                      threshold(0),
                      delta(0),
                      max_iter(0)
        {
            belief_curr.SetName("belief_curr");
            belief_next.SetName("belief_next");
            joint_probabilities.SetName("joint_probabilities");
        }

        /**
         * @brief Default destructor
         */
        virtual ~DataSlice()
        {
            Release();
        }

        cudaError_t Release()
        {
            cudaSuccess_t retval = cudaSuccess;
            if (retval = util::SetDevice(this->gpu_idx)) return retval;
            if (retval = BaseDataSlice::Release()) return retval;
            if (retval = belief_curr.Release()) return retval;
            if (retval = belief_next.Release()) return retval;
            if (retval = joint_probabilities.Release()) return retval;
        }

        /**
         * @brief copy result beliefs on the GPU back to host-side vectors
         *
         * @param[out] h_beliefs host-side vector to store compute beliefs
         *
         * @return cudaError_t object Indicates the success of all CUDA calls
         */
        cudaError_t Extract(Value *h_beliefs)
        {
            cudaError_t retval = cudaSuccess;

            if (this->num_gpus == 1)
            {
                // Set device
                if (retval == util::SetDevice(this->gpu_idx[0])) return retval;

                data_slices[0]->belief_curr.SetPointer(h_beliefs);
                if (retval = data_slices[0]->belief_curr.Move(util::DEVICE, util::HOST)) return retval;
            } else{
                Value **th_beliefs = new Value*[this->num_gpus];
                for (int gpu = 0; gpu < this->num_gpus; gpu++) {
                    if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                    if (retval = data_slices[gpu]->belief_curr.Move(util::DEVICE, util::HOST)) return retval;
                    th_beliefs[gpu] = data_slices[gpu]->belief_curr.GetPoint(util::HOST);
                }

                for (VertexId node = 0; node < this->nodes; node++)
                {
                    if (this->partition_tables[0][node] >= 0 && this->partition_tables[0][node] < this->num_gpus &&
                        this->convertion_tables[0][node] >= 0 && this->convertion_tables[0][node] < data_slices[this->partition_tables[0][node]]->belief_curr.GetSize())
                    {
                        h_beliefs[node] = th_beliefs[this->partition_tables[0][node]][this->convertion_tables[0][node]];
                    } else{
                        printf("OutOfBound: node = %d, partition = %d, convertion = %d\n",
                               node, this->partition_tables[0][node], this->convertion_tables[0][node]);
                        //data_slices[this->partition_tables[0][node]]->distance.GetSize());
                        fflush(stdout);
                    }
                }

                for (int gpu = 0; gpu < this->num_gpus; gpu++)
                {
                    if (retval = data_slices[gpu]->belief_curr.Release(util::HOST)) return retval;
                    if (retval = data_slices[gpu]->belief_next.Release(util::HOST)) return retval;
                    if (retval = data_slices[gpu]->joint_probabilities(util::HOST)) return retval;
                }
                delete[] th_beliefs; th_beliefs = NULL;
            }

            return retval;
        }

        /**
         * @brief initialization function
         *
         * @param[in] num_gpus Number of the GPUs used.
         * @param[in] gpu_idx GPU index used for testing.
         * @param[in] use_double_buffer Whether to use double buffer.
         * @param[in] graph Pointer to the graph we process on.
         * @param[in] graph_slice Pointer to the GraphSlice object.
         * @param[in] num_in_nodes
         * @param[in] num_out_nodes
         * @param[in] queue_sizing Maximum queue sizing factor.
         * @param[in] in_sizing
         * @param[in] skip_makeout_selection
         * @param[in] keep_node_num
         *
         * @return cudaError_t object Indicates the success of all CUDA calls.
         */
        cudaError_t Init(
                int num_gpus,
                int gpu_idx,
                bool use_double_buffer,
                Csr<VertexId, SizeT, Value> *graph,
                GraphSlice<VertexId, SizeT, Value> *graph_slice,
                SizeT *num_in_nodes,
                SizeT *num_out_nodes,
                float queue_sizing = 2.0,
                float in_sizing = 1.0,
                bool skip_makeout_selection = false,
                bool keep_node_num = false)
        {
            cudaSuccess_t retval = cudaSuccess;

            if (retval = BaseDataSlice::Init(
                    num_gpus,
                    gpu_idx,
                    use_double_buffer,
                    graph,
                    num_in_nodes,
                    num_out_nodes,
                    in_sizing,
                    skip_makeout_selection
            )){
                return retval;
            }

            if (retval = belief_curr.Allocate(graph->nodes, util::DEVICE)) return retval;
            if (retval = belief_next.Allocate(graph->nodes, util::DEVICE)) return retval;
            if (retval = joint_probabilities.Allocate(graph->edges, util::DEVICE)) return retval;

            // copy data
            belief_curr.SetPointer(graph->node_values, graph->nodes, util::HOST);
            if (retval = belief_curr.Move(util::HOST, util::DEVICE)) return retval;
            belief_next.SetPointer(graph->node_values, graph->nodes, util::HOST);
            if (retval = belief_next.Move(util::HOST, util::DEVICE)) return retval;
            joint_probabilities.SetPointer(graph->edge_values, util::DEVICE);
            if (retval = joint_probabilities.Move(util::HOST, util::DEVICE)) return retval;

            // labels is required
            if (retval = this->labels.Allocate(graph->nodes, util::DEVICE)) return retval;

            if (num_gpus > 1)
            {
                this->value__associate_orgs[0] = belief_curr.GetPointer(util::DEVICE);
                if (retval = this->value__associate_orgs.Move(util::HOST, util::DEVICE)) return retval;
            }

            return retval;
        }

        /**
         * @brief Reset problem function. Must be called prior to each run.
         *
         * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed).
         * @param[in] graph_slice Pointer to the graph slice we process on.
         * @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
         * @param[in] queue_sizing1 Size scaling factor for work queue allocation.
         *
         * \return cudaError_t object Indicates the success of all CUDA calls.
         */
        cudaError_t Reset(
                Value delta,
                Value threshold,
                SizeT max_iter,
                FrontierType frontier_type,
                GraphSlice<VertexId, SizeT, Value>  *graph_slice,
                double queue_sizing = 2.0,
        double queue_sizing1 = -1.0)
        {
            cudaError_t retval = cudaSuccess;
            SizeT nodes = graph_slice -> nodes;
            SizeT edges = graph_slice -> edges;
            SizeT new_frontier_elements[2] = {0,0};
            if (queue_sizing1 < 0){
                queue_sizing1 = queue_sizing;
            }

            for (int gpu = 0; gpu < this -> num_gpus; gpu++) {
                this->wait_marker[gpu] = 0;
            }
            for (int i=0; i<4; i++) {
                for (int gpu = 0; gpu < this->num_gpus * 2; gpu++) {
                    for (int stage = 0; stage < this->num_stages; stage++) {
                        this->events_set[i][gpu][stage] = false;
                    }
                }
            }
            for (int gpu = 0; gpu < this -> num_gpus; gpu++) {
                for (int i = 0; i < 2; i++) {
                    this->in_length[i][gpu] = 0;
                }
            }
            for (int peer=0; peer<this->num_gpus; peer++) {
                this->out_length[peer] = 1;
            }

            for (int peer=0;peer<(this->num_gpus > 1 ? this->num_gpus+1 : 1);peer++) {
                for (int i = 0; i < 2; i++) {
                    double queue_sizing_ = i == 0 ? queue_sizing : queue_sizing1;
                    switch (frontier_type) {
                        case VERTEX_FRONTIERS :
                            // O(n) ping-pong global vertex frontiers
                            new_frontier_elements[0] =
                            double(this->num_gpus > 1 ? graph_slice->in_counter[peer] : nodes) *queue_sizing_ + 2;
                            new_frontier_elements[1] = new_frontier_elements[0];
                            break;

                        case EDGE_FRONTIERS :
                            // O(m) ping-pong global edge frontiers
                            new_frontier_elements[0] =
                            double(edges)
                            *queue_sizing_ + 2;
                            new_frontier_elements[1] = new_frontier_elements[0];
                            break;

                        case MIXED_FRONTIERS :
                            // O(n) global vertex frontier, O(m) global edge frontier
                            new_frontier_elements[0] =
                            double(this->num_gpus > 1 ? graph_slice->in_counter[peer] : nodes) *queue_sizing_ + 2;
                            new_frontier_elements[1] =
                            double(edges)
                            *queue_sizing_ + 2;
                            break;
                    }

                    // Iterate through global frontier queue setups
                    //for (int i = 0; i < 2; i++)
                    {
                        if (peer == this->num_gpus && i == 1) continue;
                        if (new_frontier_elements[i] > edges + 2 && queue_sizing_ > 10)
                            new_frontier_elements[i] = edges + 2;
                        //if (peer == this->num_gpus && new_frontier_elements[i] > nodes * this->num_gpus) new_frontier_elements[i] = nodes * this->num_gpus;
                        if (this->frontier_queues[peer].keys[i].GetSize() < new_frontier_elements[i]) {

                            // Free if previously allocated
                            if (retval = this->frontier_queues[peer].keys[i].Release()) return retval;

                            // Free if previously allocated
                            if (false) {
                                if (retval = this->frontier_queues[peer].values[i].Release()) return retval;
                            }
                            //frontier_elements[peer][i] = new_frontier_elements[i];

                            if (retval = this->frontier_queues[peer].keys[i].Allocate(new_frontier_elements[i],
                                                                                      util::DEVICE))
                                return retval;
                            if (false) {
                                if (retval = this->frontier_queues[peer].values[i].Allocate(new_frontier_elements[i],
                                                                                            util::DEVICE))
                                    return retval;
                            }
                        } //end if
                    } // end for i<2

                    if (peer == this->num_gpus || i == 1) {
                        continue;
                    }
                    //if (peer == num_gpu) continue;
                    SizeT max_elements = new_frontier_elements[0];
                    if (new_frontier_elements[1] > max_elements) {
                        max_elements = new_frontier_elements[1];
                    }
                    if (max_elements > nodes) {
                        max_elements = nodes;
                    }
                    if (this->scanned_edges[peer].GetSize() < max_elements) {
                        if (retval = this->scanned_edges[peer].Release()) return retval;
                        if (retval = this->scanned_edges[peer].Allocate(max_elements, util::DEVICE)) return retval;
                    }
                }
            }

            // Allocate output distances if necessary
            if (this->belief_curr.GetPointer(util::DEVICE) == NULL) {
                if (retval = this->belief_curr.Allocate(nodes, util::DEVICE)) return retval;
                // copy data
                if (retval = this->belief_curr.Move(util::HOST, util::DEVICE)) return retval;
            }

            if (this->belief_next.GetPointer(util::DEVICE) == NULL) {
                if (retval = this->belief_next.Allocate(nodes, util::DEVICE)) return retval;
                // copy data
                if (retval = this->belief_next.Move(util::HOST, util::DEVICE)) return retval;
            }

            if (this->joint_probabilities.GetPoint(util::DEVICE) == NULL) {
                if (retval = this->joint_probabilities.Allocate(edges, util::DEVICE)) retval;
                // copy data
                if (retval = this->joint_probabilities.Move(util::HOST, util::DEVICE)) return retval;
            }

            if (this -> labels    .GetPointer(util::DEVICE) == NULL) {
                if (retval = this->labels.Allocate(nodes, util::DEVICE)) return retval;
            }

            this->delta = delta;
            this->threshold = threshold;
            this->max_iter = max_iter;


            util::MemsetKernel<<<128, 128>>>(
                    this -> labels .GetPointer(util::DEVICE),
                            util::InvalidValue<VertexId>(),
                            nodes);

            //util::MemsetKernel<<<128, 128>>>(this->visit_lookup.GetPointer(util::DEVICE), (VertexId)-1, nodes);
            //util::MemsetKernel<<<128, 128>>>(sssp_marker.GetPointer(util::DEVICE), (int)0, nodes);
            return retval;
        }
    };

};


}
}
}