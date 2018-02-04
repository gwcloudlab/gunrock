/**
 * @file
 * bp_problem.cuh
 *
 * @brief GPU Storage management Structure for Belief Propagation Problem Data
**/

#pragma once

#include <cub/cub.cuh>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
    namespace app {
        namespace bp {
            template <typename SizeT, typename Value>
            __global__ void Assign_Init_Value_Kernel(
                    SizeT num_elements,
                    Value init_belief,
                    Value *d_belief_current
            )
            {
                SizeT x = (SizeT)(blockIdx.x * blockDim.x + threadIdx.x);
                const SizeT STRIDE = (SizeT)(blockDim.x * gridDim.x);

                int num_beliefs_init = init_belief.num_beliefs_x;

                while (x < num_elements) {
                    for(int y = 0; y < num_beliefs_init; y++) {
                        d_belief_current[x].states[y][0] = init_belief.states[y][0];
                    }

                    x += STRIDE;
                }
            };

            /**
             * @brief Belief propagation structure stores device-side vectors for doing BP on the GPU
             *
             * @tparam VertexId Type of signed integer to use as vertex id
             * @tparam SizeT Type of unsigned int to use for array indexing
             * @tparam Value Type of belief to use for computing beliefs
             */
            template <typename VertexId, typename SizeT, typename Value, bool NORMALIZED>
            struct BPProblem : ProblemBase<VertextId, SizeT, Value,
                    true, // _MARK_PREDECESSORS
                    false> // _ENABLE IDEMPOTENCE
                    //false, // _USE_DOUBLE_BUFFER
                    //false, // _ENABLE_BACKWARD
                    //false, // _KEEP_ORDER
                    //true> // _KEEP_NODE_NUM
            {
                static const bool MARK_PREDECESSORS = true;
                static const bool ENABLE_IDEMPOENTCE = false;
                static const int MAX_NUM_VERTEX_ASSOCIATES = 0;
                static const int MAX_NUM_VALUE__ASSOCIATES = 1;
                typedef ProblemBase<VertexId, SizeT, Value,
                        MARK_PREDECESSORS, ENABLE_IDEMPOENTCE> BaseProblem;
                typedef DataSliceBase<VertexId, SizeT, Value,
                        MAX_NUM_VERTEX_ASSOCIATES, MAX_NUM_VALUE__ASSOCIATES> BaseDataSlice;
                typedef unsigned char MaskT;


                // Helper structures

                /**
                 * @brief Data slice structure which contains BP problem specific data.
                 */
                struct DataSlice : BaseDataSlice {
                    util::Array1D<SizeT, Value> belief_curr;
                    util::Array1D<SizeT, Value> belief_next;
                    util::Array1D<SizeT, Value> joint_probabilities;
                    util::Array1D<SizeT, VertexId> node_ids;
                    util::Array1D<SizeT, VertexId> local_vertices;
                    util::Array1D<SizeT, VertexId> *remote_vertices_out;
                    util::Array1D<SizeT, VertexId> *remote_vertices_in;
                    float threshold;
                    float delta;
                    SizeT max_iter;
                    SizeT num_updated_vertices;
                    DataSlice *d_data_slice;
                    util::Array1D<int, SizeT> in_counters;
                    util::Array1D<int, SizeT> out_counters;
                    util::Array1D<SizeT, unsigned char> cub_sort_storage;
                    util::Array1D<SizeT, VertexId> temp_vertex;
                };

                /**
                 * @brief default constructor
                 */
                DataSlice() : BaseDataSlice(),
                              threshold(0),
                              delta(0),
                              max_iter(0),
                              num_updated_vertices(0),
                              d_data_slice(NULL)
                {
                    belief_curr.SetName("belief_curr");
                    belief_next.SetName("belief_next");
                    joint_probabilities.SetName("joint_probabilities");
                    node_ids.SetName("node_ids");
                    local_vertices.SetName("local_vertices");
                    in_counters.SetName("in_counters");
                    out_counters.SetName("out_counters");
                    cub_sort_storage.SetName("cub_sort_storage");
                    temp_vertex.SetName("temp_vertex");
                }

                /**
                 * @brief Default destructor
                 */
                virtual ~DataSlice()
                {
                    Release();
                }

                cudaError_t Release() {
                    cudaError_t retval = cudaSuccess;
                    if (retval = util::SetDevice(this->gpu_idx)) retval;
                    if (retval = BaseDataSlice::Release()) return retval;
                    if (retval = belief_curr.Release()) return retval;
                    if (retval = belief_next.Release()) return retval;
                    if (retval = joint_probabilties.Release()) return retval;
                    if (retval = node_ids.Release()) return retval;
                    if (retval = in_counters.Release()) return retval;
                    if (retval = out_counters.Release()) return retval;
                    if (retval = cub_sort_storage.Release()) return retval;
                    if (retval = temp_vertex.Release()) return retval;

                    if (remote_vertices_in != NULL) {
                        for (int peer = 0; peer < this->num_gpus; peer++) {
                            if (retval = remote_vertices_in[peer].Release()) return retval;
                            delete[] remote_vertices_in;
                            remote_vertices_in = NULL;
                        }
                    }

                    if (remote_vertices_out != NULL) {
                        for (int peer = 0; peer < this->num_gpus; peer++)
                        {
                            if (retval = remote_vertices_out[peer].Release()) return retval;
                            delete[] remote_vertices_out;
                            remote_vertices_out = NULL;
                        }
                    }

                    return retval;
                }

                /**
                 * @brief initialization function
                 *
                 * @param[in] num_gpus Number of GPUs used
                 * @param[in] gpu_idx GPU index used for testing
                 * @param[in] use_double_buffer Whether to use double buffer
                 * @param[in] graph Pointer to the graph we process on
                 * @param[in] graph_slice Pointer to GraphSlice object
                 * @param[in] num_in_nodes
                 * @param[in] num_out_nodes
                 * @param[in] queue_sizing Maximum queue sizing factor
                 * @param[in] in_sizing
                 *
                 * @return cudaError_t object Indicates the success of all CUDA calls
                 */
                cudaError_t Init(
                       int num_gpus,
                       int gpu_idx,
                       bool use_double_buffer,
                       Csr<VertexId, SizeT, Value> *graph,
                       GraphSlice<VertexId, SizeT, Value> *graph_slice,
                       SizeT *num_in_nodes,
                       SiztT *num_out_nodes,
                       float queue_sizing = 2.0,
                       float in_sizing = 1.0
                )
                {
                    cudaError_t retval = cudaSuccess;
                    SizeT nodes = graph->nodes;
                    SizeT edges = graph->edges;
                    if ( retval = BaseDetails::Init(
                            num_gpus,
                            gpu_idx,
                            use_double_buffer,
                            graph,
                            num_in_nodes,
                            num_out_nodes,
                            in_sizing
                    )){
                        return retval;
                    }

                    // create SoA on device
                    if (retval = belief_curr.Allocate(nodes, util::DEVICE)) return retval;
                    if (retval = belief_next.Allocate(nodes, util::DEVICE)) return retval;
                    if (retval = joint_probabilities.Allocate(edges, util::DEVICE)) return retval;

                    // copy data over
                    belief_curr.SetPointer(graph->node_values, graph->nodes, util::HOST);
                    if (retval = belief_curr.Move(util::HOST, util::DEVICE)) return retval;
                    belief_next.SetPointer(graph->node_values, graph->nodes, util::HOST);
                    if (retval = belief_next.Move(util::HOST, util::DEVICE)) return retval;
                    joint_probabilities.SetPointer(graph->edge_values, graph->edges, util::HOST);
                    if (retval = joint_probabilities.Move(util::HOST, util::DEVICE)) return retval;

                    if(this->num_gpus == 1)
                    {
                        if (retval = local_vertices.Allocate(nodes, util::DEVICE)) {
                            return retval;
                        }
                        util::MemsetIdxKernel<<<128, 128>>>(local_vertices.GetPointer(util::DEVICE), nodes);
                    }
                    else {
                        out_counters.Allocate(this->num_gpus, util::HOST);
                        in_counters.Allocate(this->num_gpus, util::HOST);
                        remote_vertices_out = new util::Array1D<SizeT, VertexId>[this->num_gpus];
                        remote_vertices_in = new util::Array1D<SizeT, VertexId>[this->num_gpus];
                        for (int peer = 0; peer < this->num_gpus; peer++) {
                            out_counters[peer] = 0;
                            remote_vertices_out[peer].SetName("remote_vertices_out[]");
                            remote_vertices_in[peer].SetName("remote_vertices_in[]");
                        }

                        for (VertexId v = 0; v < graph->nodes; v++) {
                            out_counters[graph_slice->partition_table[v]]++;
                        }

                        for (int peer = 0; peer < this->num_gpus; peer++) {
                            if (retval = remote_vertices_out[peer].Allocate(
                                    out_counters[peer], util::HOST | util::DEVICE
                            )) {
                                return retval;
                            }
                            out_counters[peer] = 0;
                        }

                        for (VertexId v = 0; v < graph->nodes; v++) {
                            int target = graph_slice->partition_table[v];
                            remote_vertices_out[target][out_counters[target]] = v;
                            out_counters[target]++;
                        }

                        for (int peer = 0; peer < this->num_gpus; peer++)
                        {
                            if (retval = remote_vertices_out[peer].Move(util::HOST, util::DEVICE))
                            {
                                return retval;
                            }
                        }
                        if (retval = local_vertices.SetPointer(
                                remote_vertices_out[0].GetPointer(util::HOST),
                                out_counters[0], util::HOST
                        )){
                            return retval;
                        }
                        if (retval = local_vertices.SetPointer(
                                remote_vertices_out[0].getPointer(util::DEVICE,
                                out_counters[0], util::DEVICE)
                        )){
                            return retval;
                        }

                        return retval;
                    }
                }

                cudaError_t Reset(
                        VertexId src,
                        Value delta,
                        Value threshold,
                        SizeT max_iter,
                        bool scaled,
                        FrontierType frontier_type,
                        Csr <VertexId, SizeT, Value> *org_graph,
                        Csr <VertexId, Size, Value> *sub_graph,
                        GraphSlice<VertexId, SizeT, Value> *graph_slice,
                        double queue_sizing = 2.0,
                        bool use_double_buffer = false,
                        double queue_sizing1 = -1.0,
                        bool skip_scanned_edges = false
                )
                {
                    cudaError_t retval = cudaSuccess;
                    if (retval = BaseDataSlice::Reset(
                            frontier_type,
                            grpah_slice,
                            queue_sizing,
                            use_double_buffer,
                            queue_sizing1,
                            skip_scanned_edges
                    )) {
                        return retval;
                    }

                    SizeT nodes = sub_graph->nodes;

                    if (this->num_gpus > 1) {
                        for (int peer = 0; peer < this->num_gpus; peer++)
                        {
                            if (retval = this->keys_out[peer].Release()) return retval;
                            if (retval = this->keys_in[0][peer].Release()) return retval;
                            if (retval = this->keys_in[1][peer].Release()) return retval;
                        }
                    }

                    if (belief_curr.GetPointer(util::DEVICE) == NULL) {
                        if (retval = belief_curr.Allocate(nodes, util::DEVICE)) return retval;
                    }
                    if (belief_next.GetPointer(util::DEVICE == NULL)) {
                        if (retval = belief_next.Allocate(nodes, util::DEVICE)) return retval;
                    }
                    if (joint_probabilities.GetPointer(util::DEVICE == NULL)) {
                        if (retval = joint_probabilities.Allocate(edges, util::DEVICE)) return retval;
                    }

                    // no need to init values

                    this->delta = data;
                    this->threshold = threshold;
                    this->to_continue = true;
                    this->max_iter = max_iter;
                    this->final_event_set = false;
                    this->num_updated_vertices = 1;

                    return retval;
                }

                // Members

                // Set of data slices
                util::Array1D<SizeT, DataSlice> *data_slices;

                // whether to use scaling feature
                bool scaled;

                /**
                 * @brief BPProblem default constructor
                 */
                BPProblem(bool _scaled) : BaseProblem(
                    false, // use_double_buffer
                    false, // enable_backward
                    true, // keep_node_num
                    false, // skip_makeout_selection
                    true // unified_receive
                ),
                data_slices(NULL),
                scaled (_scaled)
                {}

                /**
                 * @brief BPProblem default destructor
                 */
                virtual ~BPProblem()
                {
                    Release();
                }

                cudaError_t Release()
                {
                    cudaError_t retval = cudaSuccess;
                    if (data_slices == NULL) return retval;
                    for (int i = 0; i < this->num_gpus; ++i)
                    {
                        if (retval = util::SetDevice(this->gpu_idx[i])) return retval;
                        if (retval = data_slices[i].Release()) return retval;
                    }
                    delete[] data_slices;
                    data_slices = NULL;
                    if (retval = BaseProblem::Release()) return retval;
                    return retval;
                }

                /**
                 * @addtogroup PublicInterface
                 */

                /**
                 * @brief Copy result beliefs and/or predecessors computed on the GPU back to host-side vectors.
                 *
                 * @param[out] h_belief host-side vector to store beliefs
                 * @param[out] h_node_id host-side vector to store node Vertex ID
                 *
                 * @return cudaError_t object Indicates the success of all CUDA calls
                 */
                cudaError_t Extract(Value *h_belief, VertexId *h_node_id)
                {
                    cudaError_t retval = cudaSuccess;

                    if (retval = util::SetDevice(this->gpu_idx[0])) return retval;
                    data_slices[0]->belief_curr.SetPointer(h_belief);
                    if (retval = data_slices[0]->belief_curr.Move(util::DEVICE, util::HOST)) return retval;
                    data_slices[0]->node_ids.SetPointer(h_node_id);
                    if (retval = data_slices[0]->node_ids.Move(util::DEVICE, util::HOST)) return retval;

                    return retval;
                }

                /**
                 * @brief initialization function.
                 *
                 * @param[in] stream_from_host Whether to stream data from host
                 * @param[in] graph Pointer to the CSR graph object we process on. @see Csr
                 * @param[in] inversegraph Pointer to the inversed CSR graph object we process on.
                 * @param num_gpus Number of GPUs used.
                 * @param gpu_idx GPU index used for testing.
                 * @param partition_method Partition method to partition input graph.
                 * @param stream CUDA stream.
                 * @param context
                 * @param queue_sizing Maximum queue sizing factor.
                 * @param in_sizing
                 * @param partition_factor Partition factor used for partitioner.
                 * @param partition_seed Partition seed used for partitioner.
                 *
                 * @return cudaError_t object Indicates the success of all CUDA calls
                 */
                cudaError_t Init(
                        bool stream_from_host, // Only meaningful for single-GPU
                        Csr<VertexId, SizeT, Value> *graph,
                        Csr<VertexId, SizeT, Value> *inversegraph = NULL,
                        int num_gpus = 1,
                        int *gpu_idx = NULL,
                        std::string partition_method = "random",
                        cudaStream_t *stream = NULL,
                        ContextPtr *context = NULL,
                        float queue_sizing = 2.0f,
                        float in_sizing = 1.0f,
                        float partition_factor = -1.0f,
                        int partition_seed = -1
                ) {
                    cudaError_t retval = cudaSuccess;

                    if (retval = BaseProblem::Init(
                            stream_from_host,
                            graph,
                            inversegraph,
                            num_gpus,
                            gpu_idx,
                            partition_method,
                            queue_sizing,
                            partition_factor,
                            partition_seed
                    )) {
                        return retval;
                    }

                    data_slices = new util::Array1D<SizeT, DataSlice>(this->num_gpus);

                    for (int gpu = 0; gpu < this->num_gpus; gpu++)
                    {
                        data_slices[gpu].SetName("data_slices[]");
                        if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                        if (retval = this->graph_slices[gpu]->out_degrees.Release()) return retval;
                        if (retval = this->graph_slices[gpu]->original_vertex.Release()) return retval;
                        if (retval = this->graph_slices[gpu]->convertion_table.Release()) return retval;
                        if (retval = this->graph_slices[gpu].Allocate(1, util::DEVICE | util::HOST)) return retval;
                        DataSlice *data_slice_ = data_slices[gpu].GetPointer(util::HOST);
                        data_slice_->d_data_slice = data_slices[gpu].GetPointer(util::DEVICE);
                        data_slice_->streams.SetPointer(streams + gpu * num_gpus * 2, num_gpus * 2);

                        if (retval = data_slice_->Init(
                                this->num_gpus,
                                this->gpu_idx[gpu],
                                this->use_double_buffer,
                                &(this->sub_graphs[gpu]),
                                this->graph_slices[gpu],
                                this->num_gpus > 1 ? this->graph_slices[gpu]->in_counter.GetPointer(util::HOST) : NULL,
                                this->num_gpus > 1 ? this->graph_slices[gpu]->out_counter.GetPointer(util::HOST) : NULL,
                                queue_sizing,
                                in_sizing
                        )){
                            return retval;
                        }
                    }

                    if (this->num_gpus == 1) return retval;

                    for (int gpu = 0; gpu < this->num_gpus; gpu++)
                    {
                        if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;

                        for (int peer = 0; peer < this->num_gpus; peer++)
                        {
                            if (peer == gpu) continue;
                            int peer_ = (peer < gpu) ? peer + 1: peer;
                            int gpu_ = (peer < gpu) ? gpu: gpu + 1;
                            data_slices[gpu]->in_counters[peer_] = data_slices[peer]->out_counters[gpu_];
                            if (gpu != 0)
                            {
                                data_slices[gpu]->remote_vertices_in[peer_].SetPointer(
                                        data_slices[peer]->remote_vertices_out[gpu_].GetPointer(util::HOST),
                                        data_slices[peer]->remote_vertices_out[gpu_].GetSize(),
                                        util::HOST
                                );
                            } else {
                              data_slices[gpu]->remote_vertices_in[peer_].SetPointer(
                                      data_slices[peer]->remote_vertices_out[gpu_].GetPointer(util::HOST),
                                      max(data_slices[peer]->remote_vertices_out[gpu_].GetSize,
                                      data_slices[peer]->local_vertices.GetSize()),
                                      util::HOST
                              );
                            }
                            if (retval = data_slices[gpu]->remote_vertices_in[peer_].Move(util::HOST, util::DEVICE,
                            data_slices[peer]->remote_vertices_out[gpu_].GetSize())) {
                                return retval;
                            }

                            for (int t = 0; t < 2; t++)
                            {
                                if (data_slices[gpu]->value__associate_in[t][peer_].GetPointer(util::DEVICE) == NULL)
                                {
                                    if (retval = data_slices[gpu]->value__associate_in[t][peer_].Allocate(
                                            data_slices[gpu]->in_counters[peer_], util::DEVICE
                                    )) {
                                        return retval;
                                    }
                                }
                                else {
                                    if (retval = data_slices[gpu]->value__associate_in[t][peer_].EnsureSize(
                                            data_slices[gpu]->in_counters[peer_]
                                    )){
                                        return retval;
                                    }
                                }
                            }
                        }
                    }

                    for (int gpu = 1; gpu < this->num_gpus; gpu++)
                    {
                        if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                        if (data_slices[gpu]->value__associate_out[1].GetPointer(util::DEVICE) == NULL)
                        {
                            if (retval = data_slices[gpu]->value_associate_out[1].Allocate(
                                    data_slices[gpu]->local_vertices.GetSize(), util::DEVICE
                            )) {
                                return retval;
                            }
                        }
                        else {
                            if (retval = data_slices[gpu]->value__associate_out[1].EnsureSize(
                                    data_slices[gpu]->local_vertices.GetSize()
                            )) {
                                return retval;
                            }
                        }
                    }

                    if (retval = util::SetDevice(this->gpu_idx[0])) return retval;

                    for (int gpu = 1; gpu < this->num_gpus; gpu++) {
                        if (data_slices[0]->value__associate_in[0][gpu].GetPointer(util::DEVICE) == NULL)
                        {
                            if (retval = data_slices[0]->value__associate_in[0][gpu].Allocate(
                                    data_slices[gpu]->local_vertices.GetSize(), util::DEVICE
                            )) {
                                return retval;
                            }
                        }
                        else {
                            if (retval = data_slices[0]->value__associate_in[0][gpu].EnsureSize(
                                    data_slices[gpu]->local_vertices.GetSize()
                            )) {
                                return retval;
                            }
                        }
                    }

                    return retval;
                }

                /**
                 * @brief Reset problem function. Must be called prior to each run
                 *
                 * @param[in] src Source node to start.
                 * @param[in] delta Delta factor
                 * @param[in] threshold Threshold for remove node from BP computation process.
                 * @param[in] max_iter Maximum number of iterations.
                 * @param[in] frontier_type The frontier type (i.e. edge/vertex/mixed)
                 * @param[in] queue_sizing Size scaling factor for work queue allocation (e.g. 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
                 * @param queue_sizing1
                 * @param skip_scanned_edges Whether to skip scanned edges
                 *
                 * @return cudaError_t object Indicates the success of all CUDA calls
                 */
                cudaError_t Reset(
                        VertexId src,
                        Value delta,
                        Value threshold,
                        SizeT max_iter,
                        FrontierType frontier_type,
                        double queue_sizing,
                        double queue_sizing1 = -1.0,
                        bool skip_scanned_edges = false
                ) {
                    cudaError_t retval = cudaSuccess;

                    for (int gpu = 0; gpu < this->num_gpus; ++gpu)
                    {
                        // Set device
                        if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                        if (retval = data_slices[gpu]->Reset(
                                src, delta, threshold, max_iter, scaled,
                                frontier_type, this->org_graph,
                                this->sub_graphs + gpu, this->graph_slices[gpu],
                                queue_sizing, this->use_double_buffer,
                                queue_sizing1, skip_scanned_edges
                        )){
                            return retval;
                        }

                        if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) {
                            return retval;
                        }

                        if (gpu == 0 && this->num_gpus > 1) {
                            for (int peer = 1; peer < this->num_gpus; peer++)
                            {
                                if (retval = data_slices[gpu]->remote_vertices_in[peer].Move(util::HOST, util::DEVICE,
                                data_slices[gpu]->in_counters[peer])) {
                                    return retval;
                                }
                            }
                        }
                    }

                    return retval;
                }
            };
        }
    }
}