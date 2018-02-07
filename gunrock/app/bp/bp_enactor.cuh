
/**
 * @file bp_enactor.cuh
 *
 * @brief BP Problem Enactor
*/

#pragma once

#include <thread>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/sort_utils.cuh>
#include <gunrock/util/sharedmem.cuh>
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>
#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/bp/bp_problem.cuh>
#include <gunrock/app/bp/bp_functor.cuh>
#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
    namespace app {
        namespace bp {


            template <typename VertexId, typename SizeT, typename Value>
            __global__ void UpdateBeliefs(
                    const SizeT num_elements,
                    const VertexId* const keys,
                    const SizeT* const markers,
                    const Value* const belief_next,
                          Value* belief_curr
            ) {
                const SizeT STRIDE = (SizeT)(gridDim.x * blockDim.x);
                SizeT x = (SizeT)(blockIdx.x * blockDim.x + threadIdx.x);
                while (x < num_elements) {
                    VertexId key = keys[x];
                    if (markers[key] == 1) {
                        belief_curr[key] = belief_next[key];
                    }
                    x += STRIDE;
                }
            };

            /**
             * @brief Expand incoming function.
             *
             * @tparam VertexId
             * @tparam SizeT
             * @tparam Value
             * @tparam NUM_VERTEX_ASSOCIATES
             * @tparam NUM_VALUE__ASSOCIATES
             *
             * @param num_elements
             * @param keys_in
             * @param array_size
             * @param array
             * @param markers
             */
            template <
                    typename VertexId,
                    typename SizeT,
                    typename Value,
                    int      NUM_VERTEX_ASSOCIATES,
                    int      NUM_VALUE__ASSOCIATES>
            __global__ void Expand_Incoming_BP (
                    const SizeT num_elements,
                    const VertexId* const keys_in,
                    const size_t array_size,
                    char* array,
                    SizeT* markers
            ){
               extern __shared__ char s_array[];
                const SizeT STRIDE = (SizeT)(gridDim.x * blockDim.x);
                size_t offset = 0;
                offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
                Value* s_value__associate_in = (Value**)&(s_array[offset]);
                offset += sizeof(Value*) * NUM_VALUE__ASSOCIATES;
                offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
                Value** s_value__associate_org = (Value**)&(s_array[offset]);
                SizeT x = threadIdx.x;
                while (x < array_size)
                {
                    s_array[x] = array[x];
                    x += blockDim.x;
                }
                __syncthreads();

                x = (SizeT)(blockIdx.x * blockDim.x + threadIdx.x);
                while (x < num_elements)
                {
                    VertexId key = keys_in[x];
                    Value mul_values_new = s_value__associate_in[0][x];
                    Value old_value = atomicMul(s_value__associate_org[0] + key, mul_values_new);
                    //Value old_value = atomicMul(s_value__associate_org[0] + key, mul_value);
                    markers[key] = 1;
                    x += STRIDE;
                }
            };

            template <typename KernelPolicy, int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
            __global__ void Expand_Incoming_BP_Kernel(
                    int thread_num,
                    typename KernelPolicy::SizeT num_elements,
                    typename KernelPolicy::VertexId *d_keys,
                    typename KernelPolicy::Value *d_belief_in,
                    typename KernelPolicy::Value *d_belief_out
            ){
                typedef typename KernelPolicy::VertexId VertexId;
                typedef typename KernelPolicy::SizeT SizeT;
                typedef typename KernelPolicy::Value Value;

                SizeT x = (SizeT)(blockIdx.x * blockDim.x + threadIdx.x);
                const SizeT STRIDE = (SizeT)(gridDim.x * blockDim.x);

                while (x < num_elements)
                {
                    VertexId key = d_keys[x];
                    Value mul_value = d_belief_in[x];
                    Value old_value = atomicMul(d_belief_out + key, mul_value);
                    x += STRIDE;
                }
            };

            template<typename VertexId, typename SizeT>
            __global__ void Assign_Marker_BP(
                    const SizeT num_elements,
                    const int peer_,
                    const SizeT* markers,
                    const int* partition_table,
                    SizeT* key_markers
            ){
                int gpu = 0;
                SizeT x = (SizeT)(blockIdx.x * blockDim.x + threadIdx.x);
                const SizeT STRIDE = (SizeT)(gridDim.x * blockDim.x);

                while (x < num_elements)
                {
                    gpu = partition_table[x];
                    if ((markers[x] == 1 || gpu == 0) && (gpu == peer_))
                    {
                        key_markers[x] = 1;
                    }
                    else
                    {
                        key_markers[x] = 0;
                    }
                    x += STRIDE;
                }
            };

            template<typename VertexId, typename SizeT>
            __global__ void Assign_Keys_BP (
                    const SizeT num_elements,
                    const int peer_,
                    const int* partition_table,
                    const SizeT* markers,
                    SizeT* keys_marker,
                    VertexId* keys_out
            ){
                const SizeT STRIDE = (SizeT)(gridDim.x * blockDim.x);
                SizeT x = (SizeT)(blockIdx.x * blockDim.x + threadIdx.x);

                while (x < num_elements)
                {
                    int gpu = partition_table[x];
                    if ((markers[x] == 1 || gpu == 0) && (gpu == peer_))
                    {
                        SizeT pos = keys_marker[x] - 1;
                        keys_out[pos] = x;
                    }
                }
                x += STRIDE;
            };


            template<typename VertexId, typename SizeT, typename Value>
            __global__ void Assign_Values_BP (
                    const SizeT num_elements,
                    const VertexId* const keys_out,
                    const Value* const belief_next,
                    Value* belief_out
            ){
                const SizeT STRIDE = (SizeT)(gridDim.x * blockDim.x);
                SizeT x = (SizeT)(blockIdx.x * blockDim.x + threadIdx.x);

                while (x < num_elements)
                {
                    VertexId key = keys_out[x];
                    belief_out[x] = belief_next[key];
                    x += STRIDE;
                }
            };

            template<typename VertexId, typename SizeT, typename Value>
            __global__ void Expand_Incoming_Final (
                    const SizeT num_elements,
                    const VertexId* const keys_in,
                    Value* const beliefs_in,
                    Value* beliefs_out
            )
            {
                const SizeT STRIDE = (SizeT)(gridDim.x * blockDim.x);
                SizeT x = (SizeT)(blockIdx.x * blockDim.x + threadIdx.x);
                while (x < num_elements) {
                    VertexId key = keys_in[x];
                    beliefs_out[key] = beliefs_in[x];
                    x += STRIDE;
                }
            };

            template<typename VertexId, typename SizeT, typename Value>
            __global__ void Assign_Final_Value_Kernel(
                    SizeT num_elements,
                    VertexId *d_local_vertices,
                    Value *d_belief_current,
                    Value *d_belief_out
            ){
                const SizeT STRIDE = (SizeT)(gridDim.x * blockDim.x);
                SizeT x = (SizeT)(blockIdx.x * blockDim.x + threadIdx.x);
                while (x < num_elements)
                {
                    VertexId key = d_local_vertices[x];
                    Value belief = d_belief_current[key];
                    if (d_belief_out != NULL)
                    {
                        d_belief_out[x] = belief;
                    }
                    else
                    {
                        d_belief_current[key] = belief;
                    }
                }
            };

            template <typename AdvanceKernelPolicy, typename FilterKernelPolicy, typename Enactor>
            struct BPIteration : public IterationBase<
                    AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
                            false, //HAS_SUBQ
                            true, //HAS_FULLQ
                            false, //BACKWARD
                            true, //FORWARD
                            false //UPDATE PREDECESSORS
                    >
            {
            public:
                typedef typename Enactor::SizeT SizeT;
                typedef typename Enactor::Value Value;
                typedef typename Enactor::VertexId VertexId;
                typedef typename Enactor::Problem Problem;
                typedef typename Problem::DataSlice DataSlice;
                typedef GraphSlice<VertexId, SizeT, Value> GraphSliceT;
                typedef BPFunctor<VertexId, SizeT, Value, Problem> BpFunctor;
                typedef BPMarkerFunctor<VertexId, SizeT, Value, Problem> BpMarkerFunctor;
                typedef typename util::DoubleBuffer<VertexId, SizeT, Value> Frontier;
                typedef IterationBase <AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
                        false, true, false, true, false> BaseIteration;

                /**
                 * @brief FullQueue_Core function.
                 *
                 * @param[in] thread_num Number of threads.
                 * @param[in] peer_ Peer GPU index.
                 * @param[in] frontier_queue Pointer to the frontier queue.
                 * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
                 * @param[in] frontier_attribute Pointer to the frontier attribute.
                 * @param[in] enactor_stats Pointer to the enactor statistics.
                 * @param[in] data_slice Pointer to the data slice we process on.
                 * @param[in] d_data_slice Pointer to the data slice on the device.
                 * @param[in] graph_slice Pointer to the graph slice we process on.
                 * @param[in] work_progress Pointer to the work progress class.
                 * @param[in] context CudaContext for ModernGPU API.
                 * @param[in] stream CUDA stream.
                 */
                static void FullQueue_Core(
                    Enactor *enactor,
                    int thread_num,
                    int peer_,
                    Frontier *frontier_queue,
                    util::Array1D<SizeT, SizeT> *scanned_edges,
                    FrontierAttribute<SizeT> *frontier_attribute,
                    EnactorStats<SizeT> *enactor_stats,
                    DataSlice *data_slice,
                    DataSlice *d_data_slice,
                    GraphSliceT *graph_slice,
                    util::CtaWorkProgressLifetime<SizeT> *work_progress,
                    ContextPtr context,
                    cudaStream_t stream
                )
                {
                    if (enactor_stats->iteration != 0)
                    {
                        frontier_attribute->queue_length = data_slice->local_vertices.GetSize();
                        enactor_stats->nodes_queued[0] += frontier_attribute->queue_length;
                        frontier_attribute->queue_reset = true;

                        if (enactor->debug)
                        {
                            util::cpu_mt::PrintMessage("Filter start.",
                            thread_num, enactor_stats->iteration, peer_);

                            gunrock::oprtr::filter::LaunchKernel
                            <FilterKernelPolicy, Problem, BpFunctor>(
                                    enactor_stats[0],
                                    frontier_attribute[0],
                                    typename BpFunctor::LabelT(),
                                    data_slice,
                                    d_data_slice,
                                    NULL,
                                    (unsigned char *)NULL,
                                    data_slice->local_vertices.GetPointer(util::DEVICE),
                                    (VertexId*)NULL,
                                    (Value *)NULL,
                                    (Value *)NULL,
                                    data_slice->local_vertices.GetSize(),
                                    graph_slice->nodes,
                                    work_progress[0],
                                    context[0],
                                    stream,
                                    util::MaxValue<SizeT>(),
                                    util::MaxValue<SizeT>(),
                                    enactor_stats->filter_kernel_stats,
                                    false
                            );
                            if (enactor->debug)
                            {
                                util::cpu_mt::PrintMessage("Filter end.",
                                thread_num, enactor_stats->iteration, peer_);
                            }
                            frontier_attribute->queue_index++;
                            if (enactor_stats->retval = work_progress->GetQueueLength(
                                    frontier_attribute->queue_index,
                                    frontier_attribute->queue_length,
                                    false, stream
                            )){
                                return;
                            }

                            util::MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
                                    data_slice->belief_curr.GetPointer(util::DEVICE),
                                    data_slice->belief_next.GetPointer(util::DEVICE),
                                    data_slice->nodes
                            );

                            if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream),
                            "cudaStreamSynchronize failed", __FILE__, __LINE__)) {
                                return;
                            }
                            data_slice->num_updated_vertices = frontier_attribute->queue_length;
                        }

                        frontier_attribute->queue_length = data_slice->local_vertices.GetSize();

                        if (enactor->debug)
                        {
                            util::cpu_mt::PrintMessage("Advance Bpfunctor start.",
                            thread_num, enactor_stats->iteration, peer_);
                        }

                        // Edge Map
                        frontier_attribute->queue_reset = false;
                        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem, BpFunctor,
                                gunrock::oprtr::advance::V2V>(
                            enactor_stats[0],
                            frontier_attribute[0],
                            typename BpFunctor::LabelT(),
                            data_slice,
                            d_data_slice,
                            (VertexId*)NULL,
                            (bool*)NULL,
                            (bool*)NULL,
                            scanned_edges->GetPointer(util::DEVICE),
                            data_slice->local_vertices.GetPointer(util::DEVICE),
                            (VertexId*)NULL,
                            (Value*)NULL,
                            (Value*)NULL,
                            graph_slice->row_offsets.GetPointer(util::DEVICE),
                            graph_slice->column_indices.GetPointer(util::DEVICE),
                            (SizeT*)NULL,
                            (VertexId*)NULL,
                            graph_slice->nodes,
                            graph_slice->edges,
                            work_progress[0],
                            context[0],
                            stream,
                            false,
                            false,
                            false
                                );

                        enactor_stats->edges_queued[0] += graph_slice->edges;
                    }
                }

                static cudaError_t Compute_OutputLength(
                        Enactor *enactor,
                        FrontierAttribute<SizeT> *frontier_attribute,
                        SizeT *d_offsets,
                        VertexId *d_indices,
                        SizeT *d_inv_offsets,
                        VertexId *d_inv_indices,
                        VertexId *d_in_key_queue,
                        util::Array1D<SizeT, SizeT> *partitioned_scanned_edges,
                        SizeT max_in,
                        SizeT max_out,
                        CudaContext &context,
                        cudaStream_t stream,
                        gunrock::oprtr::advance::TYPE  ADVANCE_TYPE,
                        bool express = false,
                        bool in_inv = false,
                        bool out_inv = false
                )
                {
                    cudaError_t retval = cudaSuccess;
                    return retval;
                }

                /**
                 * @brief Check frontier queue size function.
                 *
                 * @param[in] thread_num Number of threads.
                 * @param[in] peer_ Peer GPU index.
                 * @param[in] request_length Request frontier queue length.
                 * @param[in] frontier_queue Pointer to the frontier queue.
                 * @param[in] frontier_attribute Pointer to the frontier attribute.
                 * @param[in] enactor_stats Pointer to the enactor statistics.
                 * @param[in] graph_slice Pointer to the graph slice we process on.
                 */
                static void Check_Queue_Size(
                        Enactor *enactor,
                        int thread_num,
                        int peer_,
                        SizeT request_length,
                        Frontier *frontier_queue,
                        FrontierAttribute<SizeT> *frontier_attribute,
                        EnactorStats<SizeT> *enactor_stats,
                        GraphSliceT *graph_slice
                )
                {
                    return; // no need to check queue size for BP
                }

                template <int NUM_VERTEX_ASSSOCIATES, int NUM_VALUE__ASSOCIATES>
                static void Expand_Incoming_Old(
                        Enactor *enactor,
                        int grid_size,
                        int block_size,
                        size_t shared_size,
                        cudaStream_t stream,
                        SizeT &num_elements,
                        const VertexId* const keys_in,
                        util::Array1D<SizeT, VertexId>* keys_out,
                        const size_t array_size,
                        char* array,
                        DataSlice* data_slice
                ){
                    Expand_Incoming_BP<VertexId , SizeT, Value, NUM_VERTEX_ASSSOCIATES, NUM_VALUE__ASSOCIATES>
                            <<<grid_size, block_size, shared_size, stream>>> (
                            num_elements,
                            keys_in,
                            array_size,
                            array,
                            data_slice->markers.GetPointer(util::DEVICE));
                    num_elements = 0;
                }

                template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
                static void Expand_Incoming(
                        Enactor *enactor,
                        cudaStream_t  stream,
                        VertexId iteration,
                        int peer_,
                        SizeT received_length,
                        SizeT num_elements,
                        util::Array1D<SizeT, SizeT> &out_length,
                        util::Array1D<SizeT, VertexId> &keys_in,
                        util::Array1D<SizeT, VertexId> &vertex_associate_in,
                        util::Array1D<SizeT, Value> &value__associate_in,
                        util::Array1D<SizeT, VertexId> &keys_out,
                        util::Array1D<SizeT, VertexId*> &vertex_associate_orgs,
                        util::Array1D<SizeT, Value*> &value__associate_orgs,
                        DataSlice  *h_data_slice,
                        EnactorStats<SizeT> *enactor_stats
                ){
                    int num_blocks = num_elements / AdvanceKernelPolicy::THREADS / 2 + 1;
                    if (num_blocks > 240) {
                        num_blocks = 240;
                    }
                    Expand_Incoming_BP_Kernel<AdvanceKernelPolicy, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
                    <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>
                    (h_data_slice->gpu_idx,
                    h_data_slice->in_counters[peer_],
                    h_data_slice->remote_vertices_in[peer_].GetPointer(util::DEVICE),
                    value__associate_in.GetPointer(util::DEVICE),
                    h_data_slice->belief_next.GetPointer(util::DEVICE)
                    );
                }

                /**
                 * @brief Iteration_Update_Preds function.
                 *
                 * @param[in] graph_slice Pointer to the graph slice we process on.
                 * @param[in] data_slice Pointer to the data slice we process on.
                 * @param[in] frontier_attribute Pointer to the frontier attribute.
                 * @param[in] frontier_queue Pointer to the frontier queue.
                 * @param[in] num_elements Number of elements.
                 * @param[in] stream CUDA stream.
                 */
                static void Iteration_Update_Preds(
                        Enactor *enactor,
                        GraphSliceT *graph_slice,
                        DataSlice *data_slice,
                        FrontierAttribute<SizeT> *frontier_attribute,
                        Frontier *frontier_queue,
                        SizeT num_elements,
                        cudaStream_t stream
                ){}

                /**
                 * @brief Make_Output function.
                 *
                 * @tparam NUM_VERTEX_ASSOCIATES
                 * @tparam NUM_VALUE__ASSOCIATES
                 *
                 * @param[in] thread_num Number of threads.
                 * @param[in] num_elements
                 * @param[in] num_gpus Number of GPUs used.
                 * @param[in] frontier_queue Pointer to the frontier queue.
                 * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
                 * @param[in] frontier_attribute Pointer to the frontier attribute.
                 * @param[in] enactor_stats Pointer to the enactor statistics.
                 * @param[in] data_slice Pointer to the data slice we process on.
                 * @param[in] graph_slice Pointer to the graph slice we process on.
                 * @param[in] work_progress Pointer to the work progress class.
                 * @param[in] context CudaContext for ModernGPU API.
                 * @param[in] stream CUDA stream.
                 */

                template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
                static void Make_Output(
                    Enactor *enactor,
                    int thread_num,
                    SizeT num_elements,
                    int num_gpus,
                    Frontier *frontier_queue,
                    util::Array1D<SizeT, SizeT> *scanned_edges,
                    FrontierAttribute<SizeT> *frontier_attribute,
                    EnactorStats<SizeT> *enactor_stats,
                    util::Array1D<SizeT, DataSlice> *data_slice_,
                    GraphSliceT *graph_slice,
                    util::CtaWorkProgressLifetime<SizeT> *work_progress,
                    ContextPtr context,
                    cudaStream_t stream
                ){
                    DataSlice *data_slice = data_slice_->GetPointer(util::HOST);
                    cudaStream_t *streams = data_slice->streams + num_gpus;
                    if (num_gpus < 2) {
                        return;
                    }

                    for (int peer_ = 1; peer_ < num_gpus; peer_++)
                    {
                        data_slice->out_length[peer_] = data_slice->remote_vertices_out[peer_].GetSize();
                        int num_blocks = data_slice->out_length[peer_] / 512 + 1;
                        if (num_blocks > 480) {
                            num_blocks = 480;
                        }
                        Assign_Values_BP<<<num_blocks, 512, 0, streams[peer_]>>>(
                                data_slice->out_length[peer_],
                                data_slice->remote_vertices_out[peer_].GetPointer(util::DEVICE),
                                data_slice->belief_next.GetPointer(util::DEVICE),
                                data_slice->value__associate_outs[peer_]
                        );
                    }
                }

                /**
                 * @brief Stop_Condition check function.
                 *
                 * @param[in] enactor_stats Pointer to the enactor statistics.
                 * @param[in] frontier_attribute Pointer to the frontier attribute.
                 * @param[in] data_slice Pointer to the data slice we process on.
                 * @param[in] num_gpus Number of GPUs used.
                 */
                static bool Stop_Condition(
                        EnactorStats<SizeT> *enactor_stats,
                        FrontierAttribute<SizeT> *frontier_attribute,
                        util::Array1D<SizeT, DataSlice> *data_slice,
                        int num_gpus
                ){
                    bool all_zero = true;
                    for (int gpu = 0; gpu < num_gpus*num_gpus; gpu++)
                    {
                        if (enactor_stats[gpu].retval != cudaSuccess) {
                            return true;
                        }
                    }

                    for (int gpu =0; gpu < num_gpus; gpu++){
                        if (data_slice[gpu]->num_updated_vertices)
                        {
                            all_zero = false;
                        }
                    }

                    if (all_zero) {
                        return true;
                    }

                    for (int gpu = 0; gpu < num_gpus; gpu++)
                    {
                        if (enactor_stats[gpu * num_gpus].iteration < data_slice[0]->max_iter)
                        {
                            return false;
                        }
                    }

                    return true;
                }

                template<int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
                static void Make_Output_Old(
                    Enactor *enactor,
                    int thread_num,
                    SizeT num_elements,
                    int num_gpus,
                    Frontier *frontier_queue,
                    util::Array1D<SizeT, SizeT> *scanned_edges,
                    FrontierAttribute<SizeT> *frontier_attribute,
                    EnactorStats<SizeT> *enactor_stats,
                    util::Array1D<SizeT, DataSlice> *data_slice,
                    GraphSliceT *graph_slice,
                    util::CtaWorkProgressLifetime<SizeT> *work_progress,
                    ContextPtr context,
                    cudaStream_t stream
                ){
                    int peer_ = 0;
                    int block_size = 512;
                    int grid_size = graph_slice->nodes / block_size;
                    if ((graph_slice->nodes % block_size) != 0) {
                        grid_size++;
                    }
                    if (grid_size > 512) {
                        grid_size = 512;
                    }

                    if (num_gpus > 1 && enactor_stats->iteration == 0)
                    {
                        SizeT temp_length = data_slice[0]->out_length[0];
                        for (peer_ = 0; peer_ < num_gpus; peer_++)
                        {
                            util::MemsetKernel<<<128, 128, 0, stream>>> (
                                    data_slice[0]->keys_marker[0].GetPointer(util::DEVICE),
                                            (SizeT)0, graph_slice->nodes
                                                                        );

                            Assign_Marker_BP<<<grid_size, block_size, 0, stream>>> (
                                    graph_slice->nodes,
                                    peer_,
                                    data_slice[0]->markers.GetPointer(util::DEVICE),
                                    graph_slice->partition_table.GetPointer(util::DEVICE),
                                    data_slice[0]->keys_marker[0].GetPointer(util::DEVICE)
                                                                                   );

                            Scan<mgpu::MgpuScanTypeInc>(
                                    (SizeT*)(data_slice[0]->keys_marker[0].GetPointer(util::DEVICE)),
                                    graph_slice->nodes,
                                    (SizeT)0,
                                    mgpu::plus<SizeT>(),
                                    (SizeT*)0,
                                    (SizeT*)0,
                                    (SizeT*)(data_slice[0]->keys_marker[0].GetPointer(util::DEVICE)),
                                    context[0]
                            );

                            if (graph_slice->nodes > 0)
                            {
                                cudaMemcpyAsync(
                                        &data_slice[0]->out_length[peer_],
                                        data_slice[0]->keys_marker[0].GetPointer(util::DEVICE) + (graph_slice->nodes - 1),
                                        sizeof(SizeT), cudaMemcpyDeviceToHost, stream
                                );
                            }
                            else {
                                if (peer_ > 0) {
                                    data_slice[0]->out_length[peer_] = 0;
                                }
                            }
                            if (enactor_stats->retval = cudaStreamSynchronize(stream)) {
                                return;
                            }

                            bool over_sized = false;
                            if (peer_ > 1) {
                                data_slice[0]->keys_out[peer_] = data_slice[0]->temp_keys_out[peer_];
                                data_slice[0]->temp_keys_out[peer_] = util::Array1D<SizeT, VertexId>();
                            }
                            if (enactor_stats->retval = Check_Size<SizeT, VertexId>(
                                    enactor->size_check, "keys_out",
                                    data_slice[0]->out_length[peer_],
                                    &data_slice[0]->keys_out[peer_],
                                    over_sized, thread_num, enactor_stats->iteration, peer_
                            )){
                                return;
                            }
                            if (peer_ > 0) {
                                if (enactor_stats->retval = Check_Size<SizeT, Value>(
                                        enactor->size_check, "values_out",
                                        data_slice[0]->out_length[peer_],
                                        &data_slice[0]->value__associate_out[peer_][0],
                                        over_sized, thread_num, enactor->iteration, peer_
                                )){
                                    return;
                                }
                            }
                            data_slice[0]->keys_outs[peer_] = data_slice[0]->keys_out[peer_].GetPointer(util::DEVICE);
                            data_slice[0]->value__associate_outs[peer_][0] = data_slice[0]->value__associate_out[peer_][0].GetPointer(util::DEVICE);
                            data_slice[0]->value__associate_outs[peer_].Move(util::HOST, util::DEVICE, -1, 0, stream);

                            Assign_Keys_BP<VertexId , SizeT>
                                    <<<grid_size, block_size, 0, stream>>>
                                    (
                                            graph_slice->nodes,
                                            peer_,
                                            graph_slice->partition_table.GetPointer(util::DEVICE),
                                            data_slice[0]->markers.GetPointer(util::DEVICE),
                                            data_slice[0]->keys_marker[0].GetPointer(util::DEVICE),
                                            data_slice[0]->keys_out[peer_].GetPointer(util::DEVICE)
                                    );
                        }
                        data_slice[0]->keys_outs.Move(util::HOST, util::DEVICE, -1, 0, stream);
                    }

                    for (peer_ = 1; peer_ < num_gpus; peer_++)
                    {
                        Assign_Values_BP<VertexId , SizeT, Value>
                                <<<grid_size, block_size, 0, stream>>>(
                                data_slice[0]->out_length[peer_],
                                data_slice[0]->keys_out[peer_].GetPointer(util::DEVICE),
                                data_slice[0]->belief_next.GetPointer(util::DEVICE),
                                data_slice[0]->value__associate_out[peer_][0].GetPointer(util::DEVICE)
                        );
                    }
                    frontier_attribute->selector = data_slice[0]->BP_queue_selector;
                    if (enactor_stats->retval = cudaStreamSynchronize(stream)) {
                        return;
                    }
                }
            };

            template<typename AdvanceKernelPolicy, typename FilterKernelPolicy, typename Enactor>
            static CUT_THREADPROC BPThread(
                    void *thread_data_
            )
            {
                typedef typename Enactor::Problem Problem;
                typedef typename Enactor::SizeT SizeT;
                typedef typename Enactor::VertexId VertexId;
                typedef typename Enactor::Value Value;
                typedef typename Problem::DataSlice DataSlice;
                typedef GraphSlice<VertexId, SizeT, Value> GraphSliceT;
                typedef BPFunctor<VertexId , SizeT, Value, Problem> Functor;
                ThreadSlice *thread_data = (ThreadSlice*) thread_data_;
                Problem *problem = (Problem*) thread_data->problem;
                Enactor *enactor = (Enactor*) thread_data->enactor;

                int num_gpus = problem->num_gpus;
                int thread_num = thread_data->thread_num;
                int gpu_idx = problem->gpu_idx[thread_num];
                DataSlice *data_slice = problem->data_slices[thread_num].GetPointer(util::HOST);
                FrontierAttribute<SizeT> *frontier_attribute = &(enactor->frontier_attribute[thread_num * num_gpus]);
                EnactorStats<SizeT> *enactor_stats = &(enactor->enactor_stats[thread_num * num_gpus]);
                int *markers = new int[num_gpus];

                if (enactor_stats[0].retval = util::SetDevice(gpu_idx))
                {
                    thread_data->status = ThreadSlice::Status::Ended;
                    CUT_THREADEND;
                }

                thread_data->status = ThreadSlice::Status::Idle;
                while (thread_data->status != ThreadSlice::Status::ToKill)
                {
                    while (thread_data->status == ThreadSlice::Status::Wait ||
                            thread_data->status == ThreadSlice::Status::Idle) {
                        sleep(0);
                    }
                    if (thread_data->status == ThreadSlice::Status::ToKill) {
                        break;
                    }

                    for (int peer = 0; peer < num_gpus; peer++)
                    {
                        frontier_attribute[peer].queue_length = peer == 0 ? data_slice->local_vertices.GetSize() : 0;
                        frontier_attribute[peer].queue_index = 0;
                        frontier_attribute[peer].selector = 0;
                        frontier_attribute[peer].queue_reset = true;
                        enactor_stats[peer].iteration = 0;
                    }

                    if (num_gpus > 1)
                    {
                        data_slice->value__associate_orgs[0] = data_slice->belief_next.GetPointer(util::DEVICE);
                        data_slice->value__associate_orgs.Move(util::HOST, util::DEVICE);
                    }

                    gunrock::app::Iteration_Loop<Enactor, Functor,
                    BPIteration<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>, 0, 1>(thread_data);

                    thread_data->status = ThreadSlice::Status::Idle;
                }

                thread_data->status = ThreadSlice::Status::Idle;
                CUT_THREADEND;
            };

            template<typename _Problem>
            class BPEnactor :
                public EnactorBase<typename _Problem::SizeT> {
                // Members
                ThreadSlice *thread_slices;
                CUTThread *thread_Ids;

            public:
                _Problem *problem;
                typedef _Problem Problem;
                typedef typename Problem::SizeT SizeT;
                typedef typename Problem::VertexId VertexId;
                typedef typename Problem::Value Value;
                typedef EnactorBase<SizeT> BaseEnactor;
                typedef BPEnactor Enactor;
                typedef GraphSlice<VertexId, SizeT, Value> GraphSliceT;

                /**
                 * @addtogroup PublicInterface
                 */

                /**
                 * @brief BPEnactor constructor
                 */
                BPEnactor(
                        int num_gpus = 1,
                        int *gpu_idx = NULL,
                        bool instrument = false,
                        bool debug = false,
                        bool size_check = true) :
                        BaseEnactor(VERTEX_FRONTIERS, num_gpus, gpu_idx,
                                    instrument, debug, size_check),
                        thread_slices(NULL),
                        thread_Ids(NULL),
                        problem(NULL) {}

                /**
                 * @brief BPEnactor destructor
                 */
                virtual ~BPEnactor() {
                    Release();
                }

                cudaError_t Release() {
                    cudaError_t retval = cudaSuccess;
                    if (thread_slices != NULL) {
                        for (int gpu = 0; gpu < this->num_gpus; gpu++) {
                            thread_slices[gpu].status = ThreadSlice::Status::ToKill;
                        }
                        cutWaitForThreads(thread_Ids, this->num_gpus);
                        delete[] thread_Ids;
                        thread_Ids = NULL;
                        delete[] thread_slices;
                        thread_slices = NULL;
                    }
                    if (retval = BaseEnactor::Release()) return retval;
                    problem = NULL;

                    return retval;
                }

                /**
                 * @addtogroup PublicInterface
                 */

                /**
                 * @brief Initialize the problem.
                 *
                 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
                 * @tparam FilterKernelPolicy Kernel policy for filter operator.
                 *
                 * @param context CudaContext pointer for ModernGPU API.
                 * @param problem Pointer to Problem object.
                 * @param max_grid_size Maximum grid size for kernel calls.
                 *
                 * @return cudaError_t object Indicates the success of all CUDA calls
                 */
                template<typename AdvanceKernelPolicy, typename FilterKernelPolicy>
                cudaError_t InitBP(
                        ContextPtr *context,
                        Problem *problem,
                        int max_grid_size = 512
                ) {
                    cudaError_t retval = cudaSuccess;
                    if (retval = BaseEnactor::Init(
                            max_grid_size,
                            AdvanceKernelPolicy::CTA_OCCUPANCY,
                            FilterKernelPolicy::CTA_OCCUPANCY
                    )) {
                        return retval;
                    }

                    if (this->debug) {
                        printf("BP vertex map occupancy %d, level-grid size %d\n",
                               FilterKernelPolicy::CTA_OCCUPANCY, this->enactor_stats[0].filter_grid_size
                        );
                    }

                    this->problem = problem;
                    thread_slices = new ThreadSlice[this->num_gpus];
                    thread_Ids = new CUTThread[this->num_gpus];

                    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
                        thread_slices[gpu].thread_num = gpu;
                        thread_slices[gpu].problem = (void *) problem;
                        thread_slices[gpu].enactor = (void *) this;
                        thread_slices[gpu].context = &(context[gpu * this->num_gpus]);
                        thread_slices[gpu].status = ThreadSlice::Status::Inited;
                        thread_slices[gpu].thread_Id = cutStartThread(
                                (CUT_THREADROUTINE) &(BPThread<
                                        AdvanceKernelPolicy, FilterKernelPolicy, BPEnactor<Problem> >),
                                (void *) &(thread_slices[gpu])
                        );
                        thread_Ids[gpu] = thread_slices[gpu].thread_Id;
                    }

                    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
                        while (thread_slices[gpu].status != ThreadSlice::Status::Idle) {
                            sleep(0);
                        }
                    }
                    return retval;
                }

                cudaError_t Extract() {
                    cudaError_t retval = cudaSuccess;
                    typedef typename Problem::DataSlice DataSlice;
                    DataSlice *data_slice = NULL;
                    EnactorStats<SizeT> *enactor_stats = NULL;
                    int num_blocks = 0;
                    for (int thread_num = 1; thread_num < this->num_gpus; thread_num++) {
                        if (retval = util::SetDevice(this->gpu_idx[thread_num])) {
                            return retval;
                        }

                        data_slice = problem->data_slices[thread_num].GetPointer(util::HOST);
                        enactor_stats = &(this->enactor_stats[thread_num * this->num_gpus]);

                        num_blocks = data_slice->local_vertices.GetSize() / 512 + 1;
                        if (num_blocks > 240) {
                            num_blocks = 240;
                        }
                        Assign_Final_Value_Kernel <<< num_blocks, 512, 0, data_slice->streams[0] >>> (
                                data_slice->local_vertices.GetSize(),
                                        data_slice->local_vertices.GetPointer(util::DEVICE),
                                        data_slice->belief_curr.GetPointer(util::DEVICE),
                                        (thread_num == 0) ? (Value*) NULL
                                                          : data_slice->value__associate_out[1].GetPointer(util::DEVICE)
                        );

                        enactor_stats->iteration = 0;
                        PushNeighbor<Enactor, GraphSliceT, DataSlice, 0, 1>(
                                this,
                                thread_num,
                                0,
                                data_slice->local_vertices.GetSize(),
                                enactor_stats,
                                problem->data_slices[thread_num].GetPointer(util::HOST),
                                problem->data_slices[0].GetPointer(util::HOST),
                                problem->graph_slices[thread_num],
                                problem->graph_slices[0],
                                this->communicate_multipy
                        );
                        Set_Record(data_slice, enactor_stats->iteration, 1, 0, data_slice->streams[0]);
                        data_slice->final_event_set = true;
                    }

                    if (retval = util::SetDevice(this->gpu_idx[0])) return retval;

                    data_slice = problem->data_slices[0].GetPointer(util::HOST);
                    enactor_stats = this->enactor_stats + 0;

                    num_blocks = data_slice->local_vertices.GetSize() / 512 + 1;
                    if (num_blocks > 240) {
                        num_blocks = 240;
                    }

                    Assign_Final_Value_Kernel << < num_blocks, 512, 0, data_slice->streams[0] >> > (
                            data_slice->local_vertices.GetSize(),
                                    data_slice->local_vertices.GetPointer(util::DEVICE),
                                    data_slice->belief_curr.GetPointer(util::DEVICE),
                                    (Value*) NULL
                    );

                    for (int peer = 1; peer < this->num_gpus; peer++) {
                        if (retval = util::GRError(
                                cudaMemcpyAsync(data_slice->remote_vertices_in[peer].GetPointer(util::DEVICE)),
                                problem->data_slices[peer]->local_vertices.GetPointer(util::HOST),
                                sizeof(VertexId) * problem->data_slices[peer]->local_vertices.GetSize(),
                                cudaMemcpyHostToDevice, data_slice->streams[peer],
                                "cudaMemcpyAsync failed", __FILE__, __LINE__
                        )) {
                            return retval;
                        }
                    }

                    for (int peer = 1; peer < this->num_gpus; peer++) {
                        int peer_iteration = this->enactor_stats[peer * this->num_gpus].iteration;
                        if (retval = util::GRError(
                                cudaStreamWaitEvent(data_slice->streams[peer],
                                                    problem->data_slices[peer]->events[peer_iteration % 4][0][0], 0),
                                "cudaStreamWaitEvent failed", __FILE__, __LINE__
                        )) {
                            return retval;
                        }
                        Expand_Incoming_Final<VertexId, SizeT, Value>
                                << < 240, 512, 0, data_slice->streams[peer] >> > (
                                problem->data_slices[peer]->local_vertices.GetSize(),
                                        data_slice->remote_vertices_in[peer].GetPointer(util::DEVICE),
                                        data_slice->value__assoicate_in[peer_iteration % 2][peer].GetPointer(
                                                util::DEVICE),
                                        data_slice->belief_curr.GetPointer(util::DEVICE));
                        if (retval = util::GRError(
                                cudaEventRecord(data_slice->events[enactor_stats->iteration % 4][peer][0],
                                                data_slice->streams[peer]),
                                "cudaEventRecord failed", __FILE__, __LINE__
                        )) {
                            return retval;
                        }
                        if (retval = util::GRError(
                                cudaStreamWaitEvent(data_slice->streams[0],
                                                    data_slice->events[enactor_stats->iteration % 4][peer][0], 0),
                                "cudaStreamWaitEvent failed", __FILE__, __LINE__
                        )) {
                            return retval;
                        }
                    }

                    if (retval = data_slice->node_ids.Allocate(data_slice->nodes, util::DEVICE)) return retval;
                    if (retval = data_slice->temp_vertex.Allocate(data_slice->nodes, util::DEVICE)) return retval;
                    util::MemsetIdxKernel << < 240, 512, 0, data_slice->streams[0] >> > (
                            data_slice->node_ids.GetPointer(util::DEVICE),
                                    data_slice->nodes
                    );

                    /*

                    size_t cub_required_size = 0;
                    void* temp_storage = NULL;
                    cub::DoubleBuffer<Value> key_buffer(
                            data_slice->belief_curr.GetPointer(util::DEVICE),
                            data_slice->belief_next.GetPointer(util::DEVICE)
                    );
                    cub::DoubleBuffer<VertexId> value_buffer(
                            data_slice->node_ids.GetPionter(util::DEVICE),
                            data_slice->temp_vertex.GetPointer(util::DEVICE)
                    );

                    if (retval = util::GRError(
                            cub::DeviceRadixSort::SortPairsDescending(
                                    temp_storage,
                                    cub_required_size,
                                    key_buffer,
                                    value_buffer,
                                    data_slice->nodes
                            ),
                            "cubDeviceRadixSort failed", __FILE__, __LINE__
                    )) {
                        return retval;
                    }
                    */

                    if (retval = util::GRError(
                            cudaStreamSynchronize(data_slice->streams[0]),
                            "cudaStreamSynchronize failed", __FILE__, __LINE__
                    )) {
                        return retval;
                    }

                    return retval;
                }

                /**
                 * @brief Reset enactor
                 *
                 * @tparam AdvanceKernelPolicy
                 * @tparam FilterKernelPolicy
                 *
                 * @return cudaError_t object Indicates the success of all CUDA calls
                 */
                template<typename AdvanceKernelPolicy, typename FilterKernelPolicy>
                cudaError_t ResetBP() {
                    cudaError_t retval = cudaSuccess;
                    if (retval = BaseEnactor::Reset()) {
                        return retval;
                    }

                    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
                        thread_slices[gpu].status = ThreadSlice::Status::Wait;

                        if (retval = util::SetDevice(problem->gpu_idx[gpu])) {
                            return retval;
                        }

                        if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::TWC_FORWARD) {
                            // do nothing
                        } else {
                            bool over_sized = false;
                            if (retval = Check_Size<SizeT, SizeT>(
                                    this->size_check, "scanned edges",
                                    problem->data_slices[gpu]->local_vertices.GetSize() + 2,
                                    problem->data_slices[gpu]->scanned_edges,
                                    over_sized, -1, -1, -1, false
                            )) {
                                return retval;
                            }
                            this->frontier_attribute[gpu * this->num_gpus].queue_length =
                                    problem->data_slices[gpu]->local_vertices.GetSize();

                            retval = gunrock::oprtr::advance::ComputeOutputLength<AdvanceKernelPolicy, Problem,
                                    BPFunctor<VertexId, SizeT, Value, Problem>,
                                    gunrock::oprtr::advance::V2V>(
                                    this->frontier_attribute + gpu * this->num_gpus,
                                    problem->graph_slices[gpu]->row_offsets.GetPointer(util::DEVICE),
                                    problem->graph_slices[gpu]->column_indices.GetPointer(util::DEVICE),
                                    (SizeT *) NULL,
                                    (VertexId *) NULL,
                                    problem->data_slices[gpu]->local_vertices.GetPointer(util::DEVICE),
                                    problem->data_slices[gpu]->scanned_edges[0].GetPointer(util::DEVICE),
                                    problem->data_slices[gpu]->nodes,
                                    problem->data_slices[gpu]->edges,
                                    thread_slices[gpu].context[0][0],
                                    problem->data_slices[gpu]->streams[0],
                                    false,
                                    false,
                                    false
                            );

                            if (retval = this->frontier_attribute[gpu * this->num_gpus].output_length.Move(util::DEVICE,
                                                                                                           util::HOST,
                                                                                                           1, 0,
                                                                                                           problem->data_slices[gpu]->streams[0])) {
                                return retval;
                            }

                            if (retval = util::GRError(cudaStreamSynchronize(problem->data_slices[gpu]->streams[0]),
                                                       "cudaStreamSynchronize failed", __FILE__, __LINE__
                            )) {
                                return retval;
                            }
                        }
                    }
                    return retval;
                }

                template<typename AdvancedKernelPolicy, typename FilterKernelPolicy>
                cudaError_t EnactBP() {
                    cudaError_t retval = cudaSuccess;
                    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
                        thread_slices[gpu].status = ThreadSlice::Status::Running;
                    }
                    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
                        while (thread_slices[gpu].status != ThreadSlice::Status::Idle) {
                            sleep(0);
                        }
                    }

                    for (int gpu = 0; gpu < this->num_gpus * this->num_gpus; gpu++) {
                        if (this->enactor_stats[gpu].retval != cudaSuccess) {
                            retval = this->enactor_stats[gpu].retval;
                            return retval;
                        }
                    }

                    if (this->debug) {
                        printf("\nGPU BP Done.\n");
                    }

                    return retval;
                }

                typedef gunrock::oprtr::filter::KernelPolicy<
                        Problem, // Problem data type
                        300, // CUDA ARCH,
                        // INSTRUMENT, // INSTRUMENT
                        0, // SATURATION QUIT
                        true, // DEQUEUE_PROBLEM_SIZE,
                        sizeof(VertexId) == 4 ? 8 : 4, // MIN_CTA_OCCUPANCY
                        8, // LOG_THREADS
                        1, // LOG_LOAD_VEC_SIZE
                        0, // LOG LOADS_PER_TILE
                        5, // LOG RAKING THREADS
                        5, // END BITMASK CULL
                        8, // LOG SCHEDULE GRANULARITY
                        gunrock::oprtr::filter::BY_PASS>
                        FilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                        Problem, // Problem data type
                        300, // CUDA ARCH
                        1, // MIN CTA OCCUPANCY
                        10, // LOG THREADS
                        8, // LOG BLOCKS
                        32 * 128, // LIGHT EDGE THRESHOLD (used for partitioned advance mode)
                        1, // LOG LOAD VEC SIZE
                        0, // LOG LOADS PER TILE
                        5, // LOG RAKING THREADS
                        32, // WARP GATHER THRESHOLD
                        128 * 4, // CTA GATHER THRESHOLD
                        7, //LOG SCHEDULE GRANULARITY
                        gunrock::oprtr::advance::LB>
                        LB_AdvanceKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                        Problem, // Problem data type
                        300, // CUDA ARCH
                        1, // MIN CTA OCCUPANCY
                        10, // LOG THREADS
                        8, // LOG BLOCKS
                        32 * 128, // LIGHT_EDGE_THRESHOLD (used for partitioned advanced mode)
                        1, // LOG LOAD VEC SIZE
                        0, // LOG LOADS PER TILE
                        5, // LOG RAKING THREADS
                        32, // WARP GATHER THRESHOLD
                        128 * 4, // CTA GATHER THRESHOLD
                        7, // LOG SCHEDULE GRANULARITY
                        gunrock::oprtr::advance::LB_LIGHT>
                        LB_LIGHT_AdvanceKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                        Problem, // Problem data type
                        300, // CUDA ARCH
                        1, // MIN CTA OCCUPANCY
                        7, // LOG THREADS
                        8, // LOG BLOCKS
                        32 * 128, // LIGHT EDGE THRESHOLD (used for partitioned advance mode)
                        1, // LOG LOAD VEC SIZE
                        1, // LOG LOADS PER TILE
                        5, // LOG RAKING THREADS
                        32, // WARP GATHER THRESHOLD
                        128 * 4, // CTA GATHER THRESHOLD
                        7, // CTA GATHER THRESHOLD
                        gunrock::oprtr::advance::TWC_FORWARD>
                        TWC_AdvanceKernelPolicy;

                template<typename Dummy, typename gunrock::oprtr::advance::MODE A_MODE>
                struct MODE_SWITCH {
                };

                template<typename Dummy>
                struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB> {
                    static cudaError_t Enact(Enactor &enactor) {
                        return enactor.EnactBP<LB_AdvanceKernelPolicy, FilterKernelPolicy>();
                    }

                    static cudaError_t
                    Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 512) {
                        return enactor.InitBP<LB_AdvanceKernelPolicy, FilterKernelPolicy>(context, problem,
                                                                                          max_grid_size);
                    }

                    static cudaError_t Reset(Enactor &enactor) {
                        return enactor.ResetBP<LB_AdvanceKernelPolicy, FilterKernelPolicy>();
                    }
                };

                template<typename Dummy>
                struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB_LIGHT> {
                    static cudaError_t Enact(Enactor &enactor) {
                        return enactor.EnactBP<LB_LIGHT_AdvanceKernelPolicy, FilterKernelPolicy>();
                    }

                    static cudaError_t
                    Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 512) {
                        return enactor.InitBP<LB_LIGHT_AdvanceKernelPolicy, FilterKernelPolicy>
                                (context, problem, max_grid_size);
                    }

                    static cudaError_t Reset(Enactor &enactor) {
                        return enactor.ResetBP<LB_LIGHT_AdvanceKernelPolicy, FilterKernelPolicy>();
                    }
                };

                template<typename Dummy>
                struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::TWC_FORWARD> {
                    static cudaError_t Enact(Enactor &enactor) {
                        return enactor.EnactBP<TWC_AdvanceKernelPolicy, FilterKernelPolicy>();
                    }

                    static cudaError_t
                    Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 512) {
                        return enactor.InitBP<TWC_AdvanceKernelPolicy, FilterKernelPolicy>
                                (context, problem, max_grid_size);
                    }

                    static cudaError_t Reset(Enactor &enactor) {
                        return enactor.ResetBP<TWC_AdvanceKernelPolicy, FilterKernelPolicy>();
                    }
                };


                /**
                 * @brief BP Enact kernel entry
                 *
                 * @param[in] traversal_mode Load-balanced or Dynamic cooperative
                 *
                 * @return cudaError_t object Indicates the success of all CUDA calls
                 */
                cudaError_t Enact(
                        std::string traversal_mode = "LB"
                ) {
                    if (this->min_sm_version >= 300) {
                        if (traversal_mode == "LB") {
                            return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB>::Reset(*this);
                        } else if (traversal_mode == "LB_LIGHT") {
                            return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT>::Reset(*this);
                        } else if (traversal_mode == "TWC") {
                            return MODE_SWITCH<SizeT, gunrock::oprtr::advance::TWC_FORWARD>::Reset(*this);
                        }
                    }
                    printf("Not yet tuned for this architecture\n");
                    return cudaErrorInvalidDeviceFunction;
                }

                cudaError_t Reset(std::string traversal_mode = "LB") {
                    if (this->min_sm_version >= 300) {
                        if (traversal_mode == "LB") {
                            return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB>::Reset(*this);
                        } else if (traversal_mode == "LB_LIGHT") {
                            return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT>::Reset(*this);
                        } else if (traversal_mode == "TWC") {
                            return MODE_SWITCH<SizeT, gunrock::oprtr::advance::TWC_FORWARD>::Reset(*this);
                        }
                    }
                        printf("Not yet tuned for this architecture\n");
                        return cudaErrorInvalidDeviceFunction;
                }

                /**
                 * @brief BP Enact kernel entry
                 *
                 * @param[in] context CudaContext pointer for ModernGPU API
                 * @param[in] problem Pointer to Problem object
                 * @param[in] traversal_mode Load-balanced or Dynamic coordination
                 * @param[in] max_grid_size Maximum grid size for kernel calls
                 *
                 * @return cudaError_t object Indicates the success of all CUDA calls
                 */
                cudaError_t Init(
                        ContextPtr *context,
                        Problem *problem,
                        std::string traversal_mode = "LB",
                        int max_grid_size = 512
                ) {
                    if (this->min_sm_version >= 300) {
                        if (traversal_mode == "LB") {
                            return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB>::Init(*this, context, problem,
                                                                                         max_grid_size);
                        } else if (traversal_mode == "LB_LIGHT") {
                            return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT>::Init(*this, context, problem,
                                                                                               max_grid_size);
                        } else if (traversal_mode == "TWC") {
                            return MODE_SWITCH<SizeT, gunrock::oprtr::advance::TWC_FORWARD>::Init(*this, context,
                                                                                                  problem,
                                                                                                  max_grid_size);
                        }
                    }

                    printf("Not yet tuned for this architecture\n");
                    return cudaErrorInvalidDeviceFunction;
                }
            };
        }
    }
}