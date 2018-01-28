
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
                    const util::Array1D<SizeT, Value>* const belief_next,
                    util::Array1D<SizeT, Value>* belief_curr
            ) {
                const SizeT STRIDE = (SizeT)(gridDim.x * blockDim.x);
                SizeT x = (SizeT)(blockIdx.x * blockDim.x + threadIdx.x);
                while (x < num_elements) {
                    VertexId key = keys[x];
                    if (markers[key] == 1) {
                        Value *curr_ptr, *next_ptr;
                        for (int y = 0; y < belief_next->GetSize(); y++) {
                            curr_ptr = belief_curr[key].GetPointer(util::DEVICE) + y;
                            next_ptr = belief_next[key].GetPointer(util::DEVICE) + y;
                            *curr_ptr = *next_ptr;
                        }
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
                util::Array1D<SizeT, Value>** s_value__associate_in = (util::Array1D<SizeT, Value>**)&(s_array[offset]);
                offset += sizeof(Value*) * NUM_VALUE__ASSOCIATES;
                offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
                util::Array1D<SizeT, Value>** s_value__associate_org = (util::Array1D<SizeT, Value>**)&(s_array[offset]);
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
                    util::Array1D<SizeT, Value> mul_values = s_value__associate_in[0][x];
                    for (int y = 0; y < mul_values.GetSize(); y++) {
                        Value* mul_value_ptr = mul_values.GetPointer(util::DEVICE);
                        Value* mul_addr = s_value__associate_org[0] + key + y;
                        Value old_value = atomicMul(mul_addr, *mul_value_ptr);
                    }
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
                    util::Array1D<typename KernelPolicy::SizeT, typename KernelPolicy::Value> *d_belief_in,
                    util::Array1D<typename KernelPolicy::SizeT, typename KernelPolicy::Value> *d_belief_out
            ){
                typedef typename KernelPolicy::VertexId VertexId;
                typedef typename KernelPolicy::SizeT SizeT;
                typedef typename KernelPolicy::Value Value;

                SizeT x = (SizeT)(blockIdx.x * blockDim.x + threadIdx.x);
                const SizeT STRIDE = (SizeT)(gridDim.x * blockDim.x);

                while (x < num_elements)
                {
                    VertexId key = d_keys[x];
                    util::Array1D<SizeT, Value> in_values = d_belief_in[x];
                    util::Array1D<SizeT, Value> out_values = d_belief_out[x];
                    for (int y = 0; y < in_values.GetSize(); y++)
                    {
                        Value *mul_value_ptr = in_values.GetPointer(util::DEVICE) + y;
                        Value *out_value_ptr = out_values.GetPointer(util::DEVICE) + key + y;
                        Value old_value = atomicMul(out_value_ptr, *mul_value_ptr);
                    }
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
                    const util::Array1D<SizeT, Value>* const belief_next,
                    const util::Array1D<SizeT, Value>* belief_out
            ){
                const SizeT STRIDE = (SizeT)(gridDim.x * blockDim.x);
                SizeT x = (SizeT)(blockIdx.x * blockDim.x + threadIdx.x);

                while (x < num_elements)
                {
                    VertexId key = keys_out[x];
                    for (int y = 0; y < belief_out[x].GetSize() && y < belief_next[key].GetSize(); y++)
                    {
                        Value* out_value = belief_out[x].GetPointer(util::DEVICE) + y;
                        Value* next_value = belief_next[key].GetPointer(util::DEVICE) + y;
                        *out_value = *next_value;
                    }
                    x += STRIDE;
                }
            };

            template<typename VertexId, typename SizeT, typename Value>
            __global__ void Expand_Incoming_Final (
                    const SizeT num_elements,
                    const VertexId* const keys_in,
                    const util::Array1D<SizeT, Value>* const beliefs_in,
                    util::Array1D<SizeT, Value>* beliefs_out
            )
            {
                const SizeT STRIDE = (SizeT)(gridDim.x * blockDim.x);
                SizeT x = (SizeT)(blockIdx.x * blockDim.x + threadIdx.x);
                while (x < num_elements) {
                    VertexId key = keys_in[x];
                    for (int y = 0; y < beliefs_in[x].GetSize() && y < beliefs_out[x].GetSize(); y++)
                    {
                        Value *in_value = beliefs_in[x].GetPointer(util::DEVICE) + y;
                        Value *out_value = beliefs_out[x].GetPointer(util::DEVICE) + y;
                        *out_value = *in_value;
                    }
                    x += STRIDE;
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
                typedef BPFunctor<VertexId, Size, Value, Problem> BpFunctor;
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

                            gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem, BpFunctor>(
                                    enactor_stats[0],
                                    frontier_attribute[0],
                                    typename BPFunctor::LabelT(),
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

                            util::MemsetKernel<<<240, 512, 0, stream>>>(
                                    data_slice->belief_next.GetPointer(util::DEVICE),
                                    NULL, graph_slice->nodes
                            );

                            if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream),
                            "cudaStreamSynchronize failed", __FILE__, __LINE__)) {
                                return;
                            }
                            data_slice->num_update_vertices = frontier_attribute->queue_length;
                        }

                        frontier_attribute->queue_length = data_slice->local_vertices.GetSize();

                        if (enactor->debug)
                        {
                            util::cpu_mt::PrintMessage("Advance Bpfunctor start.",
                            thread_num, enactor_stats->iteration, peer_);
                        }

                        // Edge Map
                        frontier_attribute->queue_reset = false;
                        gunrock::oprtr::LaunchKernel<AdvanceKernelPolicy, Problem, BpFunctor,
                                gunrock::oprtr::advance::V2V>(
                            enactor_stats[0],
                            frontier_attribute[0],
                            typename BPFunctor::LabelT(),
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
                };

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
                        util::Array1D<SizeT, util::Array1D<SizeT, Value>> &value__associate_in,
                        util::Array1D<SizeT, VertexId> &keys_out,
                        util::Array1D<SizeT, VertexId*> &vertex_associate_orgs,
                        util::Array1D<SizeT, util::Array1D<SizeT, Value>*> &value__associate_orgs,
                        DataSlice  *h_data_slice,
                        EnactorStats<SizeT> *enactor_stats
                ){
                    int num_blocks = num_elements / AdvanceKernelPolicy::THREADS / 2 + 1;
                    if (num_blocks > 240) {
                        num_blocks = 240;
                    }
                    Expand_Incoming_BP_Kernel<AdvanceKernelPolicy, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
                    <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>
                    (h_data_slice->gpu_dx,
                    h_data_slice->in_counters[peer_],
                    h_data_slice->remote_vertices_in[peer_].GetPointer(util::DEVICE),
                    value__associate_in.GetPointer(util::DEVICE),
                    h_data_slice->belief_next.GetPointer(util::DEVICE)
                    );
                };

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
                };

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
                            data_slice[0]->value__associate_outs[peer][0] = data_slice[0]->value__associate_out[peer_][0].GetPointer(util::DEVICE);
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
                };
            };


        }
    }
}