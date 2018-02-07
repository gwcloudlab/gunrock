/**
 * @file bp_app.cu
 *
 * @brief Gunrock Belief propagation application
*/
#include <gunrock/gunrock.h>

// graph construction utilities
#include <gunrock/graphio/market.cuh>

// bp includes
#include <gunrock/app/bp/bp_enactor.cuh>
#include <gunrock/app/bp/bp_problem.cuh>
#include <gunrock/app/bp/bp_functor.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bp;

/**
 * @brief BP_Parameter structure
*/
struct BP_Parameter : gunrock::app::TestParameter_Base {
public:
    float delta; // delta value for BP
    float error; // error threshold for BP
    int max_iter; // maximum number of iterations for BP
    bool normalized;

    BP_Parameter()
    {
        delta = 0.85f;
        error = 0.01f;
        max_iter = 50;
        normalized = false;
    }

    ~BP_Parameter() {}
};


template <typename VertexId, typename SizeT, typename Value, bool NORMALIZED>
void runBP(GRGraph *output, BP_Parameter *parameter);

/**
 * @brief Run test
 *
 * @tparam VertexId Vertex identifier type
 * @tparam SizeT Graph size type
 * @tparam Value Attribute type
 *
 * @param output Pointer to output graph structure of the problem
 * @param parameter primitive-specific test parameters
 */
template <typename VertexId, typename SizeT, typename Value>
void normalizedBP(GRGraph *output, BP_Parameter *parameter)
{
    if (parameter->normalized)
    {
        runBP<VertexId, SizeT, Value, true> (output, parameter);
    }
    else
    {
        runBP<VertexId, SizeT, Value, false> (output, parameter);
    }
};

/**
 * @brief Run test
 *
 * @tparam VertexId Vertex Identifier type
 * @tparam SizeT Graph size type
 * @tparam Value Attribute type
 * @tparam NORMALIZED
 * @param output
 * @param parameter
 */
template <typename VertexId, typename SizeT, typename Value, bool NORMALIZED>
void runBP(GRGraph *output, BP_Parameter *parameter)
{
    typedef BPProblem <VertexId, SizeT, Value, NORMALIZED> Problem;
    typedef BPEnactor < Problem> Enactor;

    Csr<VertexId, SizeT, Value> *graph = (Csr<VertexId, SizeT, Value>*)parameter->graph;
    bool quiet = parameter->g_quiet;
    int max_grid_size = parameter->max_grid_size;
    int num_gpus = parameter->num_gpus;
    double max_queue_sizing = parameter->max_queue_sizing;
    double max_in_sizing = parameter->max_in_sizing;
    ContextPtr *context = (ContextPtr *)parameter->context;
    std::string partition_method = parameter->partition_method;
    int *gpu_idx = parameter->gpu_idx;
    cudaStream_t *streams = parameter->streams;
    float partition_factor = parameter->partition_factor;
    int partition_seed = parameter->partition_seed;
    bool g_stream_from_host = parameter->g_stream_from_host;
    VertexId src = parameter->src[0];
    Value delta = parameter->delta;
    Value error = parameter->error;
    SizeT max_iter = parameter->max_iter;
    std::string traversal_mode = parameter->traversal_mode;
    bool instrument = parameter->instrumented;
    bool size_check = parameter->size_check;
    bool debug = parameter->debug;
    size_t *org_size = new size_t[num_gpus];


    Value *h_beliefs = new Value[graph->nodes];
    VertexId *h_node_id = new VertexId[graph->nodes];

    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }

    Problem *problem = new Problem(false); // Allocate problem on GPU
    util::GRError(
            problem->Init(
                    g_stream_from_host,
                    graph,
                    NULL,
                    num_gpus,
                    gpu_idx,
                    partition_method,
                    streams,
                    context,
                    max_queue_sizing,
                    max_in_sizing,
                    partition_factor,
                    partition_seed
            ),
            "BP Intialization Failed", __FILE__, __LINE__
    );

    Enactor *enactor = new Enactor(
            num_gpus, gpu_idx, instrument, debug, size_check
    );
    util::GRError(
            enactor->Init(context, problem, traversal_mode, max_grid_size),
            "BP Enactor Init failed", __FILE__, __LINE__
    );

    CpuTimer cpu_timer;

    util::GRError(
            problem->Reset(src, delta, error, max_iter,
                enactor->GetFrontierType(), max_queue_sizing
            ),
            "BP Problem Data Reset Failed", __FILE__, __LINE__
    );
    util::GRError(
            enactor->Reset(), "BP Enactor Reset Rest failed", __FILE__, __LINE__
    );

    cpu_timer.Start();
    util::GRError(
            enactor->Enact(traversal_mode), "BP Problem Enact Failed", __FILE__, __LINE__
    );
    cpu_timer.Stop();

    float elapsed = cpu_timer.ElapsedMillis();

    // Copy out results
    util::GRError(
            problem->Extract(h_beliefs, h_node_id),
            "BP Problem Data Extraction Failed", __FILE__, __LINE__
    );

    output->node_value1 = (Value*)&h_beliefs[0];
    output->node_value2 = (VertexId *)&h_node_id[0];

    if (!quiet){
        printf("  GPU BP finished in %lf msec.\n", elapsed);
    }

    // Clean up
    if (org_size) { delete[] org_size; org_size = NULL; }
    if (problem) { delete problem; problem = NULL; }
    if (enactor) { delete enactor; enactor = NULL; }
};

void dispatchBP(
        GRGraph *grapho,
        const GRGraph *graphi,
        const GRSetup *config,
        const GRTypes data_t,
        ContextPtr *context,
        cudaStream_t *streams
)
{
    BP_Parameter *parameter = new BP_Parameter;
    parameter->src = (long long *)malloc(sizeof(long long));
    parameter->src[0] = -1;
    parameter->context = context;
    parameter->streams = streams;
    parameter->g_quiet = config->quiet;
    parameter->num_gpus = config->num_devices;
    parameter->gpu_idx = config->device_list;
    parameter->delta = config->bp_delta;
    parameter->error = config->bp_error;
    parameter->max_iter = config->max_iters;
    parameter->normalized = config->bp_normalized;
    parameter->g_undirected = false;

    switch (data_t.VTXID_TYPE)
    {
        case VTXID_INT:
        {
            switch (data_t.SIZET_TYPE)
            {
                case SIZET_INT:
                {
                    switch (data_t.VALUE_TYPE) {
                        case VALUE_INT:
                        {
                            printf("Not Yet Support For This DataType Combination.\n");
                            break;
                        }
                        case VALUE_UINT:
                        {
                            printf("Not Yet Support For This DataType Combination.\n");
                            break;
                        }
                        case VALUE_FLOAT:
                        {
                            Csr<int, int, float> csr(false);
                            csr.nodes = graphi->num_nodes;
                            csr.edges = graphi->num_edges;
                            csr.row_offsets = (int*)graphi->row_offsets;
                            csr.column_indices = (int*)graphi->col_indices;
                            parameter->graph = &csr;

                            normalizedBP<int, int, float>(grapho, parameter);

                            // rest for free memory
                            csr.row_offsets = NULL;
                            csr.column_indices = NULL;
                            break;
                        }
                    }
                    break;
                }
            }
            break;
        }
    }
    free(parameter->src);
}

void gunrock_bp(
        GRGraph *grapho,
        const GRGraph *graphi,
        const GRSetup *config,
        const GRTypes data_t
)
{
    // GPU-related configurations
    int num_gpus = 0;
    int *gpu_idx = NULL;
    ContextPtr *context = NULL;
    cudaStream_t  *streams = NULL;

    num_gpus = config->num_devices;
    gpu_idx = new int [num_gpus];
    for (int i = 0; i < num_gpus; ++i)
    {
        gpu_idx[i] = config->device_list[i];
    }

    // create streams and modernGPU context for each GPU
    streams = new cudaStream_t[num_gpus * num_gpus * 2];
    context = new ContextPtr[num_gpus * num_gpus];
    if (!config->quiet) {
        printf(" using %d GPUS:", num_gpus);
    }
    for (int gpu = 0; gpu < num_gpus; ++gpu)
    {
        if (!config->quiet) {
            printf (" %d ", gpu_idx[gpu]);
        }
        util::SetDevice(gpu_idx[gpu]);
        for (int i = 0; i < num_gpus * 2; ++i)
        {
            int _i = gpu *num_gpus * 2 + i;
            util::GRError(cudaStreamCreate(&streams[_i]),
            "cudaStreamCreate failed", __FILE__, __LINE__);
            if (i < num_gpus)
            {
                context[gpu * num_gpus + i] =
                    mgpu::CreateCudaDeviceAttachStream(gpu_idx[gpu], streams[_i]);
            }
        }
    }
    if (!config->quiet) {
        printf("\n");
    }

    dispatchBP(grapho, graphi, config, data_t, context, streams);
}

void bp(
        float *final_beliefs,
        const int num_nodes,
        const int num_edges,
        const int *row_offsets,
        const int *col_indices,
        const float *original_beliefs,
        const float *joint_probabilities,
        bool normalized
)
{
    struct GRTypes data_t;
    data_t.VTXID_TYPE = VTXID_INT;
    data_t.SIZET_TYPE = SIZET_INT;
    data_t.VALUE_TYPE = VALUE_FLOAT;

    struct GRSetup *config = InitSetup(1, NULL); // primitive-specific configures
    config->bp_normalized = normalized;

    struct GRGraph *grapho = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graphi = (struct GRGraph*)malloc(sizeof(struct GRGraph));

    graphi->num_nodes = num_nodes;
    graphi->num_edges = num_edges;
    graphi->row_offsets = (void *)&row_offsets[0];
    graphi->col_indices = (void *)&col_indices[0];
    graphi->node_value1 = (void *)&original_beliefs[0];
    graphi->edge_values = (void *)&joint_probabilities[0];

    gunrock_bp(grapho, graphi, config, data_t);
    memcpy(final_beliefs, (float *)grapho->node_value1, num_nodes * sizeof(float));

    if (graphi) {
        free(graphi);
    }
    if (grapho) {
        free(grapho);
    }

}