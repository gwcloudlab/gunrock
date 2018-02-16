/**
 * @file test_bp.cu
 *
 * Test program for belief propagation
*/

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>

#include <time.h>

#include <gunrock/gunrock.h>
#include <gunrock/csr.cuh>
#include <gunrock/graphio/market.cuh>

#include <moderngpu.cuh>


int main(int argc, char** argv)
{
    clock_t start, end;
    double time_elapsed;

    typedef int VertexId;
    typedef float Value;
    typedef int SizeT;

    gunrock::Csr<VertexId, SizeT , Value> csr(false);
    if(gunrock::graphio::BuildMarketGraph_BP<true, VertexId, SizeT, Value>("/home/mjt5v/Source_Code/gunrock/tests/bp/test.bif.edges.mtx", "/home/mjt5v/Source_Code/gunrock/tests/bp/test.bif.nodes.mtx", csr, false, false, false) != 0) {
        perror("Unable to build csr...exiting\n");
        return 1;
    }

    GRTypes data_t;
    data_t.VTXID_TYPE = VTXID_INT;
    data_t.SIZET_TYPE = SIZET_INT;
    data_t.VALUE_TYPE = VALUE_FLOAT;

    GRGraph *graphi = new GRGraph();
    GRGraph *grapho = new GRGraph();

    graphi->num_nodes = csr.nodes;
    graphi->num_edges = csr.edges;
    graphi->row_offsets = csr.row_offsets;
    graphi->col_indices = csr.column_indices;
    graphi->edge_values = csr.edge_values;
    graphi->node_value1 = csr.node_values;

    GRSetup *config = InitSetup(1, {0});
    config->quiet = false;

    start = clock();

    gunrock_bp(grapho, graphi, config, data_t);

    end = clock();
    time_elapsed = (double)(end - start)/(CLOCKS_PER_SEC);

    printf("Nodes\tEdges\tTime(s)\n");
    printf("%d\t%d\t%.6f\n", csr.nodes, csr.edges, time_elapsed);

    float *beliefs = new float[graphi->num_nodes];
    memcpy(beliefs, grapho->node_value1, graphi->num_nodes);

    for(int node = 0; node < graphi->num_nodes; ++node) {
        printf("Belief [%.10e]\n", beliefs[node]);
    }

    if(graphi) {
        delete graphi;
    }
    if(grapho) {
        delete grapho;
    }
    if(beliefs) {
        delete beliefs;
    }

    return 0;
}