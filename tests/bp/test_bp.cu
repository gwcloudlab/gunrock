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
#include <fstream>

#include <time.h>

#include <gunrock/gunrock.h>
#include <gunrock/csr.cuh>
#include <gunrock/graphio/market.cuh>

#include <moderngpu.cuh>


int RunBP(std::string in_edges_filename, std::string in_nodes_filename, std::string out_filename, std::ofstream & out) {
    clock_t start, end;
    double time_elapsed;

    typedef int VertexId;
    typedef float Value;
    typedef int SizeT;

    gunrock::Csr<VertexId, SizeT , Value> csr(false);
    if(gunrock::graphio::BuildMarketGraph_BP<true, VertexId, SizeT, Value>((char *)in_edges_filename.c_str(), (char *)in_nodes_filename.c_str(), csr, false, false, false) != 0) {
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
    out << "Nodes,Edges,Time(s)" << std::endl;
    printf("%d\t%d\t%.6f\n", csr.nodes, csr.edges, time_elapsed);
    out << out_filename << "," << csr.nodes << "," << csr.edges << "," << -1 << "," << config->max_iters << "," << time_elapsed << std::endl;
    out.flush();

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
        delete[] beliefs;
    }

    return 0;
}

int main(int argc, char** argv)
{
    const std::string out_filename = "gunrock_bp.csv";

    std::ofstream out;
    out.open(out_filename.c_str());

    out << "File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)" << std::endl;

    //RunBP("/home/mjt5v/Source_Code/gunrock/tests/bp/test.bif.edges.mtx", "/home/mjt5v/Source_Code/gunrock/tests/bp/test.bif.nodes.mtx", out_filename, out);
    RunBP("/home/trotsky/gunrock_benchmark_files/10_20.bif.edges.mtx", "/home/trotsky/gunrock_benchmark_files/10_20.bif.nodes.mtx", out_filename, out);
    RunBP("/home/trotsky/gunrock_benchmark_files/100_200.bif.edges.mtx", "/home/trotsky/gunrock_benchmark_files/100_200.bif.nodes.mtx", out_filename, out);
    RunBP("/home/trotsky/gunrock_benchmark_files/1000_2000.bif.edges.mtx", "/home/trotsky/gunrock_benchmark_files/1000_2000.bif.nodes.mtx", out_filename, out);
    RunBP("/home/trotsky/gunrock_benchmark_files/10000_20000.bif.edges.mtx", "/home/trotsky/gunrock_benchmark_files/10000_20000.bif.nodes.mtx", out_filename, out);
    RunBP("/home/trotsky/gunrock_benchmark_files/100000_200000.bif.edges.mtx", "/home/trotsky/gunrock_benchmark_files/100000_200000.bif.nodes.mtx", out_filename, out);
    RunBP("/home/trotsky/gunrock_benchmark_files/200000_400000.bif.edges.mtx", "/home/trotsky/gunrock_benchmark_files/200000_400000.bif.nodes.mtx", out_filename, out);
    RunBP("/home/trotsky/gunrock_benchmark_files/400000_800000.bif.edges.mtx", "/home/trotsky/gunrock_benchmark_files/400000_800000.bif.nodes.mtx", out_filename, out);
    RunBP("/home/trotsky/gunrock_benchmark_files/800000_1600000.bif.edges.mtx", "/home/trotsky/gunrock_benchmark_files/800000_1600000.bif.nodes.mtx", out_filename, out);

    out.close();

    return 0;
}