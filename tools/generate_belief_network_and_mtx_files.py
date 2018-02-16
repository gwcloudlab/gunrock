#!/usr/bin/python

import argparse
import random
import os
import logging
from string import Template

log = logging.getLogger('generate_belief_network')
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

VARIABLE_TEMPLATE = Template("""<VARIABLE TYPE="nature">
<NAME>Node$num</NAME>
<OUTCOME>Value1</OUTCOME>
<PROPERTY>position = (0,0)</PROPERTY>
</VARIABLE>""")

OBSERVED_NODE_TEMPLATE = Template("""<DEFINITION>
<FOR>Node$num</FOR>
<TABLE>
$prob_1
</TABLE>
</DEFINITION>""")

EDGE_TEMPLATE = Template("""<DEFINITION>
<FOR>Node$for_node</FOR>
<GIVEN>Node$given_node</GIVEN>
<TABLE>
$prob_1
</TABLE>
</DEFINITION>""")

HEADER_TEMPLATE = Template("""<?xml version="1.0"?>
<!-- DTD for the XMLBIF 0.3 format -->
<!DOCTYPE BIF [
    <!ELEMENT BIF ( NETWORK )*>
          <!ATTLIST BIF VERSION CDATA #REQUIRED>
    <!ELEMENT NETWORK ( NAME, ( PROPERTY | VARIABLE | DEFINITION )* )>
    <!ELEMENT NAME (#PCDATA)>
    <!ELEMENT VARIABLE ( NAME, ( OUTCOME |  PROPERTY )* ) >
          <!ATTLIST VARIABLE TYPE (nature|decision|utility) "nature">
    <!ELEMENT OUTCOME (#PCDATA)>
    <!ELEMENT DEFINITION ( FOR | GIVEN | TABLE | PROPERTY )* >
    <!ELEMENT FOR (#PCDATA)>
    <!ELEMENT GIVEN (#PCDATA)>
    <!ELEMENT TABLE (#PCDATA)>
    <!ELEMENT PROPERTY (#PCDATA)>
]>


<BIF VERSION="0.3">
<NETWORK>
<NAME>RandomNet</NAME>""")


FOOTER_TEMPLATE = Template("""</NETWORK>
</BIF>""")

def parse_arguments():
    """
    Parses the command line arguments
    :return: An args object holding this information
    """
    parser = argparse.ArgumentParser(description="Generate a random bayesian network")
    parser.add_argument("--nodes", type=int, required=True, help="The number of nodes to generate")
    parser.add_argument("--arcs", type=int, required=True, help="The number of arcs to generate")
    parser.add_argument("--seed", type=int, default=1, help="The seed to use for generation")
    parser.add_argument("--file", type=str, required=True, help="The file to write to")
    parser.add_argument("--observed-probability", type=float, default=0.3, help="The probability that a given node is observed")
    args = parser.parse_args()
    assert args.arcs >= args.nodes - 1, "Arcs (%d) must be >= Nodes - 1 (%d)" % (args.arcs, args.nodes - 1)
    assert args.arcs <= args.nodes * (args.nodes - 1) / 2, "Arcs (%d) must be <= Nodes * (Nodes - 1) / 2 (%d)" % (args.arcs, args.nodes * (args.nodes - 1) / 2)
    assert args.nodes > 0, "The number of nodes (%d) must be greater than 0." % args.nodes
    return args


def write_header(out):
    out.write(HEADER_TEMPLATE.substitute())


def write_footer(out):
    out.write(FOOTER_TEMPLATE.substitute())


def write_variables(out, args):
    for i in range(0, args.nodes):
        out.write(VARIABLE_TEMPLATE.substitute(num=i))
        log.info("Num variables written: %d/%d" % (i + 1, args.nodes))


def write_observed_nodes(out, mtx, args):
    k = int(args.observed_probability * args.nodes)
    population = [i for i in range(0, args.nodes)]
    samples = set(random.sample(population, k))
    for i in population:
        mtx_id = i + 1
        if i in samples:
            prob = random.random()
            out.write(OBSERVED_NODE_TEMPLATE.substitute(num=i, prob_1=prob))
            mtx.write("{:d}\t{:d}\t{:f}\n".format(mtx_id, mtx_id, prob))
        else:
            mtx.write("{:d}\t{:d}\t1.0\n".format(mtx_id, mtx_id))


def write_edge(out, mtx, src, dest):
    prob_1 = random.random()

    out.write(EDGE_TEMPLATE.substitute(given_node=src, for_node=dest, prob_1=prob_1))
    mtx.write("{:d}\t{:d}\t{:f}\n".format(src + 1, dest + 1, prob_1))


def write_and_build_tree(out, mtx, args):
    parents = {}
    unconnected_nodes = [i for i in range(0, args.nodes)]

    # based off of bayes net generator from weka: http://grepcode.com/file/repository.pentaho.org/artifactory/pentaho/pentaho.weka/pdm-3.7-ce/3.7.7.2/weka/classifiers/bayes/net/BayesNetGenerator.java#BayesNetGenerator.generateRandomNetworkStructure%28int%2Cint%29
    node_1 = random.choice(unconnected_nodes)
    unconnected_nodes.remove(node_1)
    node_2 = random.choice(unconnected_nodes)
    unconnected_nodes.remove(node_2)

    parents[node_2] = [node_1]
    parents[node_1] = []

    if node_1 == node_2:
        node_2 = (node_1 + 1) % args.nodes
    if node_2 > node_2:
        temp = node_2
        node_1 = node_2
        node_2 = temp
    write_edge(out, mtx, node_1, node_2)

    for i in range(2, args.nodes):
        connected_node = random.choice(parents.keys())
        unconnected_node = random.choice(unconnected_nodes)
        unconnected_nodes.remove(unconnected_node)
        src_node = None
        dest_node = None
        if unconnected_node < connected_node:
            src_node = unconnected_node
            dest_node = connected_node
        else:
            src_node = connected_node
            dest_node = unconnected_node
        # assert src_node != dest_node
        if dest_node not in parents:
            parents[src_node] = [dest_node]
            parents[dest_node] = []
        else:
            parents[src_node] = [dest_node] + parents[dest_node]
        log.info("Num edges written: (%d/%d)" % (i, args.arcs))
    return parents


def write_edges(out, mtx, args):
    num_arcs_written = 0
    parents = write_and_build_tree(out, mtx, args)
    for i in range(args.nodes - 1, args.arcs):
        new_edge = False
        while not new_edge:
            node_1 = random.choice(parents.keys())
            node_2 = random.choice(parents.keys())
            if node_1 == node_2:
                node_2 = (node_1 + 1) % args.nodes
            src = None
            dest = None
            if node_2 < node_1:
                src = node_2
                dest = node_1
            else:
                src = node_1
                dest = node_2
            # assert src != dest
            if src not in parents[dest]:
                write_edge(out, mtx, src, dest)
                num_arcs_written += 1
                parents[dest].append(src)
                new_edge = True
                log.info("Num edges written: (%d/%d)" % (i, args.arcs))
    return num_arcs_written


def write_mtx_edges_header(mtx, args, num_arcs_written):
    mtx.write("% Belief network generated using Python\n")
    mtx.write("% Arguments: Nodes: {:d} Edges: {:d} Seed: {:f} Observed Probability: {:f}\n".format(args.nodes,
              args.arcs, args.seed, args.observed_probability))
    mtx.write("{:d}\t{:d}\t{:d}\n".format(args.nodes, args.nodes, num_arcs_written + 1))


def write_mtx_nodes_header(mtx, args):
    mtx.write("% Belief network generated using Python\n")
    mtx.write("% Arguments: Nodes: {:d} Edges: {:d} Seed: {:f} Observed Probability: {:f}\n".format(args.nodes,
                                                                                                args.arcs, args.seed, args.observed_probability))
    mtx.write("{:d}\t{:d}\n".format(args.nodes, args.nodes))



def generate_file(args):
    random.seed(args.seed)

    temp_file = "{}.mtx.tmp".format(args.file)
    mtx_nodes_file = "{}.nodes.mtx".format(args.file)
    mtx_edges_file = "{}.edges.mtx".format(args.file)

    num_arcs_written = 0

    with open(args.file, 'w') as f:
        with open(temp_file, 'w') as mtx_edges:
            with open(mtx_nodes_file, 'w') as mtx_nodes:
                # write header
                write_header(f)
                write_mtx_nodes_header(mtx_nodes, args)
                write_variables(f, args)
                write_observed_nodes(f, mtx_nodes, args)
                num_arcs_written = write_edges(f, mtx_edges, args)
                write_footer(f)

    with open(temp_file, 'r') as mtx_tmp:
        with open(mtx_edges_file, 'w') as mtx:
            write_mtx_edges_header(mtx, args, num_arcs_written)
            for c in mtx_tmp:
                mtx.write(c)

    os.remove(temp_file)


def main():
    args = parse_arguments()
    generate_file(args)


if __name__ == '__main__':
    main()
