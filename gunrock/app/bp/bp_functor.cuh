/**
 * @file
 * bp_functor.cuh
 *
 * @brief Device functions for belief propagation problem
*/

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/bp/bp_problem.cuh>
#include <stdio.h>

namespace gunrock {
namespace app {
namespace bp {

/**
 * @brief Structure contains device functions in sample graph traverse.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam Problem     Problem data type which contains data slice for problem.
 * @tparam _LabelT     Vertex label type.
 *
 */
template<
        typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId>
    struct BPFunctor {
        typedef typename Problem
        ::DataSlice DataSlice;
        typedef _LabelT LabelT;


        /**
         * @brief Forward Edge Mapping condition function. Check if the destination node
         * has been claimed as someone else's child.
         *
         * @param[in] s_id Vertex Id of the edge source node
         * @param[in] d_id Vertex Id of the edge destination node
         * @param[out] d_data_slice Data slice object.
         * @param[in] edge_id Edge index in the output frontier
         * @param[in] input_item Input Vertex Id
         * @param[in] label Vertex label value.
         * @param[in] input_pos Index in the input frontier
         * @param[out] output_pos Index in the output frontier
         *
         * @return Whether to load the apply function for the edge and include the destination node in the next frontier.
         */
        static __device__ __forceinline__ bool CondEdge(
                VertexId s_id,
                VertexId d_id,
                DataSlice *d_data_slice,
                SizeT edge_id,
                VertexId input_item,
                LabelT label,
                SizeT input_pos,
                SizeT &output_pos
        )
        {
            return true;
        }

        /**
         * @brief Forward Edge Mapping apply function. Now we know the source node
         * has succeeded in claiming child, so it is safe to set label to its child node
         * (destination node).
         *
         * @param[in] s_id Vertex Id of the edge source node
         * @param[in] d_id Vertex Id of the edge destination node
         * @param[in] d_data_slice Data slice object
         * @param[in] edge_id Edge index in the output frontier
         * @param[in] input_item Input Vertex Id
         * @param[in] label Vertex label value
         * @param[in] input_pos Index in the input frontier
         * @param[in] output_pos Index in the output frontier
         */
        static __device__ __forceinline__ void ApplyEdge(
                VertexId s_id,
                VertexId d_id,
                DataSlice *d_data_slice,
                SizeT edge_id,
                VertexId input_item,
                LabelT label,
                SizeT input_pos,
                SizeT &output_pos
        )
        {
            Value src_belief = d_data_slice->belief_curr[s_id];
            Value joint_belief;
            util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(joint_belief,
                                                                      d_data_slice->joint_probabilities + edge_id);
            Value mul_value = joint_belief * src_belief;
            if (isfinite(mul_value)) {
                Value old_value = atomicMul(d_data_slice->belief_next + d_id, mul_value);
            }
        }

        /**
         * @brief filter condition function. Check if the Vertex Id is valid (not equal to -1).
         * Beliefs will be normalized when a source node ID is set.
         *
         * @param[in] v auxiliary value
         * @param[in] node Vertex identifier
         * @param[in] d_data_slice Data slice object
         * @param[in] nid Vertex index
         * @param[in] label Vertex label value
         * @param[in] input_pos
         * @param[in] output_pos
         * @return Whether to load the apply function for the node and
         * include it in the outgoing vertex frontier
         */
        static __device__ __forceinline__ bool CondFilter(
                VertexId v,
                VertexId node,
                DataSlice *d_data_slice,
                SizeT nid,
                LabelT label,
                SizeT input_pos,
                SizeT output_pos
        ) {
            if (node == -1)
            {
                return false;
            }
            Value new_beliefs = d_data_slice->belief_next[node];
            Value curr_beliefs = d_data_slice->belief_curr[node];
            // handle overflow
            if (!isfinite(new_beliefs) || new_beliefs > 1.0f || new_beliefs < 0.0f ) {
                return false;
            }
            d_data_slice->belief_curr[node] = new_beliefs;
            return (fabs(new_beliefs - curr_beliefs) > (d_data_slice->threshold * curr_beliefs));
        }

        /**
         * @brief filter apply function. Doing nothing for BP problem
         *
         * @param[in] v auxiliary value.
         * @param[in] node Vertex identifier.
         * @param[in] d_data_slice Data slice object.
         * @param[in] nid Vertex index.
         * @param[in] label Vertex label value.
         * @param[in] input_pos Index in the input frontier
         * @param[in] output_pos Index in the output frontier
         */
        static __device__ __forceinline__ void ApplyFilter(
                VertexId v,
                VertexId node,
                DataSlice *d_data_slice,
                SizeT nid,
                LabelT label,
                SizeT input_pos,
                SizeT output_pos
        ) {
            // do nothing
        }
    };
}
}
}