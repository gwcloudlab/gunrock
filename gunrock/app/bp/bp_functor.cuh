

/**
 * @file
 * bp_functor.cuh
 *
 * @brief Device functions for BP problem
*/

#pragma once

#include <gunrock/util/device_intrinsics.cuh>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/bp/bp_problem.cuh>

namespace gunrock {
    namespace app {
        namespace bp {

            /**
             * @brief Structure contains device functions in BP graph traverse
             *
             * @tparam VertexId Type of signed interger to use as vertex identifie
             */
            template <
                    typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId>
            struct BPMarkerFunctor
            {
                typedef typename Problem::DataSlice DataSlice;
                typedef _LABELT LabelT;

                /**
                 * @brief Forward Edge Mapping condition function. Check if the destination node has been claimed as someone else's child
                 *
                 * @param[in] s_id Vertex Id of the edge source node
                 * @param[in] d_id Vertex Id of the edge destination node
                 * @param[in] d_data_slice Data slice object
                 * @param[in] edge_id Edge index in the output frontier
                 * @param[in] input_item Input Vertex Id
                 * @param[in] label Vertex label value.
                 * @param[in] input_pos Index in the input frontier
                 * @param[in] output_pos Index in the output frontier
                 *
                 * @return Whether to load the apply function for the edge and include the destination node in the next frontier
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
                 * has succeeded in claiming child, so it is safe to set the label to its child node
                 * (destination node).
                 *
                 * @param[in] s_id Vertex Id of the edge source node
                 * @param[in] d_id Vertex Id of the edge destination node
                 * @param[in] d_data_slice Data slice object.
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
                ){
                    d_data_slice->markers[d_id] = 1;
                }
            };

            template <typename T>
            struct Make4Vector
            {
                typedef util::Array1D<SizeT, int> V4;
            };

            template <>
            struct Make4Vector<float>
            {
                typedef util::Array1D<SizeT, float> V4;
            };

            template <>
            struct Make4Vector<double>
            {
                typedef util::Array1D<SizeT, double> V4;
            };

            template <
                    typename VertexId, typename SizeT, typename Value, typename Problem>
                    struct BPFunctor
                    {
                        typedef typename Problem::DataSlice DataSlice;
                        typedef typename Make4Vector<Value>::V4 LabelT;

                        /**
                         * @brief Forward Edge Mapping condition function. Check if the destination node has been
                         * claimed as someone else's child.
                         *
                         * @param[in] s_id Vertex Id of the edge source node
                         * @param[in] d_id  Vertex Id of the edge destination node
                         * @param[in] d_data_slice Data slice object
                         * @param[in] edge_id Edge index in the output frontier
                         * @param[in] input_item Input Vertex Id
                         * @param[in] label Vertex label value.
                         * @param[in] input_pos Index in the input frontier
                         * @param[in] output_pos Index in the output frontier
                         * @return Whether to load the apply function for the edge and
                         * include the destination node in the next frontier
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
                        ){
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
                        ){
                            util::Array1D<SizeT, Value> src_belief = d_data_slice->belief_curr[s_id];
                            util::Array1D<SizeT, Value> dest_belief = d_data_slice->belief_next[d_id];
                            util::Array1D<SizeT, Value> joint_probability = d_data_slice->joint_probability[edge_id];

                            for (int i = 0; i < src_belief.GetSize(); i++)
                            {
                                for (int j = 0; j < dest_belief.GetSize(); j++) {
                                    int prob_index = i * dest_belief.GetSize() + j;
                                    Value mul_value = *(joint_probability.GetPointer(util::DEVICE) + prob_index) * *(src_belief.GetPointer(util::DEVICE) + i);
                                    if (isfinite(mul_value)) {
                                        Value old_value = atomicMul(dest_belief.GetPointer() + j, mul_value);
                                    }
                                }
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
                            util::Array1D raw_beliefs = d_data_slice->belief_next[node];
                            util::Array1D curr_beliefs = d_data_slice->belief_curr[node];
                            Value normalized_sum = 0.0;
                            Value next_sum = 0.0;
                            Value curr_sum = 0.0;
                            for (int i = 0; i < raw_beliefs.GetSize(); i++) {
                                normalized_sum += *(raw_beliefs.GetPointer(util::DEVICE) + i);
                            }
                            for (int i = 0; i < raw_beliefs.GetSize(); i++) {
                                Value normalized_value = *(raw_beliefs.GetPointer(util::DEVICE) + i) / normalized_sum;
                                if (!isfinite(normalized_value)) {
                                    normalized_value = 0.0;
                                }
                                *(raw_beliefs.GetPointer(util::DEVICE) + i) = normalized_value;
                                next_sum += normalized_value;
                                curr_sum += *(curr_beliefs.GetPointer(util::DEVICE) + i);
                            }
                            return (fabs(next_sum - curr_sum) > (d_data_slice->threshold * curr_sum));
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
                        )
                        {
                            // do nothing
                        }
                    };
        }
    }
}