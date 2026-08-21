"""
Make an AIMET QDQ ONNX loadable by the Hexagon HTP backend.

Problem it solves
-----------------
AIMET 2.34 quantizes weights per-channel *internally*, regardless of the
quantsim config. For `MatMul` that means a QuantizeLinear/DequantizeLinear pair
with `axis=1` and a (N,) scale vector on the weight. QAI Hub *compiles* such a
graph fine, but the DLC then fails on the device at graph build time:

    Failed to call QnnModel_composeGraphsFromDlc: MODEL_GRAPH_ERROR

QNN's per-channel (per-axis) weight support covers Conv/ConvTranspose and
FullyConnected; a per-axis MatMul weight is not composable on HTP. Folding those
weight quantizers down to per-tensor makes the graph legal.

Only *weight* (initializer-fed) quantizers are touched — activation quantizers
are already per-tensor and are left exactly as they are.

Usage
-----
    python src/local/fix_qdq_for_htp.py IN.onnx OUT.onnx            # MatMul only
    python src/local/fix_qdq_for_htp.py IN.onnx OUT.onnx --ops MatMul,Gemm,Conv
"""

import argparse
import collections

import numpy as np
import onnx
from onnx import numpy_helper


def per_tensor_fold(model, op_types):
    """Fold per-channel weight QDQ pairs feeding `op_types` down to per-tensor.

    The new scale is the max of the per-channel scales, which keeps every weight
    representable (no clipping) at coarser granularity. Weight quantizers are
    symmetric (int8, zero-point 0), so a scalar zero-point of 0 is exact.
    """
    graph = model.graph
    init = {i.name: i for i in graph.initializer}
    consumers = collections.defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            consumers[inp].append(node)

    folded = collections.Counter()
    for q in [n for n in graph.node if n.op_type == "QuantizeLinear"]:
        # weight quantizer = its data input is a constant initializer
        if q.input[0] not in init:
            continue
        axis_attrs = [a for a in q.attribute if a.name == "axis"]
        scale = init.get(q.input[1])
        if scale is None or numpy_helper.to_array(scale).size <= 1:
            continue  # already per-tensor

        dqs = [c for c in consumers[q.output[0]] if c.op_type == "DequantizeLinear"]
        targets = [c.op_type for dq in dqs for c in consumers[dq.output[0]]]
        if not targets or not set(targets) & op_types:
            continue

        scale_arr = numpy_helper.to_array(scale)
        new_scale = np.array(scale_arr.max(), dtype=scale_arr.dtype)
        zp = init[q.input[2]]
        zp_arr = numpy_helper.to_array(zp)
        if np.any(zp_arr != 0):
            raise SystemExit(
                f"{q.name}: per-channel weight quantizer is asymmetric "
                f"(zero-point != 0); per-tensor folding would shift the weights."
            )
        new_zp = np.array(0, dtype=zp_arr.dtype)

        scale.CopyFrom(numpy_helper.from_array(new_scale, scale.name))
        zp.CopyFrom(numpy_helper.from_array(new_zp, zp.name))
        for node in [q] + dqs:
            for a in list(node.attribute):
                if a.name == "axis":
                    node.attribute.remove(a)
        for t in set(targets) & op_types:
            folded[t] += 1
        del axis_attrs
    return folded


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--ops", default="MatMul",
                    help="comma-separated consumer op types whose per-channel weight "
                         "quantizers get folded to per-tensor (default: MatMul)")
    args = ap.parse_args()

    model = onnx.load(args.src)
    folded = per_tensor_fold(model, set(args.ops.split(",")))
    if not folded:
        print("No per-channel weight quantizers matched — nothing to fold.")
    else:
        print("Folded per-channel -> per-tensor weight quantizers:",
              dict(folded))
    onnx.save(model, args.dst)
    print(f"Wrote {args.dst}")


if __name__ == "__main__":
    main()
