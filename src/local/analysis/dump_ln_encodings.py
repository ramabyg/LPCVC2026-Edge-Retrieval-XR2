"""
Dump the calibrated int8 ranges AIMET assigned to LayerNormalization outputs in
an exported QDQ ONNX, to explain why quantizing LN collapses the image encoder.

Reads scale/zero_point off each QuantizeLinear that consumes a LayerNormalization
output. range = (qmin - zp) * scale .. (qmax - zp) * scale.

Run from the repo root:
    python src/local/analysis/dump_ln_encodings.py <qdq_model.onnx> [op_type]
"""
import sys
from collections import defaultdict

import numpy as np
import onnx
from onnx import numpy_helper


def main(path, op_type="LayerNormalization"):
    model = onnx.load(path, load_external_data=False)
    g = model.graph

    inits = {i.name: i for i in g.initializer}

    def const(name):
        if name in inits:
            return numpy_helper.to_array(inits[name])
        return None

    # tensor name -> producing node
    producer = {}
    for n in g.node:
        for o in n.output:
            producer[o] = n

    rows = []
    for n in g.node:
        if n.op_type != "QuantizeLinear":
            continue
        src = producer.get(n.input[0])
        if src is None or src.op_type != op_type:
            continue
        scale = const(n.input[1])
        zp = const(n.input[2]) if len(n.input) > 2 else None
        if scale is None:
            continue
        scale = np.asarray(scale).reshape(-1)
        zp_arr = np.asarray(zp).reshape(-1) if zp is not None else np.zeros_like(scale)
        # int8 vs uint8 from the zero_point dtype
        signed = zp is not None and zp.dtype == np.int8
        qmin, qmax = (-128, 127) if signed else (0, 255)
        lo = (qmin - zp_arr.astype(np.float64)) * scale.astype(np.float64)
        hi = (qmax - zp_arr.astype(np.float64)) * scale.astype(np.float64)
        rows.append((src.name or n.input[0], float(lo.min()), float(hi.max()),
                     float(scale.max()), "int8" if signed else "uint8"))

    if not rows:
        print(f"No QuantizeLinear found consuming a {op_type} output in {path}")
        return

    print(f"{len(rows)} quantized {op_type} outputs in {path}\n")
    print(f"  {'node':<42} {'min':>10} {'max':>10} {'step':>10}  dtype")
    print("  " + "-" * 84)
    for name, lo, hi, step, dt in rows:
        print(f"  {name:<42} {lo:>10.3f} {hi:>10.3f} {step:>10.5f}  {dt}")

    steps = np.array([r[3] for r in rows])
    spans = np.array([r[2] - r[1] for r in rows])
    print("\n  step  : min %.5f  median %.5f  max %.5f" % (steps.min(), np.median(steps), steps.max()))
    print("  span  : min %.3f  median %.3f  max %.3f" % (spans.min(), np.median(spans), spans.max()))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "LayerNormalization")
