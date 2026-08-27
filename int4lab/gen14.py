import numpy as np, onnx, io, tarfile, json
from onnx import helper, TensorProto as TP

# graph: Xi[1,4,8,8] -> DQ(Wq,s,z) -> Conv -> y[1,8,6,6]
idx = np.arange(8*4*3*3)
codes = ((idx*7+3) % 16 - 8).astype(np.int8).reshape(8,4,3,3)
Xi = helper.make_tensor_value_info("Xi", TP.FLOAT, [1, 4, 8, 8])
y = helper.make_tensor_value_info("y", TP.FLOAT, [1, 8, 6, 6])
def build(wq_bytes, wdtype, name):
    inits = [
        helper.make_tensor("Wq", wdtype, [8,4,3,3], wq_bytes, raw=True),
        helper.make_tensor("Ws", TP.FLOAT, [], np.float32(0.5).tobytes(), raw=True),
        helper.make_tensor("Wz", TP.INT8, [], np.int8(0).tobytes(), raw=True),
    ]
    nodes = [
        helper.make_node("DequantizeLinear", ["Wq","Ws","Wz"], ["Wdq"], name="dq_w"),
        helper.make_node("Conv", ["Xi","Wdq"], ["y"], name="conv0"),
    ]
    vi = [helper.make_tensor_value_info("Wdq", TP.FLOAT, [8,4,3,3])]
    g = helper.make_graph(nodes, "g", [Xi], [y], inits, value_info=vi)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, name)

build(codes.tobytes(), TP.INT8, "/home/kram/int4lab/build1/min_w8.onnx")           # control (int8 full? int4-range but int8 dtype)
full = ((idx.astype(np.int32)*37) % 255 - 127).astype(np.int8).reshape(8,4,3,3)     # full-range control too
build(full.tobytes(), TP.INT8, "/home/kram/int4lab/build1/min_full8.onnx")

# calibration tar: 4 samples Xi
rng = np.random.default_rng(7)
with tarfile.open("/home/kram/int4lab/build1/ca_xi.tar", "w") as tf:
    for i in range(4):
        d = (rng.standard_normal((1,4,8,8))*0.5).astype(np.float32).tobytes()
        info = tarfile.TarInfo(name=f"{i}.bin"); info.size = len(d)
        tf.addfile(info, io.BytesIO(d))

cfg = {
  "model_type": "ONNX", "npu_mode": "NPU1",
  "quant": {"input_configs": [{"tensor_name": "Xi", "calibration_dataset": "./ca_xi.tar", "calibration_size": 4, "calibration_format": "Numpy"}], "calibration_method": "MinMax", "precision_analysis": False},
  "input_processors": [{"tensor_name": "Xi", "tensor_format": "RAW", "src_format": "RAW", "src_dtype": "F32", "src_layout": "NCHW", "csc_mode": "NoCSC"}],
  "compiler": {"check": 0},
}
json.dump(cfg, open("/home/kram/int4lab/build1/cfg.json","w"), indent=2)
print("wrote min_w8/min_full8 + calib + cfg")
