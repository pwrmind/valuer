optimum-cli export onnx --model ./clean_model --task text-classification ./onnx_temp/

optimum-cli onnxruntime quantize --onnx_model ./onnx_temp/ --output ./browser_model/ --per_channel --arm64