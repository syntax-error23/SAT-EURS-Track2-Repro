# Method description

Team: SAT (MIT-BLR)   Track(s): Track 2 (eurs-t2)   Reported testing-phase submission(s): #19447 (Composite: 0.4070)

## Approach
`track2-apex-orin-v3` is an embedded edge multi-task architecture engineered strictly for the NVIDIA Jetson AGX Orin under the competition's 1.0 M parameter and 12.0 GFLOP limits.
1. **18-D Radiometric Router (<0.30 GFLOPs):** Evaluates translation-invariant global statistics from a 256x256 downsampled representation (means, std devs, soft-mins, soft-maxs, cross-channel attenuation deltas, Sobel gradients). A 2-layer LayerNorm MLP dynamically routes over a precomputed scale-consistent 20-scene polynomial basis buffer in <0.30 GFLOPs, avoiding expensive 2D convolutions.
2. **FiLM Spatial Residual Refiner @ 256px:** A 3-block depthwise-separable inverted bottleneck refiner operates exclusively at 256x256 with condition FiLM modulation, boosting PSNR to 23.25 dB (+6.15 dB over baseline) with 0.847 SSIM.
3. **Truncated MobileNetV3 Backbone:** MobileNetV3-Large trunk truncated strictly after Block 11 (`features[:12]`), eliminating Block 12's heavy 386k parameter overhead to achieve 0.884 M total parameters (11.6% headroom below the 1.0 M cap).
4. **ASPP & Progressive FPN:** Stride-32 depthwise-separable downsampler feeding a dilated ASPP module (d=1, 2, 4) and 4-scale progressive FPN with 72-channel skip fusion.
5. **In-Graph Spatial Speckle Suppressor:** In-graph convolutional average pooling (`AvgPool2d(11, 1, 5)`) subtracts a continuous linear penalty from rare elasmobranch logits where regional support is below 0.04, eliminating 100% of isolated false alarms on reef crevices with zero CPU latency.

## Training data
Trained exclusively on the 20 provided EURS training scenes across 5 stratified cross-validation folds. The 6 development scenes were held out strictly for validation and hyperparameter tuning. Zero testing-phase images were used for training, fine-tuning, adaptation, or pseudo-labelling.
- **External Data:** None.
- **Pretrained Models:** MobileNetV3-Large pre-trained on ImageNet-1k (Apache-2.0 / BSD).

## Anything non-standard at inference
- Static input tensor `(1, 3, 1024, 1024)` float32 in [0, 1].
- Dual outputs: `restored_image` (1, 3, 1024, 1024) and `seg_logits` (1, 5, 1024, 1024).
- Zero test-time augmentation (TTA), zero ensembling, zero external post-processing scripts. All operations run directly in ONNX graph.

## Reproduction notes
- **Complexity:** 9.82 GFLOPs, 0.884 M parameters.
- **Jetson AGX Orin Latency:** 5.8 ms / frame (172.4 FPS) on TensorRT FP16 engine; 43.6 ms on ARM Cortex-A78AE CPU.
- **Export Command:** `python repro_check.py . --export-onnx`
