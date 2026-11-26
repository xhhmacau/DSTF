# Third-party notices

The MIT grant in [LICENSE](LICENSE) covers original DSTF-Bench code and documentation. Upstream code remains subject to the rights and terms that apply to that code. The benchmark data are available through the [dataset repository](https://github.com/xhhmacau/Datasets) under the applicable dataset-source terms.

## Components with identified upstream terms

| Component | DSTF files or usage | Upstream license or notice |
| --- | --- | --- |
| [Time-Series-Library](https://github.com/thuml/Time-Series-Library) | Shared utilities and layers, including `utils/augmentation.py`, `utils/dtw.py` and `layers/Time_MICN_models/Conv_Blocks.py`. See the N-BEATS row for the separate notice in `utils/losses.py`. | [MIT](LICENSES/Time-Series-Library.txt) |
| [Element AI / N-BEATS](https://github.com/ServiceNow/N-BEATS) | `utils/losses.py`, also present in Time-Series-Library, carries the Element AI copyright and license notice. | [CC BY-NC 4.0 notice](LICENSES/N-BEATS.txt) |
| [Mammoth](https://github.com/aimagelab/mammoth/tree/neurips2020) | Original Mammoth contributions identified by the copyright header in `utils/buffer.py`. | [MIT](LICENSES/Mammoth.txt) |
| [RevIN](https://github.com/ts-kim/RevIN) | RevIN implementation referenced in `layers/OLinear/RevIN.py`. | [MIT](LICENSES/RevIN.txt) |
| [Flowformer](https://github.com/thuml/Flowformer) | Attention implementation referenced in `layers/OLinear/SelfAttention_Family.py`. | [MIT](LICENSES/Flowformer.txt) |
| [FlashAttention-PyTorch](https://github.com/shreyansh26/FlashAttention-PyTorch) | Attention implementation referenced in `layers/OLinear/SelfAttention_Family.py`. | [MIT](LICENSES/FlashAttention-PyTorch.txt) |
| [Informer2020](https://github.com/zhouhaoyi/Informer2020) | Attention implementation referenced in `layers/OLinear/SelfAttention_Family.py`. | [Apache 2.0](LICENSES/Informer2020.txt) |
| [CKA-Centered-Kernel-Alignment](https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment) | Implementation referenced in `utils/CKA.py` and `utils/layers/cka.py`. | [MIT](LICENSES/CKA-Centered-Kernel-Alignment.txt) |
| [GLIDE](https://github.com/openai/glide-text2im) | Code referenced in `utils/aurora_modules/flow_loss.py`. | [MIT](LICENSES/GLIDE.txt) |
| [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) | Expert utilities referenced in `utils/layers/moe.py`. | [Apache 2.0](LICENSES/Tensor2Tensor.txt) |

## Source attribution

Some files, including `adapter/module/base.py`, `models/FSNet.py`, `models/OneNet.py` and `utils/buffer.py`, match files in the [OnlineTSF repository](https://github.com/SJTU-DMTai/OnlineTSF).
