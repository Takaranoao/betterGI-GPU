# TenserRT分支

OCR暂时没做完

-------------------

### 使用方法:

- 参照 [installing-tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#zip-file-installation)
安装 TenserRT
- 安装 [CUDA](https://developer.nvidia.com/cuda-downloads) 12.8+, [cuDNN](https://developer.nvidia.com/cudnn-downloads)
  9.8.0+

#### 建议:

cuda安装时候请在安装选项中选择 自定义 ，然后在自定义安装选项界面仅需要勾选 CUDA 下面的 Runtime 即可。

cuDNN同理。安装时候也只需要勾选 cuDNN for CUDA cuda `你安装的cuda版本` 下的 Runtime 即可。

> Nvidia家默认安装会给你安一坨

#### 注意

TenserRT的Windows版本在Nvidia官网下载可能需要注册个Nvidia的账号

TenserRT安装后注意配置好PATH或者参照上面的文档跟CUDA放在一起

TenserRT需要更长的预热时间，模型缓存做了但是预热没有做，第一次调用模型会卡到爆炸，但是之后不会。
