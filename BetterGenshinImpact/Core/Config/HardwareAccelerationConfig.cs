using System;
using BetterGenshinImpact.Core.Recognition;
using BetterGenshinImpact.Core.Recognition.ONNX;
using CommunityToolkit.Mvvm.ComponentModel;

namespace BetterGenshinImpact.Core.Config;

[Serializable]
public partial class HardwareAccelerationConfig : ObservableObject
{
    /// <summary>
    /// 推理使用的设备
    /// </summary>
    [ObservableProperty]
    private DeviceType _inferenceDevice = DeviceType.Cpu;

    /// <summary>
    /// cuda所使用的设备ID
    /// </summary>
    [ObservableProperty]
    private int _cudaDeviceId = 0;

    /// <summary>
    /// DML所使用的设备ID
    /// </summary>
    [ObservableProperty]
    private int _dmlDeviceId = 0;

    /// <summary>
    /// 是否输出优化后的模型文件到缓存。注意:在不支持的执行器上使用会导致异常。
    /// </summary>
    [ObservableProperty]
    private bool _optimizedModel = false;
    
    /// <summary>
    /// 文字识别引擎
    /// - Paddle
    /// - Yap
    /// </summary>
    [ObservableProperty]
    private OcrEngineTypes _ocrEngine = OcrEngineTypes.Paddle;
}