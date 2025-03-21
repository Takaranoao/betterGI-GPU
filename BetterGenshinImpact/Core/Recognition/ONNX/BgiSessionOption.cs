using System;
using System.Collections.Generic;
using BetterGenshinImpact.GameTask;
using BetterGenshinImpact.Model;
using Microsoft.ML.OnnxRuntime;
using System.ComponentModel;
using Compunet.YoloSharp;
using Microsoft.Extensions.Logging;

namespace BetterGenshinImpact.Core.Recognition.ONNX;

public class BgiSessionOption : Singleton<BgiSessionOption>
{
    private static readonly ILogger<BgiSessionOption> Logger = App.GetLogger<BgiSessionOption>();

    private static readonly Dictionary<string, YoloPredictor> LoadedYoloPredictor =
        new Dictionary<string, YoloPredictor>();
    public static string[] InferenceDeviceTypes { get; } = ["CPU", "GPU_Auto", "GPU_DirectML"];

    public SessionOptions Options { get; set; } = TaskContext.Instance().Config.InferenceDevice switch
    {
        "CPU" => new SessionOptions(),
        "GPU_DirectML" => MakeSessionOptionWithDirectMlProvider(),
        "GPU_Auto" => MakeSessionOptionWithAuto(),
        _ => throw new InvalidEnumArgumentException("无效的推理设备")
    };

    public static SessionOptions MakeSessionOptionWithDirectMlProvider()
    {
        var sessionOptions = new SessionOptions();
        sessionOptions.AppendExecutionProvider_DML();
        return sessionOptions;
    }

    public static SessionOptions MakeSessionOptionWithAuto()
    {
        SessionOptions? sessionOptions = null;
        try
        {
            sessionOptions = SessionOptions.MakeSessionOptionWithTensorrtProvider();
            Logger.LogInformation("启用GPU推理: TensorRT");
        }
        catch (Exception e)
        {
            Logger.LogDebug("无法加载TensorRT。可能不支持，跳过。({Err})", e.Message);
        }

        if (sessionOptions is null)
        {
            // 优先使用Tensorrt
            try
            {
                sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider();
                Logger.LogInformation("启用GPU推理: CUDA");
            }
            catch (Exception ex)
            {
                Logger.LogDebug("无法加载CUDA Session。可能不支持，跳过。({Err})", ex.Message);
            }
        }


        if (sessionOptions == null)
        {
            sessionOptions = new SessionOptions();
            try
            {
                sessionOptions.AppendExecutionProvider_DML();
                Logger.LogInformation("启用GPU推理: DML");
            }
            catch (Exception ex)
            {
                Logger.LogDebug("无法加载DML Session。可能不支持，跳过。({Err})", ex.Message);
                sessionOptions.Dispose();
                sessionOptions = null;
            }
        }

        if (sessionOptions == null)
        {
            Logger.LogInformation("GPU加速已禁用");
            sessionOptions = new SessionOptions();
        }

        sessionOptions.AppendExecutionProvider_CPU();
        return sessionOptions;
    }

    public YoloPredictorOptions YoloPredictorOptions()
    {
        return new YoloPredictorOptions()
        {
            SessionOptions = Options
        };
    }

    public YoloPredictor MakeYoloPredictor(string path)
    {
        if (LoadedYoloPredictor.TryGetValue(path,out var result))
        {
            return result;
        }
        var r = new YoloPredictor(path, YoloPredictorOptions());
        LoadedYoloPredictor.TryAdd(path, r);
        return r;

    }

    // /// <summary>
    // /// 重新加载每个推理器（测试没用，只能重启）
    // /// </summary>
    // public void RefreshInference()
    // {
    //     // 自动秘境每次都会NEW不用管
    //     // Yap、自动钓鱼
    //     GameTaskManager.RefreshTriggerConfigs();
    // }
}