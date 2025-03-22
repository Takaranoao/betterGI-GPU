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
    public readonly SessionOptions Options;
    public readonly FeatureType[] FeatureTypes;
   public BgiSessionOption()
    {
        switch (TaskContext.Instance().Config.InferenceDevice)
        {
            case DeviceType.Cpu:
                Options = new SessionOptions();
                FeatureTypes = [FeatureType.Cpu];
                break;
            case DeviceType.GpuAuto:
                Options = MakeSessionOptionWithAuto(out var featureTypes);
                FeatureTypes = featureTypes;
                break;
            default:
                throw new InvalidEnumArgumentException("无效的推理设备");
        }
    }

    public static SessionOptions MakeSessionOptionWithAuto(out FeatureType[] featureTypes)
    {
        List<FeatureType> features = new List<FeatureType>();
        SessionOptions? sessionOptions = null;
        try
        {
            sessionOptions = SessionOptions.MakeSessionOptionWithTensorrtProvider();
            features.Add(FeatureType.TensorRt);
            features.Add(FeatureType.Cuda);
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
                features.Add(FeatureType.Cuda);
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
                features.Add(FeatureType.Dml);
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
        features.Add(FeatureType.Cpu);
        featureTypes = features.ToArray();
        return sessionOptions;
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