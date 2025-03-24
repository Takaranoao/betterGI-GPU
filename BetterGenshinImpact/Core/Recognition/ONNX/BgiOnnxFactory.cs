using System.Collections.Concurrent;
using BetterGenshinImpact.Core.Config;
using Microsoft.ML.OnnxRuntime;

namespace BetterGenshinImpact.Core.Recognition.ONNX;

public class BgiOnnxFactory
{
    // private static readonly ConcurrentDictionary<string, BgiYoloPredictor> Predictors = new();

    public static BgiYoloPredictor CreateYoloPredictor(string modelRelativePath)
    {
        // if (Predictors.TryGetValue(modelRelativePath, out BgiYoloPredictor? value)) return value;
        return   new BgiYoloPredictor(modelRelativePath);
        // Predictors[modelRelativePath] = value;
    }

    public static InferenceSession CreateInferenceSession(string modelRelativePath)
    {
        var cachePath = BgiSessionOption.Instance.GetCachedModelPath(modelRelativePath);
       return new InferenceSession(Global.Absolute(cachePath),
            BgiSessionOption.Instance.MakeSessionOption(cachePath));
    }
}