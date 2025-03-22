using System.Collections.Concurrent;

namespace BetterGenshinImpact.Core.Recognition.ONNX;

public class BgiYoloPredictorFactory
{
    private static readonly ConcurrentDictionary<string, BgiYoloPredictor> Predictors = new();

    public static BgiYoloPredictor GetPredictor(string modelRelativePath)
    {
        if (Predictors.TryGetValue(modelRelativePath, out BgiYoloPredictor? value)) return value;
        value = new BgiYoloPredictor(modelRelativePath);
        Predictors[modelRelativePath] = value;
        return value;
    }
}