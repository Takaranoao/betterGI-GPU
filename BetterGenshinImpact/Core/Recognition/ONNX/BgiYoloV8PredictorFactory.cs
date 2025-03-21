using System.Collections.Generic;

namespace BetterGenshinImpact.Core.Recognition.ONNX;

public class BgiYoloPredictorFactory
{
    static Dictionary<string, BgiYoloPredictor> _predictors = new();

    public static BgiYoloPredictor GetPredictor(string modelRelativePath)
    {
        if (!_predictors.ContainsKey(modelRelativePath))
        {
            _predictors[modelRelativePath] = new BgiYoloPredictor(modelRelativePath);
        }

        return _predictors[modelRelativePath];
    }
}