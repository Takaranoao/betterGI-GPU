using System;
using BetterGenshinImpact.Core.Recognition.OCR.onnx.yap;

namespace BetterGenshinImpact.Core.Recognition.OCR;

public class OcrFactory
{
    // public static IOcrService Media = Create(OcrEngineTypes.Media);
    public static readonly IOcrService Paddle = Create(OcrEngineTypes.Paddle);

    public static readonly IOcrService Yap = Create(OcrEngineTypes.YapModel);


    private static IOcrService Create(OcrEngineTypes type)
    {
        return type switch
        {
            OcrEngineTypes.Paddle => new PaddleOcrService(),
            OcrEngineTypes.YapModel => new YapOnnxOcrService(),
            _ => throw new ArgumentOutOfRangeException(nameof(type), type, null),
        };
    }
}