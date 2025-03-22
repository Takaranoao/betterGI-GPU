using System;
using System.Collections.Generic;
using System.Threading;
using BetterGenshinImpact.Core.Recognition.OCR.onnx.yap;
using BetterGenshinImpact.GameTask;

namespace BetterGenshinImpact.Core.Recognition.OCR;

public class OcrFactory
{
    public static IOcrService Ocr =>
        Get(TaskContext.Instance().Config.HardwareAccelerationConfig.OcrEngine);

    private static YapOnnxOcrService? _yapOnnxOcrService;
    private static PaddleOcrService? _paddleOcrService;


    public static IOcrService Get(OcrEngineTypes ocrEngine)
    {
        return ocrEngine switch
        {
            OcrEngineTypes.Yap => _yapOnnxOcrService ??= new YapOnnxOcrService(),
            OcrEngineTypes.Paddle => _paddleOcrService ??= new PaddleOcrService(),
            _ => throw new ArgumentOutOfRangeException(nameof(ocrEngine), ocrEngine, null)
        };
    }
}