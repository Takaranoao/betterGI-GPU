using OpenCvSharp;

namespace BetterGenshinImpact.Core.Recognition.OCR;

public interface IOcrResult
{
    public OcrResultRegion[] Regions { get; }
    public string Text { get; }
}