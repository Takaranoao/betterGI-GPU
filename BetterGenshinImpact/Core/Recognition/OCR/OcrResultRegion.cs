using OpenCvSharp;

namespace BetterGenshinImpact.Core.Recognition.OCR;

public record struct OcrResultRegion(RotatedRect Rect, string Text, float Score);