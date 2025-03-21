using OpenCvSharp;

namespace BetterGenshinImpact.Core.Recognition.OCR;

public record struct OcrResultRect(Rect Rect, string Text, float Score);