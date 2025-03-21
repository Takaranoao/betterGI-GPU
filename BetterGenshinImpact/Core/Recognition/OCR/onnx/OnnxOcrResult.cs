using System;
using System.Linq;

namespace BetterGenshinImpact.Core.Recognition.OCR;

public class OnnxOcrResult : IOcrResult
{
    public OcrResultRegion[] Regions { get; }

    public string Text
    {
        get
        {
            return string.Join("\n", Regions.OrderBy(x => x.Rect.Center.Y)
                .ThenBy(x => x.Rect.Center.X)
                .Select((Func<OcrResultRegion, string>)(x => x.Text)));
        }
    }
}