using System.Linq;
using Sdcb.PaddleOCR;

namespace BetterGenshinImpact.Core.Recognition.OCR;

public class PaddleOcrResultAdapter(PaddleOcrResult paddleOcrResult) : IOcrResult
{
    public OcrResultRegion[] Regions => paddleOcrResult.Regions
        .Select(r => new OcrResultRegion(r.Rect, r.Text, r.Score))
        .ToArray();
    public string Text => paddleOcrResult.Text;
}