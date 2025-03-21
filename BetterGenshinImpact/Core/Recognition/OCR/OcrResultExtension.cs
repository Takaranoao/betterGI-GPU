using BetterGenshinImpact.GameTask.AutoSkip.Model;
using BetterGenshinImpact.View.Drawable;
using OpenCvSharp;
using Sdcb.PaddleOCR;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace BetterGenshinImpact.Core.Recognition.OCR;

public static class OcrResultExtension
{
    public static bool RegionHasText(this IOcrResult result, ReadOnlySpan<char> text)
    {
        foreach (ref readonly OcrResultRegion item in result.Regions.AsSpan())
        {
            if (item.Text.AsSpan().Contains(text, StringComparison.InvariantCulture))
            {
                return true;
            }
        }

        return false;
    }

    public static OcrResultRegion FindRegionByText(this IOcrResult result, ReadOnlySpan<char> text)
    {
        foreach (ref readonly OcrResultRegion item in result.Regions.AsSpan())
        {
            if (item.Text.AsSpan().Contains(text, StringComparison.InvariantCulture))
            {
                return item;
            }
        }

        return default;
    }

    public static Rect FindRectByText(this IOcrResult result, string text)
    {
        foreach (ref OcrResultRegion item in result.Regions.AsSpan())
        {
            if (item.Text.Contains(text))
            {
                return item.Rect.BoundingRect();
            }
        }

        return default;
    }

    public static List<RectDrawable> ToRectDrawableList(this IOcrResult result, Pen? pen = null)
    {
        return result.Regions.Select(item => item.Rect.BoundingRect().ToRectDrawable(pen)).ToList();
    }

    public static List<RectDrawable> ToRectDrawableListOffset(this IOcrResult result, int offsetX, int offsetY, Pen? pen = null)
    {
        return result.Regions.Select(item => item.Rect.BoundingRect().ToRectDrawable(offsetX, offsetY, pen)).ToList();
    }

    public static OcrResultRect ToOcrResultRect(this OcrResultRegion region)
    {
        return new OcrResultRect(region.Rect.BoundingRect(), region.Text, region.Score);
    }
}
