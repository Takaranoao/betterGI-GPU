using System;
using BetterGenshinImpact.Core.Config;
using BetterGenshinImpact.GameTask.Model.Area;
using Compunet.YoloSharp;
using OpenCvSharp;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text.Json;
using BetterGenshinImpact.View.Drawable;

namespace BetterGenshinImpact.Core.Recognition.ONNX;

public class BgiYoloPredictor(string modelRelativePath) : IDisposable
{
    private readonly YoloPredictor _predictor = new(Global.Absolute(modelRelativePath), new YoloPredictorOptions
    {
        SessionOptions = BgiSessionOption.Instance.Options
    });

    public YoloPredictor Predictor => _predictor;

    /// <summary>
    /// 检测
    /// </summary>
    /// <param name="region">图像</param>
    /// <returns>类别-矩形框</returns>
    public Dictionary<string, List<Rect>> Detect(ImageRegion region)
    {
        using var memoryStream = new MemoryStream();
        region.SrcBitmap.Save(memoryStream, ImageFormat.Bmp);
        memoryStream.Seek(0, SeekOrigin.Begin);
        var result = _predictor.Detect(memoryStream);


        var dict = new Dictionary<string, List<Rect>>();
        foreach (var box in result)
        {
            if (!dict.ContainsKey(box.Name.Name))
            {
                dict[box.Name.Name] = [new Rect(box.Bounds.X, box.Bounds.Y, box.Bounds.Width, box.Bounds.Height)];
            }
            else
            {
                dict[box.Name.Name].Add(new Rect(box.Bounds.X, box.Bounds.Y, box.Bounds.Width, box.Bounds.Height));
            }
        }

        Debug.WriteLine("YOLOv8识别结果:" + JsonSerializer.Serialize(dict));

        var list = result
            .Select(box => new Rect(box.Bounds.X, box.Bounds.Y, box.Bounds.Width, box.Bounds.Height))
            .Select(rect => region.ToRectDrawable(rect, modelRelativePath)).ToList();

        VisionContext.Instance().DrawContent.PutOrRemoveRectList(modelRelativePath, list);

        return dict;
    }

    public void Dispose()
    {
        _predictor.Dispose();
    }
}