using System.Diagnostics;
using BetterGenshinImpact.Core.Recognition.OCR;
using BetterGenshinImpact.Core.Recognition.OCR.onnx.yap;
using BetterGenshinImpact.Core.Recognition.OpenCv;
using OpenCvSharp;

namespace BetterGenshinImpact.Test.Simple;

public class OcrTest
{
    public static void TestYap()
    {
        Mat mat = Cv2.ImRead(@"E:\HuiTask\更好的原神\临时文件\fuben_jueyuan.png", ImreadModes.Grayscale);
        var text = new YapOnnxOcrService().Ocr(mat);
        Debug.WriteLine(text);

        Mat mat2 = Cv2.ImRead(@"E:\HuiTask\更好的原神\临时文件\fuben_jueyuan.png", ImreadModes.Grayscale);
        var text2 = OcrFactory.Paddle.Ocr(mat2);
        Debug.WriteLine(text2);
    }
}
