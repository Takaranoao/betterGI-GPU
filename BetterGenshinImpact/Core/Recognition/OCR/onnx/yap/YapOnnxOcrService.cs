using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using BetterGenshinImpact.Core.Config;
using BetterGenshinImpact.Core.Recognition.ONNX;
using BetterGenshinImpact.Core.Recognition.OpenCv;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Diagnostics;
using System.Text;

namespace BetterGenshinImpact.Core.Recognition.OCR.onnx.yap;

public class YapOnnxOcrService : IOcrService
{
    private readonly InferenceSession _session;
    private readonly Dictionary<int, string> _wordDictionary;

    public YapOnnxOcrService()
    {
        const string relativePath = @"Assets\Model\Yap\model_training.onnx";
        if (!File.Exists(Global.Absolute(relativePath)))
            throw new FileNotFoundException("Yap模型不存在", Global.Absolute(relativePath));
        _session = BgiOnnxFactory.CreateInferenceSession(relativePath);

        var wordJsonPath = Global.Absolute(@"Assets\Model\Yap\index_2_word.json");
        if (!File.Exists(wordJsonPath)) throw new FileNotFoundException("Yap字典文件不存在", wordJsonPath);

        var json = File.ReadAllText(wordJsonPath);
        _wordDictionary = JsonSerializer.Deserialize<Dictionary<int, string>>(json) ??
                          throw new Exception("index_2_word.json deserialize failed");
    }

    public string Ocr(Mat mat)
    {
        return OcrResult(mat).Text;
    }

    public string OcrWithoutDetector(Mat mat)
    {
        long startTime = Stopwatch.GetTimestamp();
        // 将输入数据调整为 (1, 1, 32, 384) 形状的张量

        var reshapedInputData = ToTensorYapDnn(mat, out var owner);

        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;

        using (owner)
        {
            // 创建输入 NamedOnnxValue, 运行模型推理
            results = _session.Run([NamedOnnxValue.CreateFromTensor("input", reshapedInputData)]);
        }

        using (results)
        {
            // 获取输出数据
            var boxes = results[0].AsTensor<float>();

            var ans = new StringBuilder();
            var lastWord = default(string);
            for (var i = 0; i < boxes.Dimensions[0]; i++)
            {
                var maxIndex = 0;
                var maxValue = -1.0;
                for (var j = 0; j < _wordDictionary.Count; j++)
                {
                    var value = boxes[[i, 0, j]];
                    if (value > maxValue)
                    {
                        maxValue = value;
                        maxIndex = j;
                    }
                }

                var word = _wordDictionary[maxIndex];
                if (word != lastWord && word != "|")
                {
                    ans.Append(word);
                }

                lastWord = word;
            }

            TimeSpan time = Stopwatch.GetElapsedTime(startTime);
            string result = ans.ToString();
            Debug.WriteLine($"Yap模型识别 耗时{time.TotalMilliseconds}ms 结果: {result}");
            return result;
        }
    }

    public IOcrResult OcrResult(Mat mat)
    {
        return OcrFactory.Ocr.OcrResult(mat);
    }

    /// <summary>
    /// 预处理速度比unsafe快5倍以上,且吃的资源还少
    /// </summary>
    /// <param name="inputImage">输入图像，若不是灰度图会转换</param>
    /// <param name="tensorMemoryOwnser">tensor的Memory，用完需要释放</param>
    /// <returns></returns>
    private static Tensor<float> ToTensorYapDnn(Mat inputImage, out IMemoryOwner<float> tensorMemoryOwnser)
    {
        using var rt = new ResourcesTracker();
        Mat dst;
        // 221*32是个什么鬼
        if (inputImage.Channels() > 1)
        {
            var resize = rt.T(ResizeHelper.ResizeTo(inputImage, 221, 32));
            dst = rt.NewMat(resize.Size(), MatType.CV_8UC1, Scalar.Black);
            Cv2.CvtColor(resize, dst, ColorConversionCodes.BGR2GRAY);
        }
        else
        {
            dst = rt.T(ResizeHelper.ResizeTo(inputImage, 221, 32));
        }

        // 填充到 384x32
        var padded = rt.NewMat(new Size(384, 32), MatType.CV_8UC1, Scalar.Black);
        padded[new Rect(0, 0, 221, 32)] = dst;
        // 使用向量运算代替循环
        var blob = rt.T(CvDnn.BlobFromImage(padded, 1.0 / 255.0, default, default, false, false));
        var nCols = padded.Cols * padded.Rows;
        tensorMemoryOwnser = MemoryPool<float>.Shared.Rent(nCols);
        // 内存复制，如果直接传指针构建的话速度还不如多复制一份
        blob.AsSpan<float>().CopyTo(tensorMemoryOwnser.Memory.Span);
        return new DenseTensor<float>(tensorMemoryOwnser.Memory[..nCols], [1, 1, 32, 384]);
    }

    #region unsafe区域(弃用)

    [Obsolete("使用CV DNN替代")]
    public static Tensor<float> ToTensorUnsafe(Mat src, out IMemoryOwner<float> tensorMemoryOwnser)
    {
        var channels = src.Channels();
        var nRows = src.Rows;
        var nCols = src.Cols * channels;
        if (src.IsContinuous())
        {
            nCols *= nRows;
            nRows = 1;
        }

        //var inputData = new float[nCols];
        tensorMemoryOwnser = MemoryPool<float>.Shared.Rent(nCols);
        var memory = tensorMemoryOwnser.Memory[..nCols];
        unsafe
        {
            for (var i = 0; i < nRows; i++)
            {
                var b = (byte*)src.Ptr(i);
                for (var j = 0; j < nCols; j++)
                {
                    memory.Span[j] = b[j] / 255f;
                }
            }
        }

        return new DenseTensor<float>(memory, [1, 1, 32, 384]);
    }

    #endregion
}