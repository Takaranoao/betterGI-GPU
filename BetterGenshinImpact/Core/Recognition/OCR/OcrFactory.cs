using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;
using BetterGenshinImpact.GameTask;
using Microsoft.Extensions.Logging;

namespace BetterGenshinImpact.Core.Recognition.OCR;

public class OcrFactory
{
    // public static IOcrService Media = Create(OcrEngineTypes.Media);
    private static readonly ILogger<OcrFactory> Logger = App.GetLogger<OcrFactory>();

    public static IOcrService Paddle => _ocrServices.TryGetValue(OcrEngineTypes.Paddle, out var value)
        ? value.Value
        : CreateAndSet(OcrEngineTypes.Paddle, TaskContext.Instance().Config.OtherConfig.GameCultureInfoName).Value;

    /// <summary>
    /// 保存着OcrEngineTypes和cultureInfoName与IOcrService
    /// </summary>
    private static readonly ConcurrentDictionary<OcrEngineTypes, KeyValuePair<string, IOcrService>>
        _ocrServices = new();

    /// <summary>
    /// 创建并设置
    /// </summary>
    /// <param name="type">OcrEngineTypes</param>
    /// <param name="cultureInfoName">文化名称</param>
    /// <returns>cultureInfoName与IOcrService的pair</returns>
    /// <exception cref="ArgumentOutOfRangeException">如果不能创建</exception>
    private static KeyValuePair<string, IOcrService> CreateAndSet(OcrEngineTypes type, string cultureInfoName)
    {
        var result = type switch
        {
            OcrEngineTypes.Paddle => new KeyValuePair<string, IOcrService>(cultureInfoName,
                new PaddleOcrService(cultureInfoName)),
            _ => throw new ArgumentOutOfRangeException(nameof(type), type, null)
        };
        var oldCultureInfoName = GetCultureInfoName(type);
        if (oldCultureInfoName is null)
        {
            Logger.LogDebug("为 {CultureInfoName} 创建了类型为 {Type} 的 OCR服务", result.Key, Enum.GetName(type));
        }
        else
        {
            Logger.LogDebug("将类型为 {Type} 的 OCR服务从 {OldCultureInfoName} 替换为 {NewCultureInfoName}", Enum.GetName(type),
                oldCultureInfoName, result.Key);
        }

        _ocrServices[type] = result;
        return result;
    }

    private static string? GetCultureInfoName(OcrEngineTypes type)
    {
        return _ocrServices.TryGetValue(type, out KeyValuePair<string, IOcrService> value) ? value.Key : null;
    }

    public static async Task ChangeCulture(string cultureInfoName)
    {
        await Task.Run(() =>
        {
            foreach (var ocrEngineTypes in Enum.GetValues<OcrEngineTypes>())
            {
                try
                {
                    // 避免重复创建OCR服务实例
                    var old = GetCultureInfoName(ocrEngineTypes);
                    if (old is null || !cultureInfoName.Equals(old))
                    {
                        CreateAndSet(ocrEngineTypes, cultureInfoName);
                    }
                }
                catch (ArgumentOutOfRangeException)
                {
                }
            }

            GC.Collect();
        });
    }
}