using System;
using System.Collections.Generic;
using BetterGenshinImpact.GameTask;
using BetterGenshinImpact.Model;
using Microsoft.ML.OnnxRuntime;
using System.ComponentModel;
using System.IO;
using System.Linq;
using BetterGenshinImpact.Core.Config;
using Microsoft.Extensions.Logging;
using Microsoft.Win32;
using SharpCompress;
using Vanara;

namespace BetterGenshinImpact.Core.Recognition.ONNX;

public class BgiSessionOption : Singleton<BgiSessionOption>
{
    private static readonly ILogger<BgiSessionOption> Logger = App.GetLogger<BgiSessionOption>();

    private static volatile bool _isCudaPathSet = false;

    public readonly FeatureType[] FeatureTypes;

    private readonly int cudaDeviceId;
    private readonly int dmlDeviceId;
    private readonly bool optimizedModel;

    public BgiSessionOption()
    {
        cudaDeviceId = TaskContext.Instance().Config.HardwareAccelerationConfig.CudaDeviceId;
        dmlDeviceId = TaskContext.Instance().Config.HardwareAccelerationConfig.DmlDeviceId;

        FeatureTypes = Initialization(TaskContext.Instance().Config.HardwareAccelerationConfig.InferenceDevice,
            cudaDeviceId, dmlDeviceId).ToArray();
        optimizedModel = TaskContext.Instance().Config.HardwareAccelerationConfig.OptimizedModel;
    }

    private static void SetCudaPath()
    {
        // 获取所有可能包含CUDA/cuDNN路径的环境变量
        var pathVariables = new HashSet<string>();
        Environment.GetEnvironmentVariable("PATH", EnvironmentVariableTarget.Process)?.Split(Path.PathSeparator)
            .ForEach(s => pathVariables.Add(s));
        Environment.GetEnvironmentVariable("CUDA_PATH", EnvironmentVariableTarget.Process)?.Split(Path.PathSeparator)
            .ForEach(s => pathVariables.Add(s));
        Environment.GetEnvironmentVariable("CUDNN_PATH", EnvironmentVariableTarget.Process)?.Split(Path.PathSeparator)
            .ForEach(s => pathVariables.Add(s));
        Environment.GetEnvironmentVariable("LD_LIBRARY_PATH", EnvironmentVariableTarget.Process)?.Split(Path.PathSeparator)
            .ForEach(s => pathVariables.Add(s));
        // 用于存储有效的DLL路径
        var validPaths = new List<string>();
        var cudaVersion =
            Registry.GetValue(@"HKEY_LOCAL_MACHINE\SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA",
                "FirstVersionInstalled", null)?.ToString() ?? "v12.8";
        string[] filePrefix = ["cudnn", "nvrtc", "cudart", "nvinfer", "cublas", "onnx"];

        var basePaths = pathVariables.ToArray().SelectMany<string, string>(s =>
        {
            List<string> r = [s, Path.Combine(s, cudaVersion), Path.Combine(s, "bin"), Path.Combine(s, "lib")];
            return cudaVersion.StartsWith("v", StringComparison.InvariantCultureIgnoreCase)
                ? [..r, Path.Combine(s, cudaVersion[1..])]
                : r;
        });
        foreach (var basePath in basePaths)
        {
            if (string.IsNullOrWhiteSpace(basePath)) continue;
            // 检查基础路径是否存在
            if (!Directory.Exists(basePath)) continue;
            foreach (var se in filePrefix)
            {
                Directory.GetFiles(basePath, $"{se}*.dll").Select(Path.GetDirectoryName).WhereNotNull()
                    .ForEach(s => validPaths.Add(s));
            }
        }

        foreach (var validPath in validPaths)
        {
            pathVariables.Add(validPath);
        }

        // 更新环境变量
        var updatedPath = string.Join(";", pathVariables);
        Logger.LogDebug("[GpuAuto]修改PATH为:{}", updatedPath);
        Environment.SetEnvironmentVariable("PATH", updatedPath, EnvironmentVariableTarget.Process);
    }

    private static List<FeatureType> Initialization(DeviceType deviceType, int cudaDeviceId, int dmlDeviceId)
    {
        switch (deviceType)
        {
            case DeviceType.Cpu:
                return [FeatureType.Cpu];
            case DeviceType.GpuAuto:
                if (!_isCudaPathSet)
                {
                    _isCudaPathSet = true;
                    SetCudaPath();
                }

                List<FeatureType> list = [];
                SessionOptions? testSession = null;
                try
                {
                    testSession = SessionOptions.MakeSessionOptionWithTensorrtProvider(cudaDeviceId);
                    list.Add(FeatureType.TensorRt);
                }
                catch (Exception e)
                {
                    Logger.LogDebug("[init]无法加载TensorRt。可能不支持，跳过。({Err})", e.Message);
                }
                finally
                {
                    testSession?.Dispose();
                }

                if (!list.Contains(FeatureType.TensorRt))
                {
                    try
                    {
                        testSession = SessionOptions.MakeSessionOptionWithCudaProvider(cudaDeviceId);
                        list.Add(FeatureType.Cuda);
                    }
                    catch (Exception e)
                    {
                        Logger.LogDebug("[init]无法加载Cuda。可能不支持，跳过。({Err})", e.Message);
                    }
                    finally
                    {
                        testSession?.Dispose();
                    }
                }

                try
                {
                    testSession = new SessionOptions();
                    testSession.AppendExecutionProvider_DML(dmlDeviceId);
                    list.Add(FeatureType.Dml);
                }
                catch (Exception e)
                {
                    Logger.LogDebug("[init]无法加载DML。可能不支持，跳过。({Err})", e.Message);
                }
                finally
                {
                    testSession?.Dispose();
                }

                list.Add(FeatureType.Cpu);
                return list;
            default:
                throw new InvalidEnumArgumentException("无效的推理设备");
        }
    }

    public SessionOptions MakeSessionOption(string relativePath)
    {
        return MakeSessionOptionWith(FeatureTypes, relativePath);
    }

    private SessionOptions MakeSessionOptionWith(FeatureType[] features, string relativePath)
    {
        var cachePath = Global.Absolute(GetModalCachePath(relativePath, out _));
        if (!Directory.Exists(cachePath))
        {
            Directory.CreateDirectory(cachePath);
        }


        var sessionOptions = new SessionOptions();
        foreach (var type in features)
        {
            try
            {
                switch (type)
                {
                    case FeatureType.Dml:
                        sessionOptions.AppendExecutionProvider_DML(dmlDeviceId);
                        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                        break;
                    case FeatureType.Cpu:
                        sessionOptions.AppendExecutionProvider_CPU();
                        break;
                    case FeatureType.TensorRt:
                        using (var options = new OrtTensorRTProviderOptions())
                        {
                            options.UpdateOptions(GetTrtConfig(relativePath));
                            sessionOptions.AppendExecutionProvider_Tensorrt(options);
                        }

                        break;
                    case FeatureType.Cuda:
                        using (var options = new OrtCUDAProviderOptions())
                        {
                            options.UpdateOptions(GetCudaConfig(relativePath));
                            sessionOptions.AppendExecutionProvider_CUDA();
                        }

                        break;
                    default:
                        throw new InvalidEnumArgumentException("无效的推理设备");
                }
            }
            catch (Exception e)
            {
                Logger.LogWarning("无法加载{Engine}。可能不支持，跳过。({Err})", Enum.GetName(type), e.Message);
            }
        }

        if (!optimizedModel) return sessionOptions;
        var optPath = Path.Combine(cachePath, "optimized");
        if (!Directory.Exists(optPath))
        {
            Directory.CreateDirectory(optPath);
        }

        sessionOptions.OptimizedModelFilePath = optPath;
        return sessionOptions;
    }


    private Dictionary<string, string> GetTrtConfig(string relativePath)
    {
        var cachePath = GetModalCachePath(relativePath, out var fileName);
        if (relativePath.StartsWith(@"Cache\modal") && fileName.EndsWith("_ctx.onnx"))
        {
            // 已经优化过
            var r = new Dictionary<string, string>
            {
                ["device_id"] = cudaDeviceId.ToString(),
            };
            return r;
        }

        var result = new Dictionary<string, string>
        {
            ["trt_engine_cache_enable"] = "1",
            ["trt_dump_ep_context_model"] = "1",
            ["trt_ep_context_file_path"] = Global.Absolute(Path.Combine(cachePath, "trt")),
            ["trt_ep_context_embed_mode"] = "1", // 因为yoloSharp是把模型转为嵌入式运行，不这样会爆炸
            // ["trt_engine_cache_path"] = ".\\" // 没必要了
            ["trt_timing_cache_enable"] = "1",
            ["trt_timing_cache_path"] = Global.Absolute(@"User/trt_timing"),
            ["trt_force_timing_cache"] = "1",
            ["device_id"] = cudaDeviceId.ToString(),
        };
        if (!Directory.Exists(result["trt_ep_context_file_path"]))
        {
            Directory.CreateDirectory(result["trt_ep_context_file_path"]);
        }

        return result;
    }

    private Dictionary<string, string> GetCudaConfig(string relativePath)
    {
        var result = new Dictionary<string, string>
        {
            ["device_id"] = cudaDeviceId.ToString(),
        };
        return result;
    }

    /// <summary>
    /// 返回相对路径
    /// </summary>
    /// <param name="relativePath"></param>
    /// <param name="fileName"></param>
    /// <returns></returns>
    private static string GetModalCachePath(string relativePath, out string fileName)
    {
        var absolute = Global.Absolute(relativePath);
        fileName = Path.GetFileName(absolute);
        var index = relativePath.LastIndexOf(fileName, StringComparison.Ordinal);
        if (relativePath.StartsWith(@"Cache\modal"))
        {
            return relativePath[..index];
        }

        var fileName2 = fileName.Replace('.', '_');
        return index > -1
            ? Path.Combine(@"Cache\modal", relativePath[..index] + fileName2)
            : Path.Combine(@"Cache\modal", relativePath.Replace('.', '_'));
    }

    /// <summary>
    /// 返回相对路径
    /// </summary>
    /// <param name="relativePath">相对路径</param>
    /// <returns></returns>
    public string GetCachedModelPath(string relativePath)
    {
        var cachePath = GetModalCachePath(relativePath, out var fileName);
        if (relativePath.StartsWith(@"Cache\modal") && fileName.EndsWith("_ctx.onnx"))
        {
            return relativePath;
        }

        if (!FeatureTypes.Contains(FeatureType.TensorRt)) return Global.Absolute(relativePath);
        var ctxA = Path.Combine(cachePath, "trt\\_ctx.onnx");

        if (File.Exists(Global.Absolute(ctxA)))
        {
            return ctxA;
        }

        var absolute = Global.Absolute(relativePath);
        var ctxB = Path.Combine(cachePath, "trt\\" + Path.GetFileNameWithoutExtension(absolute) + "_ctx.onnx");
        return File.Exists(Global.Absolute(ctxB)) ? ctxB : relativePath;
    }
}