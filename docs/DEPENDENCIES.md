# 依赖清单

| 功能 | 所需依赖 | 安装位置 |
| --- | --- | --- |
| 编辑已有场景、保存、导出 | Node.js 22+、WebGL 浏览器 | 主机或容器 |
| 前端构建 | Vue 3、Three.js 0.160、Vite 6、TypeScript、vue-tsc | `frontend/package-lock.json` 锁定 |
| 照片构建、校验、GLB | Python 3.10+、NumPy、OpenCV、Pillow、jsonschema，使用 CPU | `builder/requirements.txt`，或镜像内 Debian 系统包 |
| 用新视频估计机位、核对屋角、准备视频纹理 | 上述 CPU 依赖和 pycolmap 4.2.1，已在 Windows、Python 3.13 验证 | `tools/scene_pipeline/requirements-video.txt`。打开示例只需浏览器和 Node.js，使用包内视频成果重新建模只需基础构建依赖。[处理流程](VIDEO_RECONSTRUCTION.md) |
| 地面融合、离线补全、补图回填 | 复用照片构建的 CPU 依赖。示例使用包内的地面选区 | `builder/` 包含示例地图与选区文件 |
| 视觉模型草案 | 支持图片消息、JSON 对象输出的 Chat Completions 服务 | 配置 `SCENE_PLANNER_URL`、`SCENE_PLANNER_MODEL`，鉴权需要时配置 `SCENE_PLANNER_KEY` |
| 远程构建客户端 | Python 3.10+ 标准库 | `tools/scene_pipeline/build_client.py`，几何依赖安装在服务端 |
| 村庄建模准备/检查 | Python 3.10+、NumPy、Pillow、OpenCV | `tools/requirements.txt` |
| 几何补全和 GLB 导出 | Blender 4.x；使用 Blender 内置 Python/bpy | 独立安装 |
| 新图深度/相机证据 | Depth Anything 3、本地 DA3 权重、与 GPU 匹配的 PyTorch | 建议单独虚拟环境；本仓库不分发权重 |
| 全局表面融合、闭合建筑和逐面贴图 | 复用 CPU 构建依赖和 WebGL。包内提供原图、七张已配准的补充图、深度缓存、局部素材和对应标注 | Windows 复建示例可直接使用缓存。为新图计算缓存时可在独立 DA3 环境运行 `tools/scene_pipeline/cache_surface_depth.py`。[数据格式与命令](SURFACE_DETAIL.md) |
| 图像编辑 API 生成机位 | Python、Pillow、requests、可达的 Images Edits 服务 | `refinement/requirements.txt`，支持官方接口或兼容网关 |
| Qwen 生成机位 | Python、Pillow、requests、QMA 或 ComfyUI 与 Qwen-Image-2.1 | `refinement/requirements.txt` 与部署端模型服务 |
| MCP 工具适配器 | Python、MCP SDK 1.x、httpx | 客户端安装 `tools/scene_pipeline/requirements-mcp.txt`。HTTP 调用可直接使用场景服务 |

基础 Node 服务仅依赖标准库。机位生成客户端与 DA3 不需要放在同一个环境中。模型推理通常需要独立 GPU；本地浏览器编辑不需要服务端推理 GPU。

已有 QMA HTTP 服务时，场景侧使用 Pillow 和 requests 调用 `QMA_API_URL`。Qwen-Image-2.1、QMA LoRA 和 ComfyUI 安装在模型服务侧。QMA 默认 30 步、CFG 3、分辨率档位 1024、强度 0.4。已有 MCP 客户端可生成并下载同一协议的机位图片，再由工作台配准。发布包只带接入代码与图片，不分发外部 QMA 服务或模型权重。

Qwen-Image-2.1 工作流需要 ComfyUI 支持 `TextEncodeQwenImage21` 与 `QwenImage21Cache`，以及下列权重：

```text
diffusion_models/qwen_image_2.1_int8_convrot.safetensors
text_encoders/qwen3vl_8b_int8_convrot.safetensors
vae/qwen_image_2.1_vae_bf16.safetensors
```

机位生成复用 `refinement/workflow.json`，提交时增加第二个 `LoadImage`。`TextEncodeQwenImage21` 必须同时支持 `images.image_1` 和 `images.image_2`。交互默认 36 步、分辨率参数 896、CFG 1；示例保留每张图实际使用的分辨率、步数和种子。配准与离线重建使用 OpenCV 的 SIFT、PnP 和 NumPy。[模型双图输入和验收](ROTATED_VIEWS.md)

模型权重、DA3 和 ComfyUI 的安装版本、GPU 驱动由部署团队根据已有环境锁定；本仓库不声称已验证全部组合。离线部署需提前准备相应软件、安装包及权重。
