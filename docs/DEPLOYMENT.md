# 部署与接口

## 独立运行

安装并构建前端后执行 `npm start`。默认绑定 `127.0.0.1:8090`，局域网访问可在 `.env` 设置 `HOST=0.0.0.0`。此 Demo 没有多用户鉴权，适合受控演示环境。

运行目录包含 `server.mjs`、`package.json`、`web/`、`assets/` 和 `builder/`。构建照片场景时用 `SCENE_PYTHON` 指定已安装 `builder/requirements.txt` 的解释器。生成新机位另需 `refinement/`、其 Python 依赖以及可访问的图像编辑 API、QMA 或 ComfyUI。`SCENE_DATA_DIR` 默认是 `data/`，包含会话、图片和可重新编辑的构建结果，迁移时整体备份。

## Linux Docker 与离线搬运

交付源码、示例资产及预构建前端时，先运行 `npm --prefix frontend ci`，再执行 `python tools/package_release.py`。生成的 `artifacts/scene-data-demo-版本-提交号.zip` 带有 `RELEASE.json`。打包器从该次 Git 提交导出源码并在隔离目录编译前端，最后输出 SHA-256。目标机解压后在包根目录执行 `npm start`，服务会读取同目录下的 `.env`。照片构建需要安装 `builder/requirements.txt`，也可使用包含这些运行依赖的容器镜像。

在联网构建机上：

```sh
docker build -t scene-data-demo:1.6.1 .
docker save -o scene-data-demo-image.tar scene-data-demo:1.6.1
```

拷贝镜像到目标机器后：

```sh
docker load -i scene-data-demo-image.tar
docker run -d --name scene-data-demo -p 8090:8090 -v scene-demo-data:/data scene-data-demo:1.6.1
```

访问 `http://<服务器地址>:8090`。若 Qwen 服务在另一台机器，运行容器时添加 `-e COMFYUI_URL=http://<模型服务地址>:8188`。容器中的 localhost 指向容器本身。环境地址只写本地配置，不提交 Git。

已有反向代理可把本服务整体代理到相同站点；前端 API 使用相对路径。无需原聊天后端。模型更新需重建镜像，或自行挂载经过验证的 `assets/`。

部署后打开村庄的补充视角页面，选右转 89.7° 的机位并点 **从此机位查看 3D**。用 **自动环绕** 绕主街一圈，再从 **背面视角** 和 **俯视** 核对同一栋建筑的道路位置与屋顶轮廓。用包内原图和七张补充图在 CPU 上重新构建，下载返回的 GLB，核对默认场景及报告中的 `mesh.surfaceDetail.globalSurface` 和 `orbitCoverage`。不同 OpenCV 版本的采样计数可能有小幅差异。[部署验证记录](VALIDATION.md) · [深度与表面标注目录](SURFACE_DETAIL.md)

## API

照片工作台的上传、估算草案、几何校验、构建和报告读取见 [建模 API](CONSTRUCTION.md)。使用反向代理时，单张照片请求上限需允许 16 MiB，构建请求的代理读取超时建议至少 150 秒。Compose 将可选的模型配置传入服务端。

已有 QMA 服务时，在部署环境设置 `QMA_API_URL` 为其 HTTP 基址，例如同机直接运行时的 `http://127.0.0.1:8190`。若使用 Docker，需要容器可达的地址。QMA 仅监听宿主机回环时，Linux 可给场景容器使用宿主机网络并省略端口映射，或在受控内网提供可达的模型服务。已有独立 ComfyUI 工作流时也可配置 `COMFYUI_URL`。场景端 `POST /api/builder/orbit-views` 返回任务编号，轮询后把通过配准的视图写入方案再构建。部署团队可先用包内机位验证 CPU 重建，再以真实模型生成新图。[旋转视角接口与验收](ROTATED_VIEWS.md) · [地面接口](GROUND.md)

接口通常返回 `{status:0,message:"ok",data:...}`；错误返回 HTTP 状态码和 `detail`。

| 路径 | 用途 |
| --- | --- |
| `GET /api/health` | 核对当前端口实际运行的版本及重绘配置 |
| `GET /api/builder/orbit-providers` | 页面与 Agent 获取同一份图像服务、模型及可用参数 |
| `GET /api/builder/tools`、`POST /api/builder/tools/call` | 发现并调用构建工具。[图像 API 与 MCP 配置](ORBIT_API.md) |
| `GET /api/builder/orbit-options` | 检查补充视角是否可生成。QMA 的返回包括可用方向和建议强度。接口返回 404 时需更新并重启当前端口的场景服务 |
| `GET /api/assets/3d` | 装备列表 |
| `GET /api/scenes/3d` | 场景列表 |
| `GET /api/{assets,scenes}/3d/{code}/file` | GLB 文件 |
| `GET/POST /api/scene/sessions` | 列出/创建会话 |
| `GET/PUT/DELETE /api/scene/sessions/{id}` | 读取/保存/删除会话 |
| `POST /api/generate` | `type=scene_export` 导出；`type=scene_refine` 重绘 |
| `POST /api/get_task_result` | 以 `task_id` 查询重绘结果 |
| `GET /api/scene/exports` | 导出列表 |
| `GET /api/scene/exports/{id}/download` | 下载 ZIP |

导出参数 `params.frames` 包含 1–24 帧，每帧有 `image_base64`、`width`、`height` 和 `objects`。每个对象的字段为 `name,xmin,ymin,xmax,ymax`。图片必须为 PNG，单边不超过 4096；请求体上限 160 MB。

旧客户端的 `scene_refine` 兼容接口参数包含 `base64_image, environment_image, protection_mask, prompt`，可选 `steps,cfg_scale`。`workflow_template` 仅允许 `scene_refine_qwen21`。结果状态为 `running/completed/failed`，成功时返回 `result[0].image` Base64 PNG。服务重启后的未完成任务会显示失败并要求重新提交。

## 资产来源与范围

示例资产来自本次用户提供的装备 GLB、村庄照片及其派生建模结果；仓库没有为这些资产新增第三方授权条款。依赖组件保留各自许可证。此仓库不包含旧仓库 Git 历史、部署配置、账号、模型权重或运行数据库。
