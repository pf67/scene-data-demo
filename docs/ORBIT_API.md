# 视角补齐 API 与 Agent 工具

照片工作台和 3D 编辑器共用场景服务的 HTTP 接口。Agent 可以用普通 HTTP、函数调用或 MCP 驱动同一流程。打开 `/?scene=VILLAGE_CALIBRATED&panel=orbit` 可逐步对照结果。

## 接入模型

| 场景服务配置 | 作用 |
| --- | --- |
| `OPENAI_API_KEY` | 官方 Images Edits 接口的密钥，保存在服务端 |
| `ORBIT_IMAGE_BASE_URL` | 兼容网关基址，例如 `https://images.example/v1`。程序在基址后追加 `/images/edits` |
| `ORBIT_IMAGE_API_KEY` | 兼容网关自己的凭据。允许免鉴权的网关可留空 |
| `ORBIT_IMAGE_MODELS` | 逗号分隔的实际模型 ID。第一项是该服务的默认模型，未配置时使用 `gpt-image-2` |
| `ORBIT_IMAGE_SIZE` | 可选。默认按原图比例换算尺寸，村庄为 1248 × 832。如网关限定尺寸，填写它支持的值 |
| `ORBIT_IMAGE_STREAM` | 官方接口默认接收 SSE 并等待最终图像。兼容网关默认读取 JSON 的 `data[0].b64_json`，支持 SSE 时可设为 `true` |
| `QMA_API_URL` / `COMFYUI_URL` | 可选的其他机位生成服务，可与图像编辑 API 同时启用 |
| `ORBIT_DEFAULT_PROVIDER` | 设为已配置的 `openai`、`qma` 或 `comfyui`。单次请求的 `provider` 优先 |

将配置放在部署环境或本地 `.env` 中，重启场景服务。Compose 部署用 `docker compose up -d` 重新载入配置。打开 `GET /api/builder/orbit-providers` 读取服务、模型、默认值和支持的字段。图像编辑服务的 `ready` 表示配置已经加载，账户权限和额度在实际出图时验证。API 按请求选择一个服务执行，错误可从任务状态读取。模型只接收所选参考照片、可选的目标结构图及提示词。[官方图像编辑协议](https://developers.openai.com/api/reference/resources/images/methods/edit)

## 从照片到新场景

1. 选择已标定的场景，或通过 `POST /api/builder/sources` 上传自己的原始照片，构建含相机与建筑轮廓的方案。多张原图在同一坐标系中标定。后续请求传 `sceneCode` 或传完整 `plan`；用 `reference` 选择其中一张真实照片，省略时取第一张。[单图与多图标定流程](CONSTRUCTION.md)
2. 查询提供方，选择它声明的模型、方向及可选参数。图像编辑 API 使用有符号的连续水平角，向左为负、向右为正，绝对值范围 10–180°。默认 `guidance:"layout"` 同时发送目标结构图和原始照片；`"photo"` 只发原图，可用于对比。QMA 按返回的方向选择，可覆盖 `strength`。
3. 提交生成任务并轮询。任务完成后检查 `quality.status`、内点分布、`actualAngle` 和轮廓叠加。`needs-points` 时用可辨认的屋檐做配准；`needs-detail` 时换候选。单次出图的请求角度描述目标机位。
4. 选择通过配准的任务构建新 GLB。`taskIds` 与 `keepSources` 合计最多 16 个补充机位。用 `scene.url` 下载模型，用 `scene.plan_url` 与 `scene.report_url` 读取方案与报告。`report.mesh.surfaceDetail.globalSurface` 记录融合面数与各机位贡献，`report.orbitCoverage.largestGapDeg` 给出相邻已配准机位之间最大的水平角空档，包含经过 ±180° 的扇区。
5. 若需要继续补齐，读取 `report.orbitCoverage.suggestedAnglesDeg`，选缺口内一个目标方向发起下一次请求。该数组以第一张原图的机位为 0°。示例可选用 `provider:"openai", model:"gpt-image-2", guidance:"layout"`。逐张检查独立点误差、不同屋角的贴合情况和遮挡次序，再把通过的结果加入下一次构建。建议方向仅描述相机分布，表面的准确度还取决于覆盖同一屋面的实拍观测。[全景建模与验收](SURFACE_DETAIL.md)

## HTTP 与函数调用

调用 `GET /api/builder/tools` 获取 `data.functions`（函数声明）及 `data.call`（执行路径）。向执行路径发送 JSON `{"name":"工具名","arguments":{...}}`。成功结果在 `data`，请求被拒时检查 HTTP 状态及 `detail`。

| 函数 | 直接 HTTP | 用途 |
| --- | --- | --- |
| `list_orbit_providers` | `GET /api/builder/orbit-providers` | 获取当前服务、模型与选项 |
| `create_orbit_view` | `POST /api/builder/orbit-views` | 在线生成或离线配准，返回 202 和 `taskId` |
| `get_orbit_job` | `GET /api/builder/orbit-jobs/{taskId}` | 查询运行状态、生成参数、图像和配准结果 |
| `build_with_orbit_views` | `POST /api/builder/orbit-build` | 选择同一场景几何下通过配准的任务，返回 201 和独立版本 |

下面是四次调用的请求体。用实际返回值替换 `TASK_ID`。示例保留其余四个补充机位，把已有的右转 90° 机位替换为新候选。

```json
{"name":"list_orbit_providers","arguments":{}}
```

```json
{"name":"create_orbit_view","arguments":{"sceneCode":"VILLAGE_CALIBRATED","reference":"village","provider":"openai","model":"gpt-image-2","angle":90,"guidance":"layout","quality":"high","instruction":"保留街道两侧的建筑密度、屋檐和院落位置。"}}
```

```json
{"name":"get_orbit_job","arguments":{"taskId":"TASK_ID"}}
```

```json
{"name":"build_with_orbit_views","arguments":{"sceneCode":"VILLAGE_CALIBRATED","taskIds":["TASK_ID"],"keepSources":["village-orbit-qma","village-orbit-left","village-orbit-right","village-orbit-image-left110"]}}
```

轮询直到 `status` 为 `completed` 或 `failed`。完成后使用 `imageUrl`、`guideUrl` 和 `overlayUrl` 下载图像。`generation` 记录实际发送的模型参数。`quality.status=="aligned"` 才能采用。若 `targetMismatch` 为真，先对照图像和测得角度，认可后显式提交 `acceptMeasuredPose:true`。建筑轮廓、原始相机和地面参数变化后需重新配准。省略 `keepSources` 会保留方案中的全部旧机位，传空数组只使用所选新任务。

## 离线配准与重现

上传已有 JPEG/PNG 的原始字节到 `POST /api/builder/sources`，取返回的 `data.source`。调用 `create_orbit_view`，传 `sceneCode,reference,angle,image` 和 `registration:"measured"`。需要重标时直接传上一任务的 `spec.source`，复用保存的像素。标注格式为 `{"guide":[u,v],"image":[u,v]}`，两幅图均以左上角为原点、坐标归一化到 0–1。在留出点上加 `"check":true`，它们只参与误差验证。须有至少六个拟合点，含留出点最多 64 对。报告给出检查点总数、阈值内点数、中位误差和最大误差，单位是候选图像素。

在示例中取 `GET /api/builder/example` 的 `supplements`，找到 `source` 为 `village-orbit-image-right90` 的项。用该项的 `source` 作 `image`，原样传入其 `angle,reference,points`，即可离线走通上述查询和重建。已有 GLB、原图、结构图、生成图、26 对标注和质量报告均随包提供。[村庄验证记录](VALIDATION.md)

## MCP 客户端

在 Agent 所在机器安装 `python -m pip install -r tools/scene_pipeline/requirements-mcp.txt`。将以下配置中的脚本路径及场景服务地址换成实际值。适配器使用标准输入输出连接 MCP 客户端，启动时从对应场景服务发现工具。

```json
{
  "mcpServers": {
    "scene-builder": {
      "command": "python",
      "args": ["/opt/scene-data-demo/tools/scene_pipeline/orbit_mcp.py", "--server", "http://127.0.0.1:8090"]
    }
  }
}
```

也可通过 `SCENE_URL` 提供地址。连接后依次调用上表四个工具。MCP 客户端只需连通场景服务，图像 API 密钥留在部署端。
