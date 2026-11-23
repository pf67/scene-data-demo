# 从原始参考图生成旋转机位并补全场景

## 在浏览器中操作

打开面板后，页面从 `/api/builder/orbit-providers` 读取已配置的图像服务、模型、角度与出图选项。部署端可指定默认服务，用户也可逐次切换。使用内网网关、官方 API 或本地 QMA 的配置方式见 [API 与 MCP 接入](ORBIT_API.md)。

1. 在照片工作台载入原图，标出屋檐和地面控制点，或打开已标定的村庄。多张原始照片分别在同一坐标系标定。只有一张照片时可用估算相机生成草模。
2. 打开 **补充视角**。在 3D 编辑器中，入口是顶部的 **补充视角建模**。村庄示例可直接打开 `/?scene=VILLAGE_CALIBRATED&panel=orbit`。
3. 选原始参考图、服务、模型和拍摄方向。在图像编辑 API 下可直接输入有符号角度。村庄 90° 示例采用 **原图和目标机位结构图**、高质量。也可在 **出图设置** 添加材质要求。QMA 的 **随角度** 模式采用下表的建议，滑块或数值输入可覆盖，点 **恢复随角度** 可重置。点 **生成补充视角** 提交，或用 **导入已有旋转图** 处理已有图片。返回面板可继续读取最近一次任务。
4. 切换 **原图**、生成图、结构图、轮廓叠加，沿主路找到同一栋房子，依次对照近景与远景屋角。面板显示实际转角和匹配点数。生成图可能只变化了色彩、改变了道路朝向或新增一排房屋，此时应重试或放弃该候选。
5. 可以辨认对应位置而自动配准失败时，点 **标记同名点**，先点目标结构图上的屋角，再点生成图中的同一屋角。至少六对参与拟合，总数最多 64 对，分布在街道两边及远近不同的房屋。勾选 **下一对作为独立检查点** 可留出对应点验证拟合。标点后重新配准复用同一候选图。村庄示例的 **查看或修改同名点** 可打开随包标注。
6. 勾选要采用的补充机位，合计最多 16 个，点击重建按钮生成独立版本。在 3D 编辑器点 **从此机位查看 3D** 对照建筑外观，再点 **自动环绕** 检查两机位之间的墙面。整圈使用同一份全局模型。出现错位的候选可取消勾选后重新构建。
7. 查看 **最大未采样方位跨度**。缺口较大时，先从靠近已核实机位、仍能认出同一排屋角的方向取得下一张实拍图，或用 `gpt-image-2` 配合目标结构图生成该方向候选。逐张比对后再扩展到更远方向。示例采用了新增的左转 110° 请求候选，接近背面的 165° 请求候选未通过整图对应检查，未加入构建方案。

| QMA 拍摄方向 | 基于服务默认值的建议 | 服务默认值为 0.40 时 |
| --- | --- | --- |
| 左前方、右前方（45° 档） | 默认值 | 0.40 |
| 左侧、右侧（90° 档） | 默认值 + 0.25 | 0.65 |
| 左后方、右后方（135° 档） | 默认值 + 0.40 | 0.80 |
| 正后方（180° 档） | 默认值 + 0.50 | 0.90 |

建议强度最高为 1。方向档位描述期望的机位，实际转角由图像配准测得。面板会显示连接失败的原因。更新场景程序后需同时重启对应端口的后端。模型服务恢复后点 **重新检测**。在生成服务离线时仍可导入旋转图、查看候选或重建已有机位。

## 图像与几何如何关联

每个机位始终从选定的原始参考照片出发。基础相机绕已有建筑群中心旋转，保留焦距与原相机姿态关系。程序在目标相机下绘制屋顶、墙面和道路，依据深度和可见性投影原照片，得到目标结构图。村庄样例还会使用随包的单目深度缓存补充背景。一般场景只需原照片、相机和建筑方案。

使用图像编辑 API 时，默认将目标结构图作为第一张输入、原图作为第二张输入。模型按结构图安排屋顶、街道与遮挡关系，从原图取得材质。建模端从候选图反求实际机位，利用原照片、目标结构图及中间角度的可见建筑寻找同一屋角。采用时仅利用有对应点支持的邻域，并避开残差过大的标点附近。新机位中的窗户与未拍摄墙面属于外观推断。

QMA 模式按相对拍摄方向请求图像，通过真实照片上可定位的屋角估计生成图的焦距与机位。`generation` 记录实际使用的方向、强度、步数、CFG、分辨率和种子。方向档位与水平实测角可能有明显差异。报告给出 `actualAngle`（有符号水平角）、`orbitDeltaDeg`（含高低变化的视线夹角）和 `targetMismatch`。偏离请求方向的候选须在界面主动勾选。夹角在 5–8° 之间时，额外要求至少 40 个分散内点、画面覆盖不少于 12%、足够低的重投影误差，以及至少两个以前未见的建筑面和 256 个受支持像素。

独立 ComfyUI 双图模式中，`TextEncodeQwenImage21` 的第一张图是目标机位结构图，第二张图是原始参考照片。大转角使用已观测到的屋角固定目标构图，模型补充旋转后新露出的墙面与地面。新立面的材料先验可来自同栋或邻近建筑的可见墙面。生成后还会把候选与同机位结构图比较，过滤空白、重复纹理和缺少对应区域支持的像素。

对自动匹配，至少需要 12 个互异内点；人工配准至少 6 个。内点覆盖的凸包面积须达到画幅的 3.5%，系统还检查相机高度、旋转方向、实际角度和观察距离。报告里的 `rmsPx` 是当前图像分辨率下的重投影误差；它描述机位对齐程度。单张照片未拍到的窗户、墙面、院落仍属于外观推断。

原始照片先参与建筑轮廓、尺度和檐高求解，生成图单列在 `supplements`。构建时按当前方案重新配准。对于有深度和标注的区域，构建器将屋檐、墙脚和屋脊放入同一个世界坐标系，生成闭合建筑。院落及地表采用通过深度和朝向检查的观测合成连续纹理。缺少观测的背面从相邻建筑和植被推断材质。GLB 默认场景可从各方向浏览。[全景建模过程](SURFACE_DETAIL.md)

## 后续 Agent 接口

| 调用 | 返回或用法 |
| --- | --- |
| `GET /api/builder/tools` | 返回函数声明、调用入口和全部 HTTP 路径；完整流程见 [Agent 调用示例](ORBIT_API.md) |
| `GET /api/builder/orbit-providers` | 返回可选服务、模型、连续角度或方向档位，以及该服务支持的参数 |
| `GET /api/builder/orbit-options` | 查看 `data.ready` 和 `provider`。QMA 已就绪时返回 `defaultStrength` 和 `directions[]`，每项有 `angle,label,azimuth,suggestedStrength`。不可用时返回 `ready:false` 和原因，可再次查询 |
| `POST /api/builder/sources` | 上传 PNG/JPEG 原始字节，返回 `data.source`。同一接口也可注册外部生成的旋转图 |
| `POST /api/builder/orbit-views` | 用 `sceneCode` 或 `plan` 指定场景，提交 `angle,provider,model` 及服务支持的选项。QMA 可传 `strength,seed`，图像编辑 API 可传 `guidance,quality,instruction`。返回 202 和 `data.taskId` |
| `GET /api/builder/orbit-jobs/{taskId}` | 返回 `running/completed/failed`。完成后包含 `spec,quality,imageUrl,guideUrl,overlayUrl`。新生成的候选另带 `generation` 记录 |
| `POST /api/builder/orbit-build` | 用 `taskIds` 与 `keepSources` 选择新旧机位，返回 201 及新场景、方案与报告 |

`completed` 表示处理结束。只有 `quality.status == "aligned"` 的候选可用于建模。`needs-points` 时先核对叠加图。重新配准时提交相同的 `plan,reference,angle`，额外传 `image: spec.source` 与 `points: [{"guide":[u,v],"image":[u,v]}, ...]`。两幅图都从左上角开始用 0–1 坐标。导入外部旋转图时直接传 `image`，可在无 GPU 的机器上配准。

`needs-detail` 表示候选缺少可用的新表面细节。QMA 成功生成的任务会记录 `generation.strength` 和 `generation.azimuth`。角度输入范围是左右 10° 至 180°，QMA 会映射到同侧最近的方向档，正负 180° 均为后视。通过相机 MCP 直接调用时，可向场景端查询对应方向的 `suggestedStrength` 后显式传给 `qma_generate`。用 `qma_job` 查询、下载并上传生成图，再向场景端提交 `image` 和 `provider:"qma"` 即可配准。

下面的 Python 示例使用部署端基址 `SCENE_URL`。它从示例方案生成一个右侧新机位，核对状态后发布新版。模型调用由服务端发起。

```python
import os
import time
import requests

base = os.environ["SCENE_URL"].rstrip("/")
def get(path):
    response = requests.get(base + path, timeout=30)
    response.raise_for_status()
    return response.json()

def post(path, payload):
    response = requests.post(base + path, json=payload, timeout=150)
    response.raise_for_status()
    return response.json()["data"]

plan = get("/api/builder/example")
job = post("/api/builder/orbit-views", {
    "plan": plan, "reference": plan["views"][0]["source"], "angle": 90,
    "provider": "openai", "model": "gpt-image-2", "guidance": "layout", "quality": "high"
})
while True:
    state = get("/api/builder/orbit-jobs/" + job["taskId"])["data"]
    if state["status"] != "running":
        break
    time.sleep(2)
if state["status"] == "failed":
    raise RuntimeError(state["message"])
if state["quality"]["status"] != "aligned":
    raise ValueError(state["quality"]["message"])
if state["quality"].get("targetMismatch"):
    raise ValueError("请对照生成图和轮廓叠加，确认实测机位后再采用")
previous = plan.get("supplements", [])
keep = [s["source"] for s in previous if (s["reference"], s["angle"]) != (state["reference"], state["angle"])]
result = post("/api/builder/orbit-build", {"plan": plan, "taskIds": [job["taskId"]], "keepSources": keep})
print(result["scene"]["url"], result["report"]["mesh"]["supplements"])
```

## 离线复现与迁移

源码包内的 `builder/sample-views/` 含已经生成的机位 JPEG、结构图、叠加图和采样记录，`builder/example.json` 记录选用的机位。从仓库根目录执行以下命令即可在 CPU 上重建样例。

```sh
mkdir -p output
python builder/worker.py build --plan builder/example.json \
  --source village assets/scene_sources/village_road_near.png \
  --out output/village.glb --report output/report.json --ground-dir output/ground
```

对于自己的图，把原始图片、地面选区及已生成的每张机位用各自的 `--source ID FILE` 传入。可用 `python builder/worker.py orbit-view --help` 查看生成与人工标点参数。报告的 `supplements[]` 记录真实参考图 ID、要求角、拟合角、相机与误差；`mesh.supplements.byView` 记录每张图贡献的建筑纹理像素，`buildingFaces` 按面去重，`mesh.ground.syntheticTexels` 统计补充地面像素。地面覆盖分区仍按原始照片观测统计。

浏览已有模型和使用包内七张已配准补充图重建可以完全离线。生成新机位需要安装 `refinement/requirements.txt`，并从场景服务连通一种图像生成 API。迁移时整体复制 `SCENE_DATA_DIR/builder/`，其中 `sources/` 保存原图，`orbit-jobs/` 保存候选与配准材料。使用自定义深度和表面标注时，也需迁移这两个目录。Linux 容器使用其网络命名空间可达的服务地址。[Linux/Docker 部署](DEPLOYMENT.md) · [依赖清单](DEPENDENCIES.md)
