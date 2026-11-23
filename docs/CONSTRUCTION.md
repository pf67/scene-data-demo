# 从照片构建场景

浏览器、HTTP 和 Python 共用 `builder/plan.schema.json` 与同一几何求解器。输出包含带嵌入纹理的 GLB、输入方案和质量报告。每栋建筑为单独命名的封闭网格，支持平顶、四角双坡顶和背面补全，可继续在工作台或 Blender 中编辑。

补充照片未覆盖的区域时，先完成屋檐轮廓和相机标定，再从工作台打开 **补充视角**。真实多图填写在 `views`，模型旋转生成图填写在 `supplements`。[旋转机位流程](ROTATED_VIEWS.md) · [视角 API 与 MCP 工具](ORBIT_API.md)

## 选用哪种输入

| 手头的资料 | 操作 | 可以检查什么 |
| --- | --- | --- |
| 一张照片，没有实测点 | 导入新场景，核对估算相机离地高度、俯角和垂直视场角，描屋檐，输入大致檐高 | 构图和体块关系。尺度标为估算，误差栏留空 |
| 一张照片和地面尺寸 | 选择地面控制点，填写四个以上分布在同一平面上的已知 X/Z 坐标，再描屋檐。标出第一个檐角正下方的墙脚，或输入已知檐高 | 相机拟合像素误差、留出检查点的地面距离误差、墙脚求得的檐高 |
| 两至六张有重叠的照片 | 分别标定到同一 X/Z 坐标系，在各图选择同一栋楼，从相同实体角出发、按同序描出同数目的檐角 | 同名屋角交会、各图重投影偏差及最小交角 |

新图默认垂直视场角为 45°。快速估算的初始相机高度为 15 m、俯角为 25°，应按拍摄视角调整。已知焦距时，垂直视场角为 `2 × atan(成像面有效高度 / (2 × 焦距))`。经过裁切或数字变焦的图片需使用裁切后的有效视场。镜头畸变明显时先去畸变，再在同一成图上标注。

## 浏览器操作

1. 打开工作台，点 **导入新场景** 选择 JPG 或 PNG。可一次选择多张，也可后续点 **增加视图**。每张至多 16 MiB 和 3200 万像素。服务端按 EXIF 调整方向并将长边限制到 3072 px。
2. 点 **地面控制点** 调整相机。实测标定可用矩形辅助预填四个地面坐标，也可逐点输入任意测得的 X/Z。各图可见的控制点可以不同，只要地面原点、方向、单位一致。另选实测点并勾选 **留作独立检查点**，它会专用于误差评估。
3. 点 **屋檐轮廓**，添加建筑。沿檐口画 3–16 个凸多边形角点。凹形建筑可以拆成体块。多图在建筑下拉框选择同一编号，再标相同檐角。滚轮或加减按钮缩放，Shift 拖动平移。选中点后，方向键移动一个原图显示像素，Shift 加方向键移动十个像素。
4. 单图可用 **墙脚测高** 标记第一个檐角的垂足。多图存在可靠交会时优先求高。双坡屋顶另填屋脊高出檐口的距离，屋脊沿长边方向。遮挡的角点若无法可靠对应，可只给这栋建筑保留有依据的观测并输入檐高。
5. **校验几何** 会指出对应的视图或建筑编号。核对误差后点 **构建 3D 场景**。质量面板提供 GLB 和报告下载，**打开构建结果** 会以参考相机视角进入编辑器。场景库支持重新打开 3D 或继续编辑保存的方案。浏览器暂存当前草稿，**保存方案 JSON** 可单独备份。

## Agent 的最短调用路径

服务运行后，在仓库根目录执行以下命令可完整构建内置村庄。该客户端只用 Python 标准库，输出目录中会有 `plan.json`、`scene.glb`、`report.json` 及 `ground/` 地面图层，终端会打印可直接打开的场景链接。

```sh
python tools/scene_pipeline/build_client.py --plan builder/example.json --output artifacts/village
```

使用生成的双视图样房可核对自己的部署。真实设定为檐高 5.40 m、占地 8 × 6 m。样例包含两张照片、逐视图标定、同名屋角和留出的地面检查点。

```sh
python tools/scene_pipeline/calibration_example.py artifacts/pair
python tools/scene_pipeline/build_client.py --plan artifacts/pair/plan.json --photo artifacts/pair/view-1.png --photo artifacts/pair/view-2.png --output artifacts/pair-result
```

界面上也可以依次 **导入新场景** 选择这两张图片，**导入方案** 选择同目录的 `plan.json`，然后构建。导入方案里的图片编号按原文件字节的 SHA-256 匹配，跨服务器迁移时可重新上传相同原图，或直接迁移 `data/builder/`。改过编码的照片会获得新编号。

### 给一张新照片生成草案

将下面的初始方案保存为 `input.json`。将相机值替换为实际或估算值，`front` 是本次客户端使用的图片别名。已有标注可直接写入 `buildings`，最小建筑形如 `{"id":"house_1","heightM":6,"heightSource":"estimated","observations":[{"source":"front","roof":[[0.42,0.40],[0.56,0.42],[0.55,0.48],[0.40,0.46]]}]}`。示例轮廓需要换成照片中的实际点位。

```json
{
  "version": 1,
  "name": "沿街建筑",
  "scaleBasis": "estimated",
  "views": [{
    "source": "front",
    "fovY": 45,
    "approxPose": { "heightM": 15, "pitchDeg": 25 },
    "controls": []
  }],
  "buildings": []
}
```

配置模型后可以先生成轮廓草案，在界面核对并调整。省略 `--draft-only` 会继续运行几何校验和构建。已有可靠建筑轮廓时省略 `--prompt` 即可本地完成。服务部署在另一台机器时传 `--server` 指定工作台地址。

```sh
python tools/scene_pipeline/build_client.py --plan input.json --source front street.jpg --prompt "描出沿街每栋房屋的檐口，分开相邻建筑" --draft-only --output artifacts/draft
python tools/scene_pipeline/build_client.py --plan artifacts/draft/plan.json --output artifacts/street
```

### HTTP 工具约定

| 方法与路径 | 请求 | 返回 |
| --- | --- | --- |
| `GET /api/builder/tools` | 无 | `data.functions` 为函数描述，`data.endpoints` 为名称到 HTTP 路径的映射，`data.planner` 表示是否配置视觉模型 |
| `GET /api/builder/schema`、`GET /api/builder/example` | 无 | 原始 JSON Schema 或可构建的村庄方案 |
| `POST /api/builder/sources` | 图片原始二进制 | `data.source,url,width,height`。重复上传相同原文件复用编号 |
| `POST /api/builder/propose` | `{"plan":{...},"instruction":"..."}` | `data.plan` 含候选建筑数组，保留已有标定，给模型推测的输入檐高标记 estimated |
| `POST /api/builder/validate` | 方案对象本身 | `data.views,buildings,notes`。不创建场景 |
| `POST /api/builder/build` | 同一个方案对象 | HTTP 201，`data.scene` 与 `data.report`。场景记录提供 GLB、方案、报告 URL |
| `GET /api/scenes/3d` | 无 | 已注册场景，构建结果的 kind 为 `calibrated_building_scene` |

Agent 先注册图片，把返回的编号填入 `views[].source` 和 `observations[].source`。工具调用名为 `validate_scene_plan` 和 `build_scene`，把模型给出的函数参数作为完整 JSON 请求体发送到对应路径。服务端执行固定的几何工具。结果中的 `scene.code` 可拼到 `/?scene=<code>` 直接预览。校验失败会返回 HTTP 422 和 `detail`，请依据具体错误修正方案。单次构建完成后才发布到场景库，重复构建生成独立版本。

### 视觉模型服务

在本机 `.env` 中填写 `SCENE_PLANNER_URL`（完整的 Chat Completions 路径）和 `SCENE_PLANNER_MODEL`，需要 Bearer 鉴权时填写 `SCENE_PLANNER_KEY`。重启服务或重新创建 Compose 容器。浏览器只访问本工作台。点击 **模型草案 → 生成轮廓草案** 才会将当前参考图和建模要求发送到配置的模型，点 **应用建筑草案** 后替换当前轮廓。

模型端需接受 `messages[].content` 内的 `text` 和 `image_url` 数据 URL，支持 `response_format.type=json_object`，并在 `choices[0].message.content` 返回 JSON 字符串，其中有一个符合 Schema 的 `buildings` 数组。这条路径用于建筑轮廓草拟。Qwen-Image-2.1 的画面重绘仍使用独立的 `COMFYUI_URL` 工作流。

## 坐标和质量报告

- 图像坐标是归一化 `[u,v]`，以校正方向后的图片左上为原点，分别除以宽和高。地面标记为 `[x,z]`，Y 向上，测量单位为米。已知非方形像素焦距时提供 `focalRatio=fx/fy`，裁切产生的主点偏移可用 `principal=[cx/width,cy/height]` 指定。
- 求解器用参与拟合的地面点拟合相机姿态。报告分别记录 `fitRmsPx`、`checkRmsPx`、`checkRmsM`。缺少独立检查点时后两项为 `null`。单图快速估算连拟合误差也为 `null`。现场验收应使用实测尺度和独立点，并由项目方确定允许误差。
- 双视图或多视图先对同名角点交会，再按共同檐高拟合同一平面。报告包括各建筑 `heightMethod`、`heightM`、`areaM2`、`roofRmsPx`、`minRayAngleDeg`。交角小于 2°、对应点顺序错误或残差过大会提示重标。单图墙脚测高依赖建筑外墙垂直和墙脚落在参考平面。
- 可见面优先从正对且清晰的照片取色。受到遮挡或跨出画面的部分逐块采样，未观察到的立面使用推断材质。报告中的 `photoFaces`、`partialFaces`、`inferredFaces` 为面数，补面数包含建筑底面。地面覆盖比例单独统计。参考平面以外可指定推断起伏或道路引导线，复杂地形可继续在 Blender 精修。[地面补全参数与操作](GROUND.md)

## 独立运行和复现

在没有 HTTP 服务的机器上，也可直接运行同一建模工具，显式传入每个图像编号对应的文件。先创建输出目录。

```sh
python -m pip install -r builder/requirements.txt
python builder/worker.py validate --plan builder/example.json --source village assets/scene_sources/village_road_near.png
python builder/worker.py build --plan builder/example.json --source village assets/scene_sources/village_road_near.png --out scene.glb --report report.json
```

村庄的 64 栋轮廓源自已有标注。运行 `python tools/village_scene/make_builder_example.py` 可从仓库深度缓存和布局重建示例方案。对有效平屋顶，脚本腐蚀轮廓排除边缘，留出五分之一像素后取稳健中值，并以原有楼高的 65%–135% 约束远处的单目深度漂移。双坡顶和深度样本不足的屋顶沿用原先的高度。精度数据及其适用范围见 [验证记录](VALIDATION.md)。

场景库附带已构建的 **村庄沿街建筑 · 标定示例**，可直接访问 `/?scene=VILLAGE_CALIBRATED`。更新示例方案后，可用下面的命令重新生成仓库中的 GLB 和报告。

```sh
python builder/worker.py build --plan builder/example.json --source village assets/scene_sources/village_road_near.png --out assets/scene_meshes/VILLAGE_CALIBRATED.glb --report builder/example-report.json --ground-dir builder/example-ground
```
