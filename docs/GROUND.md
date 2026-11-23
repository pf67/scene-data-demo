# 地面覆盖、纹理与补图回填

从照片构建时，工作台先把同一个地面位置投影到各张参考图，合成一张正射纹理。照片没有覆盖的位置可以离线合成，也可以在构建时融合已配准的旋转机位。输出可从任意视角浏览，渲染时使用同一份地面。样例可通过服务首页的 **地面补全** 打开，直接入口为 `/?scene=VILLAGE_CALIBRATED&panel=ground`。

## 在界面里操作

1. **选照片和地面范围。** 在照片工作台导入一到六张参考图。单图可填写相机高度和俯角，多图需在同一地面坐标系标定。只做空地时可保留空的建筑列表。有建筑时勾画屋檐并提供高度，可帮助求解遮挡。
2. **排除地上的物体。** 在 **地面补全 → 当前参考图的地面选区** 导入与当前照片方向和宽高比相同的黑白图。白色表示允许从这张照片采样，黑色用于排除树冠、车顶、水面或其他不适合投到地上的区域。可从分割模型、深度数据或手工标注生成。未提供选区时仍使用已建建筑遮挡和投影质量筛选。
3. **离线生成。** 在 **地面补全** 填写每侧外扩比例，选“延续邻近地面”或“田块与乡间道路”。起伏设为 0 可得到平地。大于 0 会让远离观测区、建筑基底和引导道路的地面产生平滑起伏。点 **离线补全并构建**。地面照片很少时，“延续邻近地面”主要依据已有地表颜色和程序纹理。
4. **检查覆盖。** 切换 **地面纹理 / 覆盖分区**。绿色为照片采样，黄褐色为几何遮挡，蓝色在所有相机画面以外，紫灰色表示射线在画面内但地面选区或采样质量不足。四项比例按地面纹理像素统计。切换到 3D 编辑器可从 **俯视** 和 **背面视角** 检查道路方向、场地边界和房屋基底。
5. **补充街区的侧面与地面。** 打开 **补充视角建模**，用原始照片生成新机位。勾选配准成功的视图重新构建，生成图会按遮挡关系采样到原照片未覆盖的建筑和地面。[操作与接口](ROTATED_VIEWS.md)
6. **导入已有正射图。** 点击 **下载纹理** 和 **下载补全掩膜**。白色允许补图，黑色保护实拍覆盖像素。保持朝向、裁切和宽高比，导出 PNG 或 JPEG 并点击 **导入地面补图**。单图最多 16 MiB、3200 万像素。

## 给后续 Agent 的输入与输出

在原有方案 v1 的顶层加入可选的 `ground`。此示例把已有范围向四边各扩展 35%，指定材质和一条宽 5.5 米的道路。请用场景中实际标注的地面 X/Z 坐标替换示例道路。

```json
{
  "ground": {
    "marginPercent": 35,
    "surface": "fields",
    "reliefM": 0.6,
    "roads": [
      { "points": [[-20, 4], [4, 7], [28, 9]], "widthM": 5.5, "extendEnds": true }
    ]
  }
}
```

范围依据建筑足迹和地面控制点计算。纯地面单图在没有控制点时，用向下射线与参考平面的交点估算范围。外扩可设为 0–120%，默认 25%。`surface` 默认 `local`。推断起伏幅度 `reliefM` 默认 0，可设至 6 个场景米。估算尺度的米数随相机先验变化。道路是有宽度的折线，每条可有 2–24 个点，最多 8 条。默认沿首末段继续延伸，设 `extendEnds:false` 可保留端点。引导道路仅绘入缺失区域。

每张照片的选区通过与照片相同的上传接口注册，把返回的编号写入对应的 `views[].groundMask`。标准库客户端支持将选区别名和照片别名一起通过 `--source 别名 文件` 传入。它会同时下载 `ground/texture.png`、`ground/mask.png` 和 `ground/coverage.png`。

| 接口 | 输入与结果 |
| --- | --- |
| `GET /api/builder/tools` | `data.groundRedraw` 指示是否配置模型服务。`data.ground` 描述图层、回填、重绘和轮询约定 |
| `POST /api/builder/build` | 发送完整场景方案。响应 `data.scene.ground_url` 给出本版本地面资料路径，`data.report.mesh.ground` 含方法、尺寸、覆盖、每个视图的采样数、地形范围和三角形数 |
| `GET {ground_url}/texture`、`/mask`、`/coverage` | 分别下载纹理、编辑掩膜和覆盖分区 PNG。左上为报告中最小 X、最小 Z，向右增加 X、向下增加 Z |
| `POST {ground_url}/apply` | 请求体为补图原始字节。成功返回 HTTP 201，读取 `data.scene`、`data.report` |
| `POST {ground_url}/redraw`（旧客户端兼容） | JSON 请求体 `{"prompt":"延续道路与土壤纹理，保持正上方视角"}`。返回 HTTP 202 和 `data.taskId` |
| `GET /api/builder/ground-jobs/{taskId}` | `data.status` 为 `running`、`completed` 或 `failed`。完成时取得新场景和报告，失败时读取 `message`。服务重启后仍可查询已发布结果 |

旧客户端的正射补图接口使用随包提供的 Qwen-Image-2.1 配置。提交前按比例把正射纹理放到 1024 × 1024 画布，以边缘颜色补足空白，返回后裁回同一位置。此步骤使用整张参考纹理作为上下文。最终写入范围始终由服务端掩膜决定。回填复用原 GLB 的网格、位置、UV、法线、相机与建筑材质。报告以 `method=local/imported/qwen-image-2.1` 标记本版本，并用 `parentScene` 连接回填前的版本。Qwen 服务和权重由部署环境提供。[依赖清单](DEPENDENCIES.md)

## 命令行与离线迁移

在仓库根目录执行以下示例。第一步导出可编辑地图。第二步使用已重绘的对齐俯视图生成新模型，请给新版本使用单独的目标路径。

```sh
mkdir -p work
python builder/worker.py build --plan builder/example.json --source village assets/scene_sources/village_road_near.png --out work/scene.glb --report work/report.json --ground-dir work/ground
python builder/worker.py ground-edit --model work/scene.glb --ground-dir work/ground --image edited-ground.png --out work/edited.glb --out-ground work/edited-ground
```

离线合成、旋转图配准、构建与正射图回填都可以在 CPU 上运行。地面选区与补图任务随 `data/builder/` 整体备份。[Linux 与 Docker 部署](DEPLOYMENT.md)

## 精度边界

单图画外的田块、道路走向和地面起伏属于场景推断。纹理更真实时，几何精度仍取决于照片标定、实测控制点和深度资料。要把推断区域变成照片覆盖区，可增加拍到该区域的真实照片并在同一坐标系重新构建。生成的俯视补图用于更新表面外观。测绘或真实坡高复原需要独立的地形观测。
