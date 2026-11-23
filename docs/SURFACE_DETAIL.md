# 从参考照片构建可连续环绕的村庄

在 1.6 版打开右转机位并沿主街拖动，近处的屋顶和远处的田地会产生视差。继续环绕到背面或切换俯视，编辑器始终渲染同一个 GLB 默认场景。该模型融合了原图、已配准的补充图、按建筑校正的立面以及连续地表。光照面板的 **保留原图光影** 适合对照照片，**全场景重光照** 可以检查建筑轮廓。[打开右转示例](http://127.0.0.1:8090/?scene=VILLAGE_CALIBRATED&panel=orbit&view=village-orbit-image-right90)

1.5 版给每个参考相机单独生成深度表面，近距离能够复现原图的像素。单张图里前景屋檐遮住的墙面没有相应像素，移动到另一侧便会露出空缺。因此当时的编辑器在右转机位约 12 米之外换回闭合建筑。1.6 将房屋和地形固定在同一世界坐标系。视角之间的外观变化发生在这套模型的表面。[1.5 版对照图](images/right90-model-comparison.webp)

在 **保留原图光影** 下，编辑器将当前位置可见的墙面、屋顶和地面投回已标定的相机。每个像素先对照该机位拍摄方向上最近表面的深度。通过遮挡检查后，才从相应照片取色。在原机位，参考照片可提供瓦片、屋檐和街边纹理。离开机位时，若有校正过的图像深度，编辑器还会估算照片中的物体与网格表面在新视角会错开多少像素。预计错位越大，照片在该处的权重越低，以减轻屋顶被拉到地面的现象。另一个有效机位可以补上新露出的表面，其余位置沿用模型本身的材质。完整重新打光使用模型的基础材质。修改过的地面保留编辑后的贴图。

GLB 默认场景的 `extras.photo_projectors` 记录可投影机位。每台相机包含照片和可见深度的内嵌纹理索引、图像尺寸、内参、光心、从世界坐标到相机坐标的旋转，以及深度换算比例。深度纹理对应最终场景的遮挡关系。编辑器最多同时采样两个相近机位；截图和图像导出使用同一套材质。通用 GLB 查看器可直接显示固定网格和基础纹理。复现编辑器的照片外观需要读取这些相机数据。

## 在 Windows 复核三个后方机位

1.6.1 版以同一张村庄原图和每个目标机位各自的结构图调用 `gpt-image-2`。保留生成服务返回的原始画幅，沿可辨认屋顶描出四个檐口，在远近不同的房屋上留出独立检查点。旋转请求角、实际拟合角、检查点偏差和能够使用的像素区域分别记录。以下三张图可以直接对照随包素材。

| 生成图 | 怎样进入 3D 场景 |
| --- | --- |
| [右转 130°](../builder/sample-views/village-orbit-image-right130.jpg) | 已配准。只在匹配到的区域投影。逐栋核对后墙和屋顶，并把可识别的局部面绑定到所属建筑。[屋角标点](../builder/sample-views/village-orbit-image-right130-points.json) |
| [左转 150°](../builder/sample-views/village-orbit-image-left150.jpg) | 已配准。逐栋描出后墙，另外在 60、61 号房附近用两个墙脚校正菜圃和小径的位置，三个留出的墙脚检验局部误差。[局部材质标注](../builder/sample-views/village-orbit-image-left150-materials.json) |
| [右转 170°](../builder/sample-views/village-orbit-image-right170.jpg) | 独立检查表明整村机位不可靠。按建筑编号与朝向取用局部屋顶和立面，另将 52 号房前一小块院地通过局部检查后贴入地面。该来源仅用于这些表面，方位覆盖统计沿用合格相机。[局部材质标注](../builder/sample-views/village-orbit-image-right170-materials.json) |

在编辑器先从右转 90° 起始机位看沿街正面，再拖到后排，观察同一栋房屋的门窗是否随相机移动而跑到院子里。切到全场景重光照，复核后墙和院地仍附着于同一实体。两张新增的合格机位把匹配区域以透明度写进可见深度纹理，边缘逐渐淡出。原始机位和原有右转 90°、左转 110° 的标定继续使用。示例包含一个原图机位和七个已配准补充机位，旧的 165° 素材继续按墙面使用。[Windows 实际环绕与检查](VALIDATION.md)

## 先检查误差来自哪里

1. 检查整张图的机位，再检查单栋建筑。示例右转请求为 90°，用对应点求得 89.7°。14 个内点覆盖图像面积的 18.28%，7 个独立检查点的偏差中位数为 10.47 px。一些近景屋檐对齐后，远景房屋仍会错位。生成相机会改变房屋的局部尺度、遮挡和相对位置。
2. 在照片上核查地面取样。旧版用已建建筑做遮挡测试，未建的树冠、附房和部分外墙会漏过测试，沿地面拉成长条。新版结合深度估算高度、表面朝向、建筑遮挡和地面选区。道路另外用原图路中线和路面样色校正。
3. 分别检查有观测和无观测的建筑面。可辨认的屋顶与立面按屋檐和墙脚分别校正贴图。未见的背面优先使用同一场景的墙体和屋面材料，报告将这些面计入推断。

## 给新场景准备证据

1. **有条件实拍时**，选固定街区中心，在不同方位和高度拍摄重叠的图像。相邻照片至少要能认出多栋相同建筑的屋角、墙脚和路口。对优先还原的房屋，先用两张有明显侧向位移的真实照片求解同名点的空间位置，再用第三张独立实拍机位检查重投影。记录实测地面点或建筑尺度，有助于发现整排房屋同时偏移的问题。原始照片在工作台共用同一个坐标系。[多视图标定方法](CONSTRUCTION.md)
2. **只有单图时**，先标原图可见的屋檐、墙脚和道路。可选用图像编辑接口的 `gpt-image-2`，配合目标结构图向两侧逐段生成补充视角。沿已有对应屋角向方位缺口推进，保存请求角和实际输出。用街道两侧、远近不同建筑的同名点配准，留出整栋建筑的屋角检查相机。个别院落要在地面上另找对应的墙脚，有余下的点检查位置时再局部贴图。未通过整图校验的候选可以保留已逐栋识别的素材。生成的背面仍属于推断。[旋转视角接口](ORBIT_API.md)
3. **准备深度缓存**。下载服务实际保存的图片。原图使用 `/api/builder/sources/<source>/image`，生成图使用任务返回的 `imageUrl`。上传端可能校正方向和转码，标注与缓存按保存文件的 SHA-256 关联。在已安装 DA3、PyTorch 和本地权重的环境运行下方命令，得到每张图的 `<sha256>.npz`。包内已有村庄深度缓存，复建示例仅需 CPU。以下 PowerShell 示例中的模型目录按本机实际位置填写。

```powershell
python tools/scene_pipeline/cache_surface_depth.py --model-dir ./models/DA3-LARGE-1.1 --images original.jpg right.jpg --output ./data/surface-depth --resolution 1008 --device cuda
```

4. 需要精修单栋或补院落时，为对应照片准备 `<sha256>.json`。参照包内两份完整样例。坐标采用标注图的像素，原点在左上角，宽高填写在 `size` 中。照片的标注可独立使用。

| 标注字段 | 填写方式与作用 |
| --- | --- |
| `buildings` | 使用方案中的建筑 ID，沿屋檐依次给四个角点 `roof`，同序填写四个墙脚 `feet`。看不见的墙脚填 `null`。可选 `hipRiseM` 为根据照片估算的四坡屋脊升高 |
| `infill` | 附房名称、四角、墙脚和估算檐高 `height`。构建器用可见墙脚校正高度，检查与既有建筑和道路的冲突 |
| `walls`、`roofs` | 截取完整材质的四边形 `quad`，填写其近似宽高（米）。依次填写左上、右上、右下、左下 |
| `ground`、`streets` | 地面可信多边形；原图可用道路下标 `road`、中心线 `center`、沥青与路肩取样点 `asphalt` 和 `shoulder` 校正已有道路 |
| `terrainMaterials` | 在已确认的地面上用四角 `quad` 取样，填写材料类别 `kind` 和米制大小 `sizeM`，供缺少观测的院落、土路和田地选材 |
| `boundaries`、`trees` | 依次标记院墙墙脚折线；树木标记树冠框 `box` 和树根 `foot`。输出可环绕的几何 |
| `eaves` | 建筑照片显示屋檐凸出时设为 `true`，整村模型将生成实体檐口 |
| `projectionRegion`、`projectionExclusions` | 新生成图填写 `matched` 后，按同名点的局部支持范围构建投影遮罩。`projectionExclusions` 以标注图像素列出需排除的局部墙面多边形 |
| `materialReferences` | 引用 `builder/sample-views/` 中源图及其 `-materials.json`。标注中的 `sha256` 绑定原始 JPEG。屋顶 `quad` 按建模方案四个屋角的原顺序，墙面 `edge` 对应原轮廓的边号。院地用 `groundQuad` 对应源图 `quad`，局部检验通过才填写 `verified: true`。写入时裁掉建筑占地、锁定的原图地面，并在选区内渐变接缝 |

5. 把深度缓存和标注放进服务可读的 `SCENE_SURFACE_DEPTH_DIR` 和 `SCENE_SURFACE_MARKS_DIR`。Compose 默认读取数据卷下的 `/data/surface-depth`、`/data/surface-marks`，也会查找随包示例。未准备标注时，构建器会从可见图像自动寻找立面材料；未提供深度时仍可生成闭合建筑。
6. 调用原有 `POST /api/builder/build`，或选择已配准视图调用 `POST /api/builder/orbit-build`。在输出场景的 `report_url` 查看以下字段。要复现随包场景，可直接运行下方命令。

```powershell
New-Item -ItemType Directory -Force output | Out-Null
python builder/worker.py build --plan builder/example.json --source village assets/scene_sources/village_road_near.png --out output/village.glb --report output/report.json --ground-dir output/ground
```

## 如何读取结果

| 报告字段 | 含义 |
| --- | --- |
| `mesh.surfaceDetail.reviewedFacesByView` | 各机位中，通过逐栋屋檐或墙脚标注校正的基础建筑面数 |
| `mesh.surfaceDetail.context` | 每张图实际补建的附房、树木和经过裁剪的院墙段数 |
| `mesh.surfaceDetail.depthAnchorMedianPercent` | 相对深度对几何锚点的拟合残差中位数（百分比），用于筛选机位 |
| `orbitCoverage.anglesDeg`、`largestGapDeg`、`suggestedAnglesDeg` | 以第一个原始相机为基准，记录各机位实际水平角、最大未采样方位跨度及建议补拍方向。包括原图和已配准的补充图 |
| `mesh.surfaceDetail.globalSurface` | 启用全局融合时出现。`triangles` 为构建器统计的融合三角面数，`byView` 分机位统计有图像支持的面，`inferredTriangles` 记录推断面数。随模型保留的基础网格也可参与显示 |
| `mesh.surfaceDetail.photoProjectors.sources`、`depthSizes` | 记录进入运行时投影的机位及可见深度图尺寸。新版使用原图、左转 110°、右转 90° 以及按匹配区域取色的左转 150° 和右转 130°。局部素材另按屋顶、墙面和院地区域使用 |
| `mesh.supplements.buildingFaces`、`byView` | 前者是补充图参与贴图的基础建筑面数。后者仅统计进入局部纹理图集的采样像素，直接校正的面另见 `reviewedFacesByView` |

## 如何验收一圈

固定相同的浏览器视口、轨道中心、相机距离和光照，保存起始机位及环绕一整圈的分方位画面。首先对照起始照片核对街道与屋顶。转到中间方位，连续跟踪同一栋建筑的屋脊、墙脚和前后遮挡关系。最后回到起点，再缩放和平移查看近景。对有独立实拍图的方向，比较留出屋角的像素偏差，并用实测控制点检查尺度。

方位空档缩小代表相机分布更均匀。屋檐是否落在正确位置、屋后有无附房、背面窗户如何排列，需要相应的图像观测分别支持。单张原图配合生成视角可以制作连贯的演示场景，实测精度需要额外的真实视角和尺寸证据。[版本验收记录](VALIDATION.md)
