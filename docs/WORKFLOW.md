# 从参考图到可用场景

日常换图、标注、多视图交会和批量构建请从 [构建工具与模型接入](CONSTRUCTION.md) 开始。下文记录原村庄的 DA3 与 Blender 离线制作流程，可用于继续精修地形、道路、植被及不可见结构。

## 1 输入和目录

准备参考照片、装备 GLB，以及可用的尺度/相机信息。保留原图。示例图位于 `assets/scene_sources/village_road_near.png`。新增装备置于 `assets/equipment_3d/CODE.glb`，场景置于 `assets/scene_meshes/CODE.glb`；CODE 使用英文字母、数字、下划线或短横线。

同名 JSON 设置 `label`、`kind` 或 `category`。装备还可设置 `dimensions`、`origin`、`up_axis`、`yaw_offset`、`default_scale`，格式参考现有文件。更新模型后刷新页面。

## 2 示例复现

示例已经附带原图、深度/相机缓存和屋顶轮廓，不必重新运行 DA3。

```sh
python -m pip install -r tools/requirements.txt
python tools/village_scene/prepare.py
blender --background --python tools/village_scene/build_blender.py
python tools/village_scene/verify.py
```

`prepare.py` 拟合地面并把屋顶轮廓投影到三维，生成布局与纹理。Blender 脚本创建封闭墙面、屋顶、道路、植被等，原图投影到可见面、背面使用补全材质。结果写入 `assets/scene_meshes/VILLAGE_COMPLETE.glb` 和 `assets/village_studio/village_complete.blend`。

附带缓存只适用于附带照片。`roof_layout.json` 中的像素轮廓、房高和 `prepare.py` 的道路/地面区域都与该图绑定，不能原样套用新图。

## 3 新单图

在独立 DA3 Python 环境中，使用预先准备的模型目录：

```sh
python tools/scene_pipeline/run_da3.py --model-dir models/da3 --images assets/scene_sources/new.png --output data/da3/new
```

原始输出是深度、相机与重建证据。若要接入此例的建模脚本，用 `select_view.py` 将一个视图转换为二维 `depth.npy` 与 3×3 `K.npy`：

```sh
python tools/scene_pipeline/select_view.py --input data/da3/new --output data/da3/new-view --index 0
```

为新图单独建立布局参数和脚本输入目录；标定地面、道路、屋顶轮廓与高度，检查相机与坐标轴，随后在 Blender 中补全缺失墙面/屋顶。保留输入照片和变更参数以便复现。需要真实米制尺寸时，加入控制尺寸重新校准。

## 4 多图

向 `run_da3.py --images` 传入多张重叠照片或目录。检查拍摄重叠、视差和相机估计后，使用选择的重建工具完成视图配准、深度融合、网格重建和纹理烘焙，再进入 Blender 修补。仓库提供证据提取与最终导出，不包含一键自动多视图稠密重建服务。

## 5 导出接口

在 Blender 中检查八个方向、建筑闭合性、法线、纹理、地面承载和比例。设置相机后导出：

```sh
blender --background reviewed.blend --python tools/scene_pipeline/finalize_blender.py -- --output assets/scene_meshes/NEW_SCENE.glb --label "新场景"
python tools/scene_pipeline/inspect_glb.py assets/scene_meshes/NEW_SCENE.glb
```

场景使用 Y-up、内嵌纹理的 GLB，`extras` 提供 `preserve_scale`、`camera_position`、`camera_target`、`camera_fov`。通用导出器会生成同名 JSON 和拓扑报告。

## 6 编辑与形成数据

启动服务，在右侧选择场景，拖入装备并调整位置、旋转和缩放；检查地面接触、遮挡、相机视角和光照。保存场景会话后重新打开，确认资产和参数恢复。

选择当前或多角度画面并导出。客户端渲染 PNG、计算三维包围盒的二维投影，服务端生成 VOC XML 与 ZIP。人工检查遮挡框和边缘截断；投影框并非可见掩膜标注。不要把截图文件数直接等同于有效样本数。

## 7 根据参考图补充机位

在粗几何上确定要补充的机位，保留可见屋角和道路透视。将该机位结构图与原始参考照片一起输入 Qwen-Image-2.1。检查生成图与目标机位的轮廓叠加，将配准通过的机位选入方案并重新构建。原始照片参与相机标定和建筑尺度求解，补充视图参与缺失表面的投影与局部细节建模。生成成果随新版 GLB 和报告保存。[旋转视角操作与接口](ROTATED_VIEWS.md)

## 后续 Agent 交接

先确定输入是现成 GLB、单图还是多图，再选择上述路径。每个新场景必须有素材清单、尺度说明、建模参数、GLB/元数据、至少正反侧面的预览，以及编辑/保存/导出的验收记录。新模型和环境配置本地提供，不硬编码绝对路径、服务器地址或密钥。
