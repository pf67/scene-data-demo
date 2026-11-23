# 在 Windows 上从两侧逐步补建背面

从已经校准的两张照片向空档中央转动。每步都保存父图、父图摘要、请求角度、实测角度和局部复核结果。下一步使用刚验收的照片。所有三维点始终在最初村庄的世界坐标系中。

## 生成一个相邻视角

示例依赖见 `builder/requirements.txt`。从仓库根目录执行以下 PowerShell 命令。Node 读取本机已有的图像服务配置并传递给 Python。计算引导图和之后的复核都在本机完成。

```powershell
$stage = "artifacts/new-right-step"
$parent = "village-orbit-image-right130"
$photo = "builder/sample-views/$parent.jpg"
$common = @("--plan","builder/example.json","--report","builder/example-report.json","--source","village","assets/scene_sources/village_road_near.png")
node --env-file-if-exists=.env -e "const p=require('node:child_process').spawnSync('python',process.argv.slice(1),{stdio:'inherit',windowsHide:true});process.exit(p.status??1)" -- -X utf8 tools/scene_pipeline/step_view.py @common --parent $parent --parent-image $photo --target 145 --directory $stage --model gpt-image-2
```

目标角相对于最初的相机。例如从实测 +129.7° 请求 +145°，相对步长为 +15.3°。`guide.jpg` 使用父照片及其深度投影到目标机位，`registration-guide.jpg` 用原始世界坐标生成，适合对照控制点。`generation.json` 记录两张图的来源和摘要。`--guide-only` 可先查看引导图。当一次返回几乎复制父机位时，在新目录加 `--detail-swatches` 重新生成。这会保留完整的目标构图，将第二张参考图改为父图的局部细节。

## 校准并复核可用于建模的区域

1. 将前一版已验收的 GLB 单独保存，随后都使用这份几何锚定两条链。为新图运行已有的 `cache_surface_depth.py`，提供本地 DA3 权重目录、图像列表和输出目录。用 `SCENE_SURFACE_DEPTH_DIR` 指向缓存。发布包已包含示例的逐图深度，浏览和重建示例均可离线进行。
2. 下面第一条命令匹配原照片之间的局部特征，并将对应点绑定到已建模型上的物理位置。独立检查点按楼栋或地块留出。查看 `track-quality.json` 的 `status`、`actualAngle` 和检查点误差。实测角度没有向目标前进的候选需要另拍。表面归属不明确的点留给人工检查。
3. 第二条命令交叉检查两幅图的深度、遮挡、法向以及同一细节的往返位移，输出 `paired-mask.png`。若要复核院地，传入 `--ground-polygons`，文件为候选原始像素坐标的多边形数组。工具会另存 `yard-mask.png`，只允许可在两图中定位到地面的部分进入院地材质。

```powershell
$geometry = "assets/scene_meshes/VILLAGE_CALIBRATED.glb"
python -X utf8 tools/scene_pipeline/register_step.py @common --parent-image $photo --directory $stage --geometry $geometry
python -X utf8 tools/scene_pipeline/verify_surface_pair.py @common --parent-image $photo --directory $stage --geometry $geometry
```

4. 两端接近时，运行 `check_step_overlap.py --directories 第一端目录 第二端目录 --geometry 锚定模型 --output overlap.json`。它向两个方向检查共同可见的区域，并为每张图保存剔除冲突后的 `joined-mask.png`。两个相近的相机位置仍然可能生成不同的房屋或菜畦。
5. 有明确对应的整片屋顶可用 `transfer_roofs.py` 处理。它需要父图的具名屋面资料以及新图同一屋顶的四个檐口，输入格式可对照随包的 `*-materials.json`。工具独立核对屋面内部的纹理或留出的檐口，通过的屋面才写入 `materials.json`。模型优先使用同一屋面的完整材质，支持区域横切屋顶时回到经校正的整片材质或原网格。
6. 把通过的 `track-quality.json` 加入工作报告的 `supplements`，补上唯一的 `source` 和最初的 `reference`。下一步把该图作为 `--parent-image`，把该标识作为 `--parent`，配准时加上 `--parent-region` 指向它的验收掩膜。对不确定的部位使用现场补拍或留在原模型。

## 离线重建与对照

已发布的六个机位均进入 `builder/example.json`。查看 `builder/sample-views/manifest.json` 可还原每个机位的父图、角度和采集方式。按图像摘要保存的遮罩位于 `builder/surface-marks`，深度位于 `builder/surface-depth`。模型导出时会核对遮罩摘要，并保护原图已覆盖的院地以及实体建筑占地。

```powershell
python -X utf8 builder/worker.py build --plan builder/example.json --source village assets/scene_sources/village_road_near.png --out artifacts/rebuilt/scene.glb --report artifacts/rebuilt/report.json --ground-dir artifacts/rebuilt/ground
```

新场景可以在方案中引用自己的源图和标点。将新照片、深度、验收掩膜和同一局部坐标系的 `world` 标点一并交给构建方。用同一套相机矩阵、焦距、光照、画幅和转速录制前后两版，分别检查整院和通过复核的小块，保存整圈回到起点的帧。示例录像每 5° 采一帧，路线文件包含每个机位的位置、朝向、焦距参数以及各机位的停留帧数。[验收用相机路线](village-rear-orbit-route.json) · [同路线对比录像](images/village-rear-round3-comparison.mp4)
