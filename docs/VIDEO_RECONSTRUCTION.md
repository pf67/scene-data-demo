# 用视频校正已有场景

这次处理的素材有 5 个文件，按完整 SHA-256 去重后是 4 段 1280 × 720、24 fps 的镜头。四段都从同一张村庄图起步，有两段短环绕、两段较长的镜头运动。解算了 114 个抽样机位。能跨帧核对的檐口、可见上层外墙和屋面进入了村庄示例。[实测记录](VALIDATION.md) · [视频同机位对比](images/village-video-orbit.mp4)

## 直接查看和离线复建

在 Windows 上启动发布包并打开村庄，右侧点 **近看相邻屋顶接缝**。拖动环绕可检查这两栋房子的侧面。**保留原图光影** 显示已有照片投影，**全场景重光照** 可单独查看实体网格和烘焙底色。[进入村庄](http://127.0.0.1:8090/?scene=VILLAGE_CALIBRATED&panel=orbit)

发布包内有最终 GLB、示例方案、檐角坐标、已核对的材质和相机报告。在项目根目录执行下面命令，可以用包内成果重建。原视频只在重新求解机位或重新选择纹理时需要。

```powershell
python -m pip install -r builder/requirements.txt
New-Item -ItemType Directory -Force artifacts/rebuilt | Out-Null
python -X utf8 builder/worker.py build --plan builder/example.json --source village assets/scene_sources/village_road_near.png --out artifacts/rebuilt/scene.glb --report artifacts/rebuilt/report.json --ground-dir artifacts/rebuilt/ground
```

## 从新视频开始

1. 准备已标定的场景方案、原始参考图和准备改善的基础 GLB。视频首帧要能找到同一组屋角、屋檐或路口，允许裁切和缩放。对从不同位置开拍的镜头，先单独标定其起始照片。先按文件内容去重，同一个文件的多个副本只参加一次验证。
2. 抽取有重叠的帧，用双向特征匹配估算相机运动。以首帧和已有场景对齐尺度，在独立的图像区域检验深度，记录每帧的相机和跟踪误差。示例中，96 帧的短镜头每 5 帧抽一次，360 帧的长镜头每 10 帧抽一次，均包含结尾帧。
3. 将可见三角面投回视频，逐面跟踪。候选纹理要通过同名建筑、屋面或立面类别、遮挡和边界检查，再比较留出帧。完整屋面可以重贴，局部只在底色相符的位置融合细节。若相机运动与同一屋面上的特征明显矛盾，保留原网格和材质。
4. 需要处理接缝时，在原图和多个独立镜头标记同一圈檐角，使用固定顺序。两屋共边时填写两对端点。墙面分别标出用于跟踪的区域和实际可替换的可见墙面，避开遮挡物。选一个镜头作纹理来源，用至少另外两个镜头检查几何。只接受檐角偏差下降、轮廓不额外侵入相邻建筑的修正。[本次标注文件](examples/village-video-eaves.json)
5. 每栋通过检查的几何和材质都绑定基础网格指纹。构建器在照片投影的遮挡图生成前修正实体屋面和相连墙体，再把材质写入标准 GLB。查看 `report.mesh.surfaceDetail.videoSurfaces` 中的实际采用数量和跳过数量。重用同一路线、镜头和光照对比。[相机路线](village-rear-orbit-route.json)

### Windows 命令

以下在项目根目录运行，假设已将原始视频放入 `video/`。准备其他场景时替换参考图、方案和基础模型。示例以剔除本轮修正的方案先构建基线，避免把同一组檐角校正两次。

```powershell
python -m pip install -r tools/scene_pipeline/requirements-video.txt
$work = 'artifacts/video-job'
$tool = 'tools/scene_pipeline'
$photo = 'assets/scene_sources/village_road_near.png'
New-Item -ItemType Directory -Force $work | Out-Null
python -X utf8 -c "import json,sys,pathlib; p=json.loads(pathlib.Path('builder/example.json').read_bytes()); p.pop('videoEvidence',None); pathlib.Path(sys.argv[1]).write_text(json.dumps(p,ensure_ascii=False),encoding='utf8')" "$work/base.json"
python -X utf8 builder/worker.py build --plan "$work/base.json" --source village $photo --out "$work/base.glb" --report "$work/base-report.json" --ground-dir "$work/ground"
$common = @('--plan',"$work/base.json",'--reference',$photo)
$geometry = @('--geometry',"$work/base.glb")
$shots = @(Get-ChildItem -LiteralPath video -File -Filter '*.mp4' | Get-FileHash -Algorithm SHA256 | Group-Object Hash | ForEach-Object { $_.Group[0].Path })
foreach ($shot in $shots) {
    $count = [int](python -c "import cv2,sys; c=cv2.VideoCapture(sys.argv[1]); print(int(c.get(cv2.CAP_PROP_FRAME_COUNT))); c.release()" $shot)
    $stride = if ($count -le 100) { 5 } else { 10 }
    python -X utf8 "$tool/video_sfm.py" @common @geometry --video $shot --stride $stride --out "$work/sfm"
    if ($LASTEXITCODE) { throw '机位求解未通过，请检查起始视角及抽帧间隔。' }
    python -X utf8 "$tool/video_materials.py" @common @geometry --video $shot --sfm "$work/sfm"
    if ($LASTEXITCODE) { throw '表面跟踪失败。' }
}
python -X utf8 "$tool/video_bake.py" @common @geometry --sfm "$work/sfm" --videos @shots --out "$work/materials"
```

上一步的 `materials` 已包含可构建的材质清单。如果有手工核对的檐角，继续下面两步。示例标注的短哈希取自原视频文件，冒号后为从零计数的帧序号。每个新场景都需使用对应的标注。

```powershell
$survey = 'docs/examples/village-video-eaves.json'
python -X utf8 "$tool/video_eaves.py" @common --sfm "$work/sfm" --survey $survey --videos @shots --out "$work/tracks"
python -X utf8 "$tool/fit_video_eaves.py" @common @geometry --survey $survey --tracks "$work/tracks" --materials "$work/materials" --videos @shots --out "$work/final"
```

将清单和图集装入新方案。只做自动材质提取时，把 `$chosen` 指向 `"$work/materials"`。把示例的 `village` 换成新方案里的源图 ID。HTTP 端也可使用同一方案，向服务提供相同的证据目录即可。

```powershell
$chosen = "$work/final"
$key = (Get-FileHash -Algorithm SHA256 "$chosen/manifest.json").Hash.ToLowerInvariant()
$env:SCENE_VIDEO_EVIDENCE_DIR = (New-Item -ItemType Directory -Force "$work/library").FullName
$dest = New-Item -ItemType Directory -Force (Join-Path $env:SCENE_VIDEO_EVIDENCE_DIR $key)
Copy-Item "$chosen/manifest.json","$chosen/atlas.jpg" -Destination $dest.FullName
python -X utf8 -c "import json,sys,pathlib; p=json.loads(pathlib.Path(sys.argv[1]).read_bytes()); p['videoEvidence']=sys.argv[2]; pathlib.Path(sys.argv[3]).write_text(json.dumps(p,ensure_ascii=False),encoding='utf8')" "$work/base.json" $key "$work/plan.json"
python -X utf8 builder/worker.py build --plan "$work/plan.json" --source village $photo --out "$work/scene.glb" --report "$work/report.json" --ground-dir "$work/final-ground"
```

## 如何判断院子有没有补到

回到背面大院的已校准机位，在地面多边形自身找纹理对应，按地平面和新相机预测像素位置。镜头转过街道中心后可能已拍向村外。此次四块已标记的背院地面在 114 个抽样帧中均未取得足够的对应，沿用上一版约 7.94 平方米的已核对院地。生成视频可以约束可见屋角的相对位置。真实米制高度和镜头从未拍到的区域还需要实拍或测量。
