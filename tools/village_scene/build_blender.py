"""Blender background builder: closed photo-matched village, completed facades, PBR lighting."""
from pathlib import Path
import json
import math
import random
import sys

import bpy
from mathutils import Vector
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'assets/village_studio'
DATA = json.loads((OUT/'layout.json').read_text(encoding='utf-8'))
TEXTURES = OUT/'textures'
RNG = random.Random(24)
FRAME = np.array(DATA['frame'])
ORIGIN = np.array(DATA['origin'])
K = np.array(DATA['intrinsics'])
EYE = np.array(DATA['camera']['position'])


def to_blender(v):
    return (float(v[0]),float(-v[2]),float(v[1]))


def project(v):
    p=(np.asarray(v)/DATA['scale']) @ FRAME + ORIGIN
    dw,dh=DATA['depthSize']
    return (float((p[0]/p[2]*K[0,0]+K[0,2])/dw),float(1-(p[1]/p[2]*K[1,1]+K[1,2])/dh))


def on_ground(u,v):
    w,h=DATA['imageSize'];dw,dh=DATA['depthSize']
    ray=np.array([(u/w*dw-K[0,2])/K[0,0],(v/h*dh-K[1,2])/K[1,1],1.]) @ FRAME.T
    return EYE-ray*EYE[1]/ray[1]


def material(name,color=(.5,.5,.5,1),texture=None,roughness=.9):
    mat=bpy.data.materials.new(name);mat.use_nodes=True
    bsdf=next(node for node in mat.node_tree.nodes if node.type=='BSDF_PRINCIPLED')
    bsdf.inputs['Base Color'].default_value=color
    bsdf.inputs['Roughness'].default_value=roughness
    if texture:
        tex=mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex.image=bpy.data.images.load(str(TEXTURES/texture),check_existing=True)
        mat.node_tree.links.new(tex.outputs['Color'],bsdf.inputs['Base Color'])
    return mat


def mesh_object(name,vertices,faces,mats,uvs=None,indices=None):
    mesh=bpy.data.meshes.new(name)
    mesh.from_pydata([to_blender(v) for v in vertices],[],faces);mesh.update()
    ob=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(ob)
    for mat in mats: mesh.materials.append(mat)
    if uvs:
        uv=mesh.uv_layers.new(name='UVMap')
        for face,coords in zip(mesh.polygons,uvs):
            for loop,coord in zip(face.loop_indices,coords): uv.data[loop].uv=coord
    if indices:
        for poly,index in zip(mesh.polygons,indices):poly.material_index=index
    return ob


def box(name,center,size,mat):
    x,y,z=center; a,b,c=[s/2 for s in size]
    verts=[(x-a,y-b,z-c),(x+a,y-b,z-c),(x+a,y-b,z+c),(x-a,y-b,z+c),
           (x-a,y+b,z-c),(x+a,y+b,z-c),(x+a,y+b,z+c),(x-a,y+b,z+c)]
    return mesh_object(name,verts,[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],[mat])


def beam(name,a,b,width,height,mat):
    a=np.array(a);b=np.array(b); mid=(a+b)/2
    ob=box(name,mid,[float(np.linalg.norm(b-a)),height,width],mat)
    # Rotate globally-origin geometry about its own centroid.
    center=Vector(to_blender(mid))
    for v in ob.data.vertices:v.co-=center
    ob.location=center
    direction=Vector(to_blender(b-a))
    ob.rotation_euler[2]=math.atan2(direction.y,direction.x)
    return ob


def build_house(item,index):
    top=np.array(item['points']);n=len(top)
    # Ensure counter-clockwise footprint in Blender XY.
    xy=np.array([[p[0],-p[2]] for p in top])
    if np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1)) < 0:
        top=top[::-1]
    bottom=top.copy();bottom[:,1]=0
    verts=np.concatenate([bottom,top]);faces=[tuple(range(n-1,-1,-1))]
    uv=[[(0,0)]*n];mi=[1]
    for i in range(n):
        j=(i+1)%n
        face=(i,j,j+n,i+n)
        a,b,c=verts[list(face)][:3]
        normal=np.cross(b-a,c-a); normal/=max(np.linalg.norm(normal),1e-5)
        view=EYE-(a+b+c)/3;view/=np.linalg.norm(view)
        photo=normal @ view > .28
        coords=[project(verts[k]) for k in face]
        photo=photo and all(0 <= u <= 1 and 0 <= v <= 1 for u,v in coords)
        face_width=np.linalg.norm(b-a)
        faces.append(face); mi.append(0 if photo else 1)
        uv.append(coords if photo else [(0,0),(face_width/9,0),(face_width/9,item['h']/6.2),(0,item['h']/6.2)])
    if item.get('type')=='gable' and n==4:
        ridge1=(top[0]+top[1])/2;ridge1[1]+=1.25
        ridge2=(top[2]+top[3])/2;ridge2[1]+=1.25
        verts=np.concatenate([verts,[ridge1,ridge2]])
        roofs=[(n,2*n,2*n+1,n+3),(2*n,n+1,n+2,2*n+1),(n,n+1,2*n),(n+2,n+3,2*n+1)]
        for roof in roofs:
            faces.append(roof);uv.append([project(verts[k]) for k in roof]);mi.append(0)
    else:
        faces.append(tuple(range(n,2*n)));uv.append([project(v) for v in top]);mi.append(0)
    ob=mesh_object(item['id'],verts,faces,[PHOTO,FACADES[index%len(FACADES)]],uv,mi)
    ob['source']='photo-traced roof; inferred height; completed hidden facades'
    ob['closed_building']=True
    if not item.get('type'):
        for a,b in zip(top,np.roll(top,-1,axis=0)):
            a=a.copy();b=b.copy();a[1]+=.12;b[1]+=.12
            beam('roof_parapet',a,b,.18,.24,TRIM)
    if index%4==0:
        center=top.mean(0)
        box('rooftop_utility',center+[1,.35,0],[1.0,.7,.8],TRIM)
    if index%3==0 and index>17:
        # Small enclosed yards extend away from the main road.
        road_a=on_ground(0,765);road_b=on_ground(1536,229)
        center=bottom.mean(0);road_direction=road_b-road_a
        nearest=road_a+road_direction*np.clip(np.dot(center-road_a,road_direction)/np.dot(road_direction,road_direction),0,1)
        edge=max(range(n),key=lambda j:np.dot((bottom[j]+bottom[(j+1)%n])/2-center,center-nearest))
        a,b=bottom[edge].copy(),bottom[(edge+1)%n].copy()
        outward=(a+b)/2-bottom.mean(0);outward/=max(np.linalg.norm(outward),.01)
        c=a+outward*3;d=b+outward*3
        for p,q in [(a,c),(c,d),(d,b)]:
            p=p.copy();q=q.copy();p[1]=q[1]=1.0
            beam('courtyard_wall',p,q,.22,2.,BRICK)


def tree(point,height):
    # Keep reconstructed street-side trees out of the carriageway.
    w,h=DATA['imageSize']
    road=[on_ground(u,v) for u,v in [(0,712),(.26*w,.568*h),(.55*w,.408*h),(.79*w,.291*h),
          (w,.205*h),(w,.246*h),(.8*w,.35*h),(.55*w,.48*h),(.27*w,.656*h),(0,.794*h)]]
    inside=False;closest=None;min_distance=float('inf')
    for a,b in zip(road,road[1:]+road[:1]):
        if (a[2]>point[2]) != (b[2]>point[2]):
            if point[0] < (b[0]-a[0])*(point[2]-a[2])/(b[2]-a[2])+a[0]:inside=not inside
        delta=b-a;t=np.clip(np.dot(point-a,delta)/np.dot(delta,delta),0,1)
        nearest=a+delta*t;distance=float(np.linalg.norm(point-nearest))
        if distance<min_distance:min_distance=distance;closest=nearest
    if inside and min_distance>.001:point=closest+(closest-point)/min_distance*2.5
    x,_,z=point
    trunk_vertices=[(x+.14*math.cos(i*math.tau/7),y,z+.14*math.sin(i*math.tau/7))
                    for y in (0,height*.67) for i in range(7)]
    trunk_faces=[tuple(range(6,-1,-1)),tuple(range(7,14))]
    trunk_faces.extend((i,(i+1)%7,(i+1)%7+7,i+7) for i in range(7))
    mesh_object('tree_trunk',trunk_vertices,trunk_faces,[WOOD])
    golden=(1+math.sqrt(5))/2
    base=np.array([[-1,golden,0],[1,golden,0],[-1,-golden,0],[1,-golden,0],
                   [0,-1,golden],[0,1,golden],[0,-1,-golden],[0,1,-golden],
                   [golden,0,-1],[golden,0,1],[-golden,0,-1],[-golden,0,1]])
    base/=np.linalg.norm(base[0])
    ico_faces=[(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),(1,5,9),(5,11,4),
               (11,10,2),(10,7,6),(7,1,8),(3,9,4),(3,4,2),(3,2,6),(3,6,8),
               (3,8,9),(4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)]
    vertices=[];faces=[];indices=[]
    for _ in range(54):
        angle=RNG.uniform(0,math.tau);radius=math.sqrt(RNG.random())*2.2
        loc=np.array((x+math.cos(angle)*radius,height*RNG.uniform(.56,.93),z+math.sin(angle)*radius))
        scale=np.array([1,RNG.uniform(1,1.6),1])*RNG.uniform(.45,.85)
        offset=len(vertices)
        vertices.extend(loc+v*scale*RNG.uniform(.68,1.22) for v in base)
        faces.extend(tuple(offset+i for i in f) for f in ico_faces)
        indices.extend([RNG.randrange(len(LEAVES))]*len(ico_faces))
    canopy=mesh_object('tree_crown',vertices,faces,LEAVES,indices=indices)
    for p in canopy.data.polygons:p.use_smooth=True


def join_detail(prefix):
    objects=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.name.startswith(prefix)]
    if not objects:return
    bpy.ops.object.select_all(action='DESELECT')
    for ob in objects:ob.select_set(True)
    bpy.context.view_layer.objects.active=objects[0]
    bpy.ops.object.join();bpy.context.object.name=prefix+'_combined'


bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
PHOTO=material('Photo-derived neutral albedo',texture='photo_albedo.jpg')
FACADES=[material(f'Completed plaster facade {i}',texture=f'facade_{i}.jpg') for i in range(4)]
GROUND=material('Courtyards and road',texture='ground.jpg')
TRIM=material('Weathered concrete trim',(.43,.41,.36,1))
BRICK=material('Courtyard brick',(.35,.24,.16,1))
WOOD=material('Tree bark',(.12,.095,.07,1))
LEAVES=[material(f'Leaves {i}',color) for i,color in enumerate([(.14,.20,.08,1),(.18,.24,.10,1),(.12,.17,.07,1),(.21,.27,.12,1)])]
STEEL=material('Pole steel',(.25,.26,.24,1),roughness=.7)

for i,item in enumerate(DATA['buildings']):build_house(item,i)
mesh_object('village_ground',[(-200,-.04,-330),(200,-.04,-330),(200,-.04,80),(-200,-.04,80)],
            [(0,3,2,1)],[GROUND],[[(0,1),(0,0),(1,0),(1,1)]])
box('earth_foundation',(0,-1.05,-125),(400,2,410),material('Earth',(.23,.21,.15,1)))

# Trees follow the observed tree line and greenery between the compounds.
for u,v in [(310,131),(983,156),(999,144),(1152,182),(320,124),(370,116),(949,157),(923,157),
             (970,186),(1108,207),(1454,287),(1470,272),(683,404),(880,481),(711,219)]:
    tree(on_ground(u,v),RNG.uniform(5,8))
for _ in range(88):
    u=RNG.uniform(50,1530);v=RNG.uniform(8,53)
    p=on_ground(u,v)
    if -315<p[2]<-90 and abs(p[0])<190:tree(p,RNG.uniform(5.5,9))

pole_points=[]
for u,v in [(392,595),(666,455),(867,370),(1091,304),(1300,246)]:
    p=on_ground(u,v);pole_points.append(p)
    box('utility_pole',p+[0,4.7,0],(.14,9.4,.14),STEEL)
    box('pole_crossarm',p+[0,8.4,0],(1.5,.1,.1),STEEL)
for a,b in zip(pole_points,pole_points[1:]):
    for offset in [-.5,0,.5]:
        curve=bpy.data.curves.new('overhead_wire','CURVE');curve.dimensions='3D';curve.bevel_depth=.018;curve.bevel_resolution=0
        spline=curve.splines.new('POLY');spline.points.add(12)
        for i,p in enumerate(spline.points):
            t=i/12;v=a*(1-t)+b*t+[offset,8.4-math.sin(t*math.pi)*.7,0]
            p.co=(*to_blender(v),1)
        ob=bpy.data.objects.new('overhead_wire',curve);bpy.context.collection.objects.link(ob);curve.materials.append(STEEL)

for prefix in ['roof_parapet','rooftop_utility','courtyard_wall','tree_trunk','tree_crown','utility_pole','pole_crossarm']:
    join_detail(prefix)

scene=bpy.context.scene
scene['scene_kind']='photo_completed_village'
scene['camera_position']=DATA['camera']['position']
scene['camera_target']=DATA['camera']['target']
scene['camera_fov']=DATA['camera']['fov']
scene['preserve_scale']=True
scene['building_count']=len(DATA['buildings'])
scene['source_image']='village_road_near.png'
scene['completion_note']='64 traced roof outlines; hidden walls/ground inferred and completed in Blender.'

bpy.ops.object.camera_add(location=to_blender(DATA['camera']['position']))
cam=bpy.context.object;cam.name='Reference photo camera';scene.camera=cam
cam.data.type='PERSP';cam.data.lens=35;cam.data.clip_end=1000
cam.data.sensor_fit='VERTICAL';cam.data.angle_y=math.radians(DATA['camera']['fov'])
target=Vector(to_blender(DATA['camera']['target']));cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.light_add(type='SUN',location=(30,20,60))
sun=bpy.context.object;sun.name='Editable Sun';sun.data.energy=2.1;sun.data.angle=math.radians(3)
sun.rotation_euler=(math.radians(32),math.radians(-22),math.radians(-30))
scene.world.color=(.24,.27,.31)
scene.render.engine='CYCLES';scene.cycles.samples=24
scene.cycles.use_denoising=True
try:
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
    for device in prefs.devices:device.use=device.type!='CPU'
    scene.cycles.device='GPU'
except Exception:pass
scene.view_settings.view_transform='AgX'
scene.render.resolution_x=1200;scene.render.resolution_y=800;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'

# Export mesh-only scene; camera/light extras tell the existing editor how to frame it.
bpy.ops.object.select_all(action='DESELECT')
for ob in scene.objects:
    if ob.type in {'MESH','CURVE'}:ob.select_set(True)
dest=ROOT/'assets/scene_meshes/VILLAGE_COMPLETE.glb'
dest.parent.mkdir(parents=True,exist_ok=True)
bpy.ops.export_scene.gltf(filepath=str(dest),export_format='GLB',use_selection=True,export_extras=True,export_animations=False)
meta={'code':'VILLAGE_COMPLETE','label':'村庄 · 完整建筑与背面','kind':'photo_completed_village','map_id':'VILLAGE_CN',
      'building_count':len(DATA['buildings']),'preserve_scale':True,'source_image':'village_road_near.png',
      'camera':DATA['camera'],'notes':'原图屋顶轮廓 + 推断高度 + Blender 闭合建筑与背面材质；非实测建筑背面。'}
dest.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'village_complete.blend'))

if '--render' in sys.argv:
    for name,position in [('reference',DATA['camera']['position']),('reverse',[-55,48,-90]),('overview',[80,100,75])]:
        cam.location=to_blender(position);cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
        scene.render.filepath=str(OUT/f'{name}.png');bpy.ops.render.render(write_still=True)
print('VILLAGE_COMPLETE_READY',dest)
