"""Calibrate the supplied photograph and prepare editable building footprints/textures."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/village_studio"
TEXTURES = OUT / "textures"


def prepare():
    TEXTURES.mkdir(parents=True, exist_ok=True)
    rgb = np.array(Image.open(ROOT / "assets/scene_sources/village_road_near.png").convert("RGB"))
    h, w = rgb.shape[:2]
    depth_root = ROOT / "assets/da3_out/village_road_near"
    depth = np.load(depth_root / "depth.npy")
    k = np.load(depth_root / "K.npy")
    dh, dw = depth.shape
    v, u = np.mgrid[:dh, :dw]
    points = np.stack([(u-k[0,2])*depth/k[0,0], (v-k[1,2])*depth/k[1,1], depth], -1)
    road = np.array([[0,.722],[.26,.568],[.55,.408],[.79,.291],[1,.205],
                     [1,.24],[.8,.35],[.55,.48],[.27,.656],[0,.794]])
    mask = np.zeros((dh,dw), np.uint8)
    cv2.fillPoly(mask, [(road * [dw,dh]).astype(np.int32)], 1)
    sample = points[mask > 0][::3]
    rng = np.random.default_rng(7)
    best = np.ones(len(sample), bool)
    best_count = 0
    for _ in range(180):
        a,b,c = sample[rng.choice(len(sample),3,replace=False)]
        n = np.cross(b-a,c-a)
        n /= max(np.linalg.norm(n),1e-8)
        inside = np.abs((sample-a) @ n) < .026
        if inside.sum() > best_count:
            best, best_count = inside, inside.sum()
    origin = np.median(sample[best],axis=0)
    _,_,basis = np.linalg.svd(sample[best]-sample[best].mean(0),full_matrices=False)
    normal = basis[-1]
    if normal @ origin > 0:
        normal = -normal
    right = np.array([1.,0.,0.])-normal*normal[0]
    right /= np.linalg.norm(right)
    frame = np.stack([right,normal,np.cross(right,normal)])
    scale = 90 / np.median(depth)
    eye = -origin @ frame.T * scale

    def unproject(pixels, height=0):
        pixels=np.asarray(pixels,dtype=float)
        uv=pixels*[dw/w,dh/h]
        rays=np.stack([(uv[:,0]-k[0,2])/k[0,0],(uv[:,1]-k[1,2])/k[1,1],np.ones(len(uv))],-1) @ frame.T
        t=(height-eye[1])/rays[:,1]
        return eye + rays*t[:,None]

    def project(vertices):
        cv=(np.asarray(vertices)/scale) @ frame + origin
        return np.stack([(cv[:,0]/cv[:,2]*k[0,0]+k[0,2])*w/dw,
                         (cv[:,1]/cv[:,2]*k[1,1]+k[1,2])*h/dh],-1)

    layout=json.loads((Path(__file__).parent/'roof_layout.json').read_text(encoding='utf-8'))
    buildings=[]
    for index,item in enumerate(layout['buildings']):
        roof=np.array(item['roof'])
        verts=unproject(roof,item['h'])
        buildings.append({**item,'id':f'house_{index+1:02d}','points':verts.tolist()})

    # A clean courtyard/road ground excludes projected building silhouettes.
    gx=np.linspace(-200,200,2048);gz=np.linspace(-330,80,2048)
    xx,zz=np.meshgrid(gx,gz)
    pixels=project(np.stack([xx.ravel(),np.zeros(xx.size),zz.ravel()],-1)).reshape(2048,2048,2).astype(np.float32)
    observed=cv2.remap(rgb,pixels[:,:,0],pixels[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(99,108,72))
    coarse=cv2.resize(rng.normal(0,1,(64,64)).astype(np.float32),(2048,2048),interpolation=cv2.INTER_CUBIC)
    grain=rng.normal(0,4,(2048,2048))
    ground=np.clip(np.array([142,133,113])+coarse[:,:,None]*8+grain[:,:,None],0,255).astype(np.uint8)
    # Retain observed farmland only; replace hidden courtyards with a shadow-free
    # dirt/concrete albedo instead of stretching buildings over the ground.
    fields=(pixels[:,:,1]<88)&(pixels[:,:,1]>=0)&(pixels[:,:,0]>=0)&(pixels[:,:,0]<w)
    ground[fields]=observed[fields]
    outside=(pixels[:,:,1]<0)|(pixels[:,:,0]<0)|(pixels[:,:,0]>=w)|(pixels[:,:,1]>=h)
    ground[outside]=np.clip(np.array([112,121,82])+grain[outside,None]+coarse[outside,None]*6,0,255)
    def ground_pixels(points):
        p=np.asarray(points)
        return np.stack([(p[:,0]+200)/400*2047,(p[:,2]+330)/410*2047],-1).astype(np.int32)
    for b in buildings:
        p=np.array(b['points']);center=p.mean(0);p=center+(p-center)*1.28
        tone=int(rng.integers(144,168))
        cv2.fillConvexPoly(ground,ground_pixels(p),(tone,tone-5,tone-14))
    road_uv=np.array([[0,712],[.26*w,.568*h],[.55*w,.408*h],[.79*w,.291*h],[w,.205*h],
                       [w,.246*h],[.8*w,.35*h],[.55*w,.48*h],[.27*w,.656*h],[0,.794*h]])
    road_world=unproject(road_uv)
    road_mask=np.zeros((2048,2048),np.uint8)
    cv2.fillPoly(road_mask,[ground_pixels(road_world)],255)
    cv2.polylines(ground,[ground_pixels(road_world[:5]),ground_pixels(road_world[5:])],False,(186,179,161),13)
    asphalt=np.clip(np.array([80,85,84])+grain[:,:,None]*.8+coarse[:,:,None]*2,0,255).astype(np.uint8)
    ground[road_mask>0]=asphalt[road_mask>0]
    centers=unproject([[0,765],[400,604],[800,448],[1200,307],[1536,229]])
    for a,b in zip(centers,centers[1:]):
        length=np.linalg.norm(b-a)
        for distance in np.arange(0,length,6):
            line=ground_pixels([a+(b-a)*distance/length,a+(b-a)*min(distance+2.8,length)/length])
            cv2.line(ground,tuple(line[0]),tuple(line[1]),(184,177,145),1,cv2.LINE_AA)
    ground=np.clip(ground.astype(float)+rng.normal(0,2,(2048,2048,1)),0,255).astype(np.uint8)
    Image.fromarray(ground).save(TEXTURES/'ground.jpg',quality=92)
    # Keep photographic roof detail, compress baked contrast for usable relighting.
    photo=np.clip(rgb.astype(float)*.82+29,0,255).astype(np.uint8)
    Image.fromarray(photo).save(TEXTURES/'photo_albedo.jpg',quality=97)

    for variant,base_color in enumerate([(172,166,151),(175,168,150),(147,137,120),(186,181,168)]):
        noise=rng.normal(0,6,(512,512,1))
        tex=np.clip(np.array(base_color)+noise,0,255).astype(np.uint8)
        im=Image.fromarray(tex);d=ImageDraw.Draw(im)
        # Plaster, brick skirting, lintels and window frames, repeating every 8m.
        d.rectangle([0,440,511,511],fill=(133,111,87))
        for yy in range(444,512,12):
            d.line([0,yy,512,yy],fill=(157,139,113),width=1)
            for x in range((yy//12%2)*22,512,44):
                d.line([x,yy,x,yy+12],fill=(157,139,113))
        for yy in (92,296):
            for x in (45,219,393):
                d.rectangle([x-5,yy-5,x+78,yy+106],fill=(209,204,185))
                d.rectangle([x,yy,x+73,yy+100],fill=(49,64,65))
                d.rectangle([x+5,yy+5,x+32,yy+46],fill=(72,87,87))
                d.rectangle([x+39,yy+5,x+68,yy+46],fill=(76,89,86))
                d.line([x+36,yy,x+36,yy+100],fill=(193,190,169),width=4)
                d.line([x,yy+52,x+73,yy+52],fill=(185,182,163),width=4)
                d.line([x-6,yy+108,x+80,yy+108],fill=(145,140,121),width=4)
        im.save(TEXTURES/f'facade_{variant}.jpg',quality=94)

    optical_target=eye+frame[:,2]*(-eye[1]/frame[1,2])
    calibration={'frame':frame.tolist(),'origin':origin.tolist(),'scale':float(scale),'intrinsics':k.tolist(),
                 'imageSize':[w,h],'depthSize':[dw,dh], 'camera':{'position':eye.tolist(),'target':optical_target.tolist(),
                 'fov':float(np.degrees(2*np.arctan(dh/(2*k[1,1]))))},
                 'groundBounds':[-200,200,-330,80], 'buildings':buildings}
    (OUT/'layout.json').write_text(json.dumps(calibration,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Prepared {len(buildings)} traced buildings; camera {eye.round(2)}')


if __name__=='__main__':
    prepare()
