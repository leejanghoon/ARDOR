#utils.py
import genesis as gs
import numpy as np
import trimesh
import math
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append("_3D_recon")
import fusion
import copy
import open3d as o3d
from scipy.spatial import cKDTree 
from skimage import measure    
#############################


def mesh_from_tsdf_safe_colored(tsdf_vol,
                                level: float = 0.0,
                                min_weight: float = 1.0,
                                require_zero_crossing: bool = True):
   
    _ = tsdf_vol.get_volume()
    if getattr(tsdf_vol, "gpu_mode", False):
        try:
            import pycuda.driver as cuda
            cuda.memcpy_dtoh(tsdf_vol._weight_vol_cpu, tsdf_vol._weight_vol_gpu)
        except Exception:
            pass

    tsdf, color_vol = tsdf_vol.get_volume()
    tsdf = tsdf.copy()

    w = getattr(tsdf_vol, "_weight_vol_cpu", None)
    if w is not None and min_weight is not None:
        tsdf[w < float(min_weight)] = 1.0 

    mn, mx = float(tsdf.min()), float(tsdf.max())
    if require_zero_crossing and not (mn <= level <= mx):
        return None

    try:
        v, f, n, _ = measure.marching_cubes(tsdf, level=level)
    except ValueError:
        return None
    if v.size == 0 or f.size == 0:
        return None

    vi = np.rint(v).astype(int)
    vi[:, 0] = np.clip(vi[:, 0], 0, tsdf_vol._vol_dim[0]-1)
    vi[:, 1] = np.clip(vi[:, 1], 0, tsdf_vol._vol_dim[1]-1)
    vi[:, 2] = np.clip(vi[:, 2], 0, tsdf_vol._vol_dim[2]-1)

    rgb_vals = color_vol[vi[:, 0], vi[:, 1], vi[:, 2]]
    color_const = getattr(tsdf_vol, "_color_const", 256*256)
    b = np.floor(rgb_vals / color_const)
    g = np.floor((rgb_vals - b * color_const) / 256)
    r = rgb_vals - b * color_const - g * 256
    colors = np.stack([r, g, b], axis=1).astype(np.uint8)

    v_world = v * tsdf_vol._voxel_size + tsdf_vol._vol_origin

    mesh = trimesh.Trimesh(vertices=v_world, faces=f, vertex_normals=n,
                           vertex_colors=colors, process=False)
    return mesh

def f3(x):
    if isinstance(x, (float, np.floating)):
        return f"{x:.3f}" if np.isfinite(x) else str(x) 
    return x

def mesh_from_tsdf(tsdf_vol) -> trimesh.Trimesh | None:
    verts, faces, norms, _ = tsdf_vol.get_mesh()
    if verts is None or len(verts) == 0 or faces is None or len(faces) == 0:
        return None
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, process=False)

def _auto_nsamples(mesh: trimesh.Trimesh,
                   density_per_m2: float = 5000,
                   n_min: int = 5000,
                   n_max: int = 15000) -> int:
    n = int(np.clip(mesh.area * density_per_m2, n_min, n_max))
    return n

def sample_surface(mesh: trimesh.Trimesh, n_samples: int | None = None) -> np.ndarray:
    """Sample points on the mesh surface."""
    if n_samples is None:
        n_samples = _auto_nsamples(mesh)
    try:
        pts, _ = trimesh.sample.sample_surface_even(mesh, n_samples)
    except Exception:
        pts = mesh.sample(n_samples)
    return pts.astype(np.float32)

def precision_recall_fscore_chamfer(mesh_gt: trimesh.Trimesh,
                                   mesh_rec: trimesh.Trimesh,
                                   tau: float,
                                   n_samples: int | None = None):
    """Surface-based Precision / Recall / F1 @ τ, 
    where τ is a distance threshold (recommended ≈ voxel_size)."""
    pa = sample_surface(mesh_gt, n_samples)
    pb = sample_surface(mesh_rec, n_samples)
    ta, tb = cKDTree(pa), cKDTree(pb)
    # GT→REC (Recall)
    da, _ = tb.query(pa, k=1)
    # REC→GT (Precision)
    db, _ = ta.query(pb, k=1)
    recall = float((da <= tau).mean())
    precision = float((db <= tau).mean())
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    cd=float(da.mean() + db.mean())
    return precision, recall, f1,cd


def entity_to_trimesh(entity):
    """
    Genesis RigidEntity → trimesh.Trimesh
    """
    vertices = []
    faces = []
    base_vcount = 0

    links_quat = entity.get_links_quat()   # shape = (n_env, n_links, 4)
    links_quat = links_quat[0].cpu().numpy()
    links_pos = entity.get_links_pos()
    links_pos = links_pos[0].cpu().numpy()

    for link_idx, link in enumerate(entity.links):
        w, x, y, z = links_quat[link_idx]      
        link_rot = R.from_quat([x, y, z, w])   
        link_p   = links_pos[link_idx]

        for vgeom in link.vgeoms:
            vmesh = vgeom.vmesh
            local_verts = vmesh.verts
            local_faces = vmesh.faces

            vg_w, vg_x, vg_y, vg_z = vgeom.init_quat
            vg_rot = R.from_quat([vg_x, vg_y, vg_z, vg_w])
            vg_p   = np.array(vgeom.init_pos, dtype=float)

        
            lv = vg_rot.apply(local_verts) + vg_p
    
            lv = link_rot.apply(lv) + link_p

            vertices.append(lv)
            faces.append(local_faces + base_vcount)
            base_vcount += len(lv)

    if not vertices:
        return None

    vertices = np.concatenate(vertices, axis=0)
    faces    = np.concatenate(faces, axis=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh

