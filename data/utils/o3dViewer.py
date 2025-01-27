
import numpy as np
import open3d as o3d
from psbody.mesh import Mesh as Obj


def o3dRender(meshes:list):
    o3d.visualization.draw_geometries(meshes)
    


def getO3dMesh(vertices:np.array,
                faces:np.array,
                v_color = None,
                ):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if v_color is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(v_color)
    else:
        base_colors2 = np.array([
        [1, 0, 0],    # Red
        [0, 1, 0],    # Green
        [0, 0, 1],    # Blue
        [1, 1, 0],    # Yellow
        [1, 0.647, 0], # Orange
        [0.5, 0, 0.5], # Purple
        [0, 1, 1],    # Cyan
        # [0.6, 0.6, 0.6] # Light Gray
        ])
        v_color = np.array([base_colors2[i%7] for i in range(vertices.shape[0])])

    return mesh

def getO3dText(text:str, position:np.array):
    mesh_text =   o3d.t.geometry.TriangleMesh.create_text(text=text)
    mesh_text.scale(0.01,[0,0,0])
    mesh_text.translate(position)
    return mesh_text.to_legacy().paint_uniform_color([1, 0, 0])


def getO3dWeightMesh(vertices:np.array,
                    bone_pos: np.array,
                    weights: np.array,
                    faces=None
                    ):
    # distrib
    base_colors = np.array([
    [0.9, 0.9, 0.7],  # (Light Yellow)
    [0.7, 0.9, 0.7],  # (Light Green)
    [0.7, 0.9, 1],    # (Light Blue)
    [0.9, 0.8, 0.9],  # (Light Purple)
    [1, 0.8, 0.6],    # (Peach Orange)
    # [0.8, 0.7, 0.5],  # (Sand Brown)
    [0.6, 0.8, 0.8],  # (Light Cyan)
    [0.8, 0.8, 0.8],  # (Light Gray)
    ])
    base_colors2 = np.array([
    [1, 0, 0],    # Red
    [0, 1, 0],    # Green
    [0, 0, 1],    # Blue
    [1, 1, 0],    # Yellow
    [1, 0.647, 0], # Orange
    [0.5, 0, 0.5], # Purple
    [0, 1, 1],    # Cyan
    # [0.6, 0.6, 0.6] # Light Gray
])

    bone_colors = np.array([base_colors2[i%7] for i in range(bone_pos.shape[0])])
    vertices_colors = np.matmul(weights, bone_colors)

    if faces is not None:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertices_colors)
    else:
        mesh = o3d.geometry.PointCloud()
        mesh.points = o3d.utility.Vector3dVector(vertices)
        mesh.colors = o3d.utility.Vector3dVector(vertices_colors)


    bones = []
    rad = 0.03
    for i, position in enumerate(bone_pos):
        b = o3d.geometry.TriangleMesh.create_sphere(radius=rad, resolution=5)
        b.translate(position)
        b.paint_uniform_color(bone_colors[i])  # 设置该立方体的颜色
        bones.append(b)

    return [mesh] + bones
    # o3d.visualization.draw_geometries([mesh] + bones)




def renderWeights(vertices:np.array,
                    bone_pos: np.array,
                    weights: np.array,
                    faces=None
                    ):
    # distrib
    base_colors = np.array([
    [0.9, 0.9, 0.7],  # (Light Yellow)
    [0.7, 0.9, 0.7],  # (Light Green)
    [0.7, 0.9, 1],    # (Light Blue)
    [0.9, 0.8, 0.9],  # (Light Purple)
    [1, 0.8, 0.6],    # (Peach Orange)
    # [0.8, 0.7, 0.5],  # (Sand Brown)
    [0.6, 0.8, 0.8],  # (Light Cyan)
    [0.8, 0.8, 0.8],  # (Light Gray)
    ])
    base_colors2 = np.array([
    [1, 0, 0],    # Red
    [0, 1, 0],    # Green
    [0, 0, 1],    # Blue
    [1, 1, 0],    # Yellow
    [1, 0.647, 0], # Orange
    [0.5, 0, 0.5], # Purple
    [0, 1, 1],    # Cyan
    # [0.6, 0.6, 0.6] # Light Gray
])

    bone_colors = np.array([base_colors2[i%7] for i in range(bone_pos.shape[0])])
    vertices_colors = np.matmul(weights, bone_colors)

    if faces is not None:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertices_colors)
    else:
        mesh = o3d.geometry.PointCloud()
        mesh.points = o3d.utility.Vector3dVector(vertices)
        mesh.colors = o3d.utility.Vector3dVector(vertices_colors)


    bones = []
    for i, position in enumerate(bone_pos):
        b = o3d.geometry.TriangleMesh.create_box(width=0.03, height=0.03, depth=0.03)
        b.translate(position)
        b.paint_uniform_color(bone_colors[i])  # 设置该立方体的颜色
        bones.append(b)

    o3d.visualization.draw_geometries([mesh] + bones)

    pass



def showCloth(cloth:Obj,
              vertices_color = None,
              ):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(cloth.v)
    mesh.triangles = o3d.utility.Vector3iVector(cloth.f)
    if vertices_color is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertices_color)
    o3d.visualization.draw_geometries([mesh])
    pass



def RenderSeqNP(v:list,f:np.array,s=0,e=500):
    import time
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for frame in range(s,e):
        vis.clear_geometries()
        for cur_v in v:
            vis.add_geometry(getO3dMesh(cur_v[frame],f))
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1/30)
    vis.destroy_window()



def threadRnederingSeqTensor(vertices_list:list, faces,s=0,e=500):
    import threading
    seq_v = [v.detach().cpu().numpy() for v in vertices_list]
    f = faces.numpy()
    frame_count = seq_v[0].shape[0]
    e = frame_count if e == -1 else e
    render_thread = threading.Thread(target=RenderSeqNP, args=(seq_v,f,s,e))
    render_thread.daemon = True 
    render_thread.start()


