import torch
import numpy as np

def getEdgeLenth(vertices:torch.Tensor, edges:torch.Tensor):
    '''
    compute edgeLenth
    input:
        vertices : [???,v,3]
        edges : [2,e]
    return:
        lenth: [???,e]
    '''
    assert edges.shape[0] == 2

    v0 = vertices.index_select(dim=-2,index=edges[0])
    v1 = vertices.index_select(dim=-2,index=edges[1])

    return torch.sum((v0 - v1) ** 2,dim=-1) ** 0.5

def getFaceArea(vertices:torch.Tensor, faces:torch.Tensor):
    '''
    compute faceArea
    input:
        vertices : [???,v,3]
        faces : [f,3]
    return:
        area: [???,f]
    '''
    assert faces.shape[-1] == 3

    v0 = vertices.index_select(dim=-2,index=faces[:,0])
    v1 = vertices.index_select(dim=-2,index=faces[:,1])
    v2 = vertices.index_select(dim=-2,index=faces[:,2])

    return (v1 - v0).cross(v2 - v0).norm(dim=-1) / 2



def getMeshNormals(vertices:torch.Tensor, faces:torch.Tensor, normalized=True):

    assert faces.shape[-1] == 3
    input_shape = vertices.shape
    vertices = vertices.view(-1, *input_shape[-2:])
    
    v0 = vertices[..., faces[:,0],:]
    v1 = vertices[..., faces[:,1],:]
    v2 = vertices[..., faces[:,2],:]

    normals = torch.linalg.cross(v1-v0, v2-v1)
    if normalized:
        normals = normals / ( torch.linalg.norm(normals, axis=-1, keepdims=True) + 1e-7)
    return normals.view(*input_shape[:-2], -1, 3)


def getVertexNormals(vertices:torch.Tensor, faces:torch.Tensor, mesh_normals = None, normalized=True):
    """
    TODO: compute mesh area as weight
    input:
        vertices: [batch,seq,v,3]
        faces: m*3
        mesh_noraml: [batch,seq,m,3]
    return:
        vertex_normal: [batch,seq,v,3]
    """
    assert faces.shape[-1] == 3

    if mesh_normals is None:
        mesh_normals = getMeshNormals(vertices, faces, normalized=False)

    vertex_shape = vertices.shape
    vertex_normals = torch.zeros(*vertex_shape, dtype=torch.float32).to(vertices.device)
    vertices = vertices.view(-1, *vertex_shape[-2:]) # [batch*seq,k,3]

    vertex_normals = vertex_normals.index_add(-2,faces[:,0],mesh_normals)
    vertex_normals = vertex_normals.index_add(-2,faces[:,1],mesh_normals)
    vertex_normals = vertex_normals.index_add(-2,faces[:,2],mesh_normals)

    if normalized:
        vertex_normals /= (torch.linalg.norm(vertex_normals, axis=-1, keepdims=True) + 1e-7)
    return vertex_normals.view(vertex_shape)

def getDihedralAngle(face_normals:torch.Tensor, adj_faces_ind:torch.Tensor):
    '''
    get dihedra angle betwen faces
    input:
        faceNormals: [...,m,3]
        faces: [k,2]. face index of two face
    return:
        dihedralAngle: [...,k]
    '''

    face_normals_shape = face_normals.shape
    face_normals = face_normals.view(-1, *face_normals.shape[-2:])

    normals0 = face_normals.index_select(dim=1,index=adj_faces_ind[:,0])
    normals1 = face_normals.index_select(dim=1,index=adj_faces_ind[:,1])

    cos = torch.einsum("ijk,ijk->ij", normals0, normals1)
    sin = torch.norm(torch.linalg.cross(normals0, normals1, dim=-1), dim=-1)

    dihedral_angle = torch.atan2(sin, cos)
    return dihedral_angle.view(*face_normals_shape[:-2], adj_faces_ind.shape[0])





def getFaceAdjacency(faces:np.ndarray):
    '''
    not support any non-manifold mesh
    get face adjacency
    input:
        faces: m*3
    return:
        adjacency: [?,4]. face_index, face_adj_index, common_edge_v0, common_edge_v1
    '''

    eg_fcs = getEdgeToFaces(faces)
    adjacency = []

    for edge, face_list in eg_fcs.items():
        assert len(face_list) == 1 or len(face_list) ==2
        if len(face_list) > 1:
            adjacency.append([face_list[0], face_list[1], edge[0], edge[1]])

    adjacency = np.array(adjacency, dtype=np.int32)
    return adjacency


def getEdgeToFaces(faces:np.ndarray):
    '''
    get a map from edge to faces
    input:
        faces: m*3
    return:
        dict {edgeAsTuple: faceIdx}
    '''
    edge_dict = {}
    for i, face in enumerate(faces):
        s_fc = sorted(face)
        edges = [(s_fc[0], s_fc[1]), (s_fc[1], s_fc[2]), (s_fc[0], s_fc[2])]
        for j, edge in enumerate(edges):
            if edge in edge_dict:
                edge_dict[edge].append(i)
            else:
                edge_dict[edge] = [i]
    return edge_dict



def getEdgesFromFaces(faces, as_dist = False):   
    '''
    extract edge from faces
        faces: n*3 dim faces
        as_dist: return as a dict(adjacency list) or list(edge array, e*2)
    '''
    edges = []

    for face in faces:
        edges.append((face[1], face[0]))
        edges.append((face[2], face[1]))
        edges.append((face[0], face[2]))

        edges.append((face[0], face[1]))
        edges.append((face[1], face[2]))
        edges.append((face[2], face[0]))
    
    unique_edges = list(set(edges))
    if as_dist:
        edge_dict = {}
        for edge in unique_edges:
            u, v = edge
            if u not in edge_dict:
                edge_dict[u] = []
            edge_dict[u].append(v)
            if v not in edge_dict:
                edge_dict[v] = []
            edge_dict[v].append(u)
        return edge_dict
    else:
        return unique_edges


def bfs(index, edges, max_distance = 10):
    from queue import Queue
    '''
        index: startIndex
        edges: a dict contain graph edges
    '''
    index_arr = []
    distance_arr = []
    visited = set()
    queue = Queue()
    queue.put((index,0))
    while queue:
        cur_ind,distance  = queue.get()
        index_arr.append(cur_ind)
        distance_arr.append(distance)
        if cur_ind not in visited:
            visited.add(cur_ind)
            if distance+1 > max_distance:
                break
            for neighbor in edges[cur_ind]:
                if neighbor not in visited:
                    queue.put((neighbor,distance +1))
    return index_arr,distance_arr

def computeAngle(vector_A,vector_B):
    '''
        vector_A: nd array vector
        vector_B: nd array vector
    '''
    direction_A = vector_A / np.linalg.norm(vector_A)
    direction_B = vector_B / np.linalg.norm(vector_B)
    cosinus = np.dot(direction_A,direction_B)
    if abs(cosinus) < 0.70710678118655:
        return np.degrees(np.arccos(cosinus)) # [0, 45)
    else:
        sinus = np.linalg.norm(np.cross(direction_A,direction_B))
        if cosinus < 0.0:
            return np.degrees(3.14159265358979323846 - np.arcsin(sinus)) # (90, 180]
        else:
            return np.degrees(np.arcsin(sinus)) # [45, 90]


from psbody.mesh import Mesh as Obj

def mergeTwoObj(obj1:Obj, obj2:Obj):
    obj_v = np.concatenate([obj1.v,obj2.v], axis=0)
    obj_f = np.concatenate([obj1.f,obj2.f+obj1.v.shape[0]],axis=0)

    return Obj(v=obj_v, f=obj_f)
