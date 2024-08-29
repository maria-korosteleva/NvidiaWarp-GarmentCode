import warp as wp
import numpy as np

def process_body_seg(seg, smpl_parts=False, limbs_merge=False):
    """
        * smpl_parts -- merge smpl segmentation into bigger chunks
        * limbs_merge -- add new labels 'arms' and 'legs', where each combines 
            both arms and both legs into one new label
    """
    # for spines: the toppest is spine2(next to shoulder), then spine1, then spine (next to hip)
    if not smpl_parts and not limbs_merge:
        return seg
    
    if smpl_parts:  # Segmentation has subparts 
        smpl_parts = {
            "left_arm": ['leftArm', 'leftForeArm', 'leftHand', 'leftHandIndex1'],
            "right_arm": ['rightArm', 'rightForeArm', 'rightHand', 'rightHandIndex1'],
            "left_leg": ['leftUpLeg', 'leftLeg', 'leftFoot', 'leftToeBase', 'hips'],
            "right_leg": ['rightUpLeg', 'rightLeg', 'rightFoot', 'rightToeBase', 'hips'],
            "body": ['spine', 'spine1', 'spine2', 'neck', 'leftShoulder', 'rightShoulder', 'hips',
                    'leftUpLeg', 'leftLeg', 'leftFoot', 'leftToeBase',
                    'rightUpLeg', 'rightLeg', 'rightFoot', 'rightToeBase'],
        }

        new_seg = dict()
        for big_part, small_parts in smpl_parts.items():
            new_seg[big_part] = []
            for part in small_parts:
                new_seg[big_part] += seg[part]
    else:
        new_seg = seg
    
    if limbs_merge:
        limbs = {
            'arms': ['left_arm', 'right_arm'],
            'legs': ['left_leg', 'right_leg']
        }
        for big_part, small_parts in limbs.items():
            new_seg[big_part] = []
            for part in small_parts:
                new_seg[big_part] += new_seg[part]
    
    return new_seg


def read_segmentation(path):
    seg_dict = dict()
    with open(path, 'r') as file:
        for idx, row in enumerate(file):
            line = row.rstrip('\n')
            entries = line.split(',')
            if entries[0].startswith('stitch'):
                entry = "stitch"
            else:
                entry = entries[0]

            if entry in seg_dict:
                seg_dict[entry].append(idx)
            else:
                seg_dict[entry] = [idx]
        return seg_dict

def extract_submesh(v, inds, v_sub_indices):
        f = inds.reshape(-1, 3)
        v_sub = v[v_sub_indices]
        reindex_map = {v_old_idx: v_new_idx for v_new_idx, v_old_idx in enumerate(v_sub_indices)}
        f_sub_indices = np.array([all(v_idx in reindex_map for v_idx in face) for face in f])
        f_sub_old_idx = f[f_sub_indices]
        f_sub = np.vectorize(reindex_map.get)(f_sub_old_idx)
        
        return v_sub, f_sub.flatten()

def create_face_filter(_body_verts, _body_indices, _body_seg, parts, smpl_body=False):
    """Create a filter that excldues faces belonging to indicated body parts on a given mesh"""

    body_seg = process_body_seg(_body_seg, smpl_parts=smpl_body)

    body_vert_labels = ['' for _ in range(len(_body_verts))]
    for index, (part, verts) in enumerate(body_seg.items()):
        for vert in verts:
            # FIXME Vertex could have multiple labels -- the last one is assigned!
            body_vert_labels[vert] = part

    filter = []
    for i in range(int(len(_body_indices) / 3)):
        tri = _body_indices[i*3 + 0], _body_indices[i*3 + 1], _body_indices[i*3 + 2]
        part = [0, 0, 0]
        face_filter = [False, False, False]

        part[0], part[1], part[2] = body_vert_labels[tri[0]], body_vert_labels[tri[1]], body_vert_labels[tri[2]] 

        # NOTE: Small number of vertices may not have a lable: they won't be filtered
        face_filter[0], face_filter[1], face_filter[2] = part[0] in parts, part[1] in parts, part[2] in parts

        # Assign label by max voting
        # Two options for filter -> at least two vertices share the label
        if face_filter[0] == face_filter[1] or face_filter[0] == face_filter[2]:
            filter.append(face_filter[0])
        else: 
            filter.append(face_filter[1])

    return filter
        

def assign_face_filter_points(
        panel_verts_label, 
        parts, 
        filter_id, 
        vert_connectivity=None, 
        current_vertex_filter=None,
    ):
    """Add filter id to the cloth vertices that belong to a given part
    
        * vert_connectivity -- information of each vertex's neighbours. 
            If given, it's used to assign filters to unlabeled vertices (e.g. on stitches)
        * current_vertex_filter -- existing filter assignments on vertices
            If given, the function returns the filter assignment merged with
            current_face_filter, with existing assignments having higher priority 
            (e.g. if current_vertex_filter[i] == k and new_vertex_filter[i] == j, k is kept)
    """
    vertex_filter = [-1] * len(panel_verts_label)
    
    for i, label in enumerate(panel_verts_label):
        if label in parts:
            vertex_filter[i] = filter_id
        elif label == -1 and vert_connectivity is not None:
            # Assign by neighbours if not labeled (e.g. it's a stitch vertex)
            neignbours = vert_connectivity[i]
            n_labels = [panel_verts_label[n] for n in neignbours]
            filter_vote_count = sum((n_l in parts for n_l in n_labels))
            if filter_vote_count > len(n_labels) / 2:
                vertex_filter[i] = filter_id

    if current_vertex_filter is not None:
        # Merge with the filter before
        # NOTE: One filter per particle and body shape
        for i in range(len(panel_verts_label)):
            if current_vertex_filter[i] != -1:
                vertex_filter[i] = current_vertex_filter[i]

    return vertex_filter

def panel_assignment(
        panels, panel_verts, panel_indices, panel_transform, 
        _body_seg, _body_verts, _body_indices, body_transform, 
        device, 
        panel_init_labels=None,
        strategy='closest', 
        merge_two_legs=False, 
        smpl_body=False
        ):
    """
    * strategy: 'closest', 'ray_hit'
    * merge_two_legs: Assign 'body' label to panels that should be labeled with one of the legs, 
                      but have ~same number of hits to either leg (e.g. skirt panels)
    """

    body_seg = process_body_seg(_body_seg, smpl_parts=smpl_body)
    body_seg_names = list(body_seg.keys())
    body_verts = _body_verts
    body_indices = _body_indices

    # Invert label assignment
    body_labels = [[] for _ in range(len(body_verts))]
    for index, (part, verts) in enumerate(body_seg.items()):
        for vert in verts:
            body_labels[vert].append(index)
    
    body_shape = wp.Mesh(
        points = wp.array(body_verts, dtype=wp.vec3, device=device),
        indices = wp.array(body_indices, dtype=int, device=device)
    )
    
    body_transform = wp.transform_multiply(body_transform, wp.transform_inverse(panel_transform))

    used_body_parts = set()
    panel_verts_label = [-1] * len(panel_verts)
    for p in panels:
        if p == 'stitch' or p == 'None':
            continue
        # Test per-vertex assignments
        p_vert_index = panels[p]
        p_verts = wp.array(panel_verts[p_vert_index], dtype=wp.vec3, device=device)
        if strategy == 'ray_hit':
            results = _count_ray_hits(
                panel_verts, panel_indices, p_vert_index, p_verts, 
                body_shape.id, body_transform, device=device)
        elif strategy == 'closest':
            results = _count_closest_hits(
                p_verts, body_shape.id, body_transform, device=device)
        
        # process the test result
        statistics = [0] * len(body_seg)
        for hit in results.numpy():
            if hit == -1:
                # no hit
                continue
            # each vertex on the body can belongs to multiple body parts
            f1, f2, f3 = body_indices[hit*3 + 0], body_indices[hit*3 + 1], body_indices[hit*3 + 2]
            for l in body_labels[f1]:
                statistics[l] += 1/3
            for l in body_labels[f2]:
                statistics[l] += 1/3
            for l in body_labels[f3]:
                statistics[l] += 1/3

        if panel_init_labels and panel_init_labels[p]: 
            # Panel has a preferred segmentation (could be less detailed)
            base_label = panel_init_labels[p]

            for i, label in enumerate(body_seg_names): 
                if base_label not in label:
                    statistics[i] = -1    # Cancel out stats of non-matching labels

        max_index = np.argmax(statistics)
        label = body_seg_names[max_index]
        if (merge_two_legs  # TODOLOW Deprecared parameter
                and body_seg_names[max_index] in ['left_leg', 'right_leg']
                and 'pant' not in p     # NOTE: Heuristic: separate legs only for pant panels for more stable drag 
            ):
            label = 'legs'
    
        print("{}:{}".format(p, label))
        used_body_parts.add(label)
        for v in p_vert_index:
            panel_verts_label[v] = label
        
    # FIXME These are different from the call above by one parameter -- why?
    body_seg_for_assignment = process_body_seg(_body_seg, smpl_parts=smpl_body, limbs_merge=True)   

    used_body_seg = {k: v for k, v in body_seg_for_assignment.items() if k in used_body_parts}
    return panel_verts_label, used_body_seg

def _count_closest_hits(p_verts, body_id, body_transform, device):
    """Count closest hits from the cloth vertices to the body"""
    results = wp.zeros(len(p_verts), dtype=wp.int32, device=device)
    wp.launch(
        kernel=panel_assignment_closest_point_test,
        dim = len(p_verts),
        inputs=[body_id,
                body_transform,
                p_verts],
        outputs=[results],
        device=device
    )
    return results

def _count_ray_hits(panel_verts, panel_indices, p_vert_index, p_verts, body_id, body_transform, device):
    """Count ray hits from the cloth vertices to the body"""
    # find any face in the panel to compute the normal of the panel
    for i in range(len(panel_indices)//3):
        f1, f2, f3 = panel_indices[i*3 + 0], panel_indices[i*3 + 1], panel_indices[i*3 + 2]
        if f1 in p_vert_index and f2 in p_vert_index and f3 in p_vert_index:
            break

    normal = -1 * wp.normalize(wp.cross(panel_verts[f2] - panel_verts[f1], panel_verts[f3] - panel_verts[f1]))
    # do a ray hit test
    results = wp.zeros(len(p_verts), dtype=wp.int32, device=device)
    wp.launch(
        kernel=panel_assignment_ray_hit_test,
        dim = len(p_verts),
        inputs=[body_id,
                body_transform,
                p_verts,
                normal],
        outputs=[results],
        device=device
    )

    return results

@wp.kernel
def panel_assignment_ray_hit_test(
    shape: wp.uint64,
    trans: wp.transform,
    particles: wp.array(dtype=wp.vec3),
    ray_dir: wp.vec3,
    hit: wp.array(dtype=wp.int32)
):
    tid = wp.tid()
    p = particles[tid]
    X_ws = trans
    X_sw = wp.transform_inverse(X_ws)
    p_local = wp.transform_point(X_sw, p)
    dir_local = wp.transform_vector(X_sw, ray_dir)
    
    face_index = int(0)
    t = float(0.0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    if wp.mesh_query_ray(shape, p_local, dir_local, 10000.0, t, face_u, face_v, sign, normal, face_index):
        hit[tid] = face_index
    else:
        hit[tid] = -1

@wp.kernel
def panel_assignment_closest_point_test(
    shape: wp.uint64,
    trans: wp.transform,
    particles: wp.array(dtype=wp.vec3),
    # output
    closest: wp.array(dtype=wp.int32)
):
    tid = wp.tid()
    p = particles[tid]
    X_ws = trans
    X_sw = wp.transform_inverse(X_ws)
    p_local = wp.transform_point(X_sw, p)

    sign = float(0.)
    face = int(0)
    u = float(0.)
    v = float(0.)
    
    if wp.mesh_query_point(shape, p_local, 10000.0, sign, face, u, v):
        closest[tid] = face
    else:
        closest[tid] = -1

    # out = wp.mesh_query_point(shape, p_local, 10000.0)
    # if out.result:
    #     closest[tid] = out.face
    # else:
    #     closest[tid] = -1
