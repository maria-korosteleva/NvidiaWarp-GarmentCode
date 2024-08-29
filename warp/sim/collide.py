# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Collision handling functions and kernels.
"""

import warp as wp
from .model import PARTICLE_FLAG_ACTIVE, ModelShapeGeometry


@wp.func
def triangle_closest_point_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)

    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)

    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return wp.vec3(1.0 - v, v, 0.0)

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)

    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return wp.vec3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return wp.vec3(0.0, w, 1.0 - w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    return wp.vec3(1.0 - v - w, v, w)


@wp.func
def sphere_sdf(center: wp.vec3, radius: float, p: wp.vec3):
    return wp.length(p - center) - radius


@wp.func
def sphere_sdf_grad(center: wp.vec3, radius: float, p: wp.vec3):
    return wp.normalize(p - center)


@wp.func
def box_sdf(upper: wp.vec3, p: wp.vec3):
    # adapted from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    qx = abs(p[0]) - upper[0]
    qy = abs(p[1]) - upper[1]
    qz = abs(p[2]) - upper[2]

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))

    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)


@wp.func
def box_sdf_grad(upper: wp.vec3, p: wp.vec3):
    qx = abs(p[0]) - upper[0]
    qy = abs(p[1]) - upper[1]
    qz = abs(p[2]) - upper[2]

    # exterior case
    if qx > 0.0 or qy > 0.0 or qz > 0.0:
        x = wp.clamp(p[0], -upper[0], upper[0])
        y = wp.clamp(p[1], -upper[1], upper[1])
        z = wp.clamp(p[2], -upper[2], upper[2])

        return wp.normalize(p - wp.vec3(x, y, z))

    sx = wp.sign(p[0])
    sy = wp.sign(p[1])
    sz = wp.sign(p[2])

    # x projection
    if qx > qy and qx > qz or qy == 0.0 and qz == 0.0:
        return wp.vec3(sx, 0.0, 0.0)

    # y projection
    if qy > qx and qy > qz or qx == 0.0 and qz == 0.0:
        return wp.vec3(0.0, sy, 0.0)

    # z projection
    return wp.vec3(0.0, 0.0, sz)


@wp.func
def capsule_sdf(radius: float, half_height: float, p: wp.vec3):
    if p[1] > half_height:
        return wp.length(wp.vec3(p[0], p[1] - half_height, p[2])) - radius

    if p[1] < -half_height:
        return wp.length(wp.vec3(p[0], p[1] + half_height, p[2])) - radius

    return wp.length(wp.vec3(p[0], 0.0, p[2])) - radius


@wp.func
def capsule_sdf_grad(radius: float, half_height: float, p: wp.vec3):
    if p[1] > half_height:
        return wp.normalize(wp.vec3(p[0], p[1] - half_height, p[2]))

    if p[1] < -half_height:
        return wp.normalize(wp.vec3(p[0], p[1] + half_height, p[2]))

    return wp.normalize(wp.vec3(p[0], 0.0, p[2]))


@wp.func
def cylinder_sdf(radius: float, half_height: float, p: wp.vec3):
    dx = wp.length(wp.vec3(p[0], 0.0, p[2])) - radius
    dy = wp.abs(p[1]) - half_height
    return wp.min(wp.max(dx, dy), 0.0) + wp.length(wp.vec2(wp.max(dx, 0.0), wp.max(dy, 0.0)))


@wp.func
def cylinder_sdf_grad(radius: float, half_height: float, p: wp.vec3):
    dx = wp.length(wp.vec3(p[0], 0.0, p[2])) - radius
    dy = wp.abs(p[1]) - half_height
    if dx > dy:
        return wp.normalize(wp.vec3(p[0], 0.0, p[2]))
    return wp.vec3(0.0, wp.sign(p[1]), 0.0)


@wp.func
def cone_sdf(radius: float, half_height: float, p: wp.vec3):
    dx = wp.length(wp.vec3(p[0], 0.0, p[2])) - radius * (p[1] + half_height) / (2.0 * half_height)
    dy = wp.abs(p[1]) - half_height
    return wp.min(wp.max(dx, dy), 0.0) + wp.length(wp.vec2(wp.max(dx, 0.0), wp.max(dy, 0.0)))


@wp.func
def cone_sdf_grad(radius: float, half_height: float, p: wp.vec3):
    dx = wp.length(wp.vec3(p[0], 0.0, p[2])) - radius * (p[1] + half_height) / (2.0 * half_height)
    dy = wp.abs(p[1]) - half_height
    if dy < 0.0 or dx == 0.0:
        return wp.vec3(0.0, wp.sign(p[1]), 0.0)
    return wp.normalize(wp.vec3(p[0], 0.0, p[2])) + wp.vec3(0.0, radius / (2.0 * half_height), 0.0)


@wp.func
def plane_sdf(width: float, length: float, p: wp.vec3):
    # SDF for a quad in the xz plane
    if width > 0.0 and length > 0.0:
        d = wp.max(wp.abs(p[0]) - width, wp.abs(p[2]) - length)
        return wp.max(d, wp.abs(p[1]))
    return p[1]


@wp.func
def closest_point_plane(width: float, length: float, point: wp.vec3):
    # projects the point onto the quad in the xz plane (if width and length > 0.0, otherwise the plane is infinite)
    if width > 0.0:
        x = wp.clamp(point[0], -width, width)
    else:
        x = point[0]
    if length > 0.0:
        z = wp.clamp(point[2], -length, length)
    else:
        z = point[2]
    return wp.vec3(x, 0.0, z)


@wp.func
def closest_point_line_segment(a: wp.vec3, b: wp.vec3, point: wp.vec3):
    ab = b - a
    ap = point - a
    t = wp.dot(ap, ab) / wp.dot(ab, ab)
    t = wp.clamp(t, 0.0, 1.0)
    return a + t * ab


@wp.func
def closest_point_box(upper: wp.vec3, point: wp.vec3):
    # closest point to box surface
    x = wp.clamp(point[0], -upper[0], upper[0])
    y = wp.clamp(point[1], -upper[1], upper[1])
    z = wp.clamp(point[2], -upper[2], upper[2])
    if wp.abs(point[0]) <= upper[0] and wp.abs(point[1]) <= upper[1] and wp.abs(point[2]) <= upper[2]:
        # the point is inside, find closest face
        sx = wp.abs(wp.abs(point[0]) - upper[0])
        sy = wp.abs(wp.abs(point[1]) - upper[1])
        sz = wp.abs(wp.abs(point[2]) - upper[2])
        # return closest point on closest side, handle corner cases
        if sx < sy and sx < sz or sy == 0.0 and sz == 0.0:
            x = wp.sign(point[0]) * upper[0]
        elif sy < sx and sy < sz or sx == 0.0 and sz == 0.0:
            y = wp.sign(point[1]) * upper[1]
        else:
            z = wp.sign(point[2]) * upper[2]
    return wp.vec3(x, y, z)


@wp.func
def get_box_vertex(point_id: int, upper: wp.vec3):
    # get the vertex of the box given its ID (0-7)
    sign_x = float(point_id % 2) * 2.0 - 1.0
    sign_y = float((point_id // 2) % 2) * 2.0 - 1.0
    sign_z = float((point_id // 4) % 2) * 2.0 - 1.0
    return wp.vec3(sign_x * upper[0], sign_y * upper[1], sign_z * upper[2])


@wp.func
def get_box_edge(edge_id: int, upper: wp.vec3):
    # get the edge of the box given its ID (0-11)
    if edge_id < 4:
        # edges along x: 0-1, 2-3, 4-5, 6-7
        i = edge_id * 2
        j = i + 1
        return wp.spatial_vector(get_box_vertex(i, upper), get_box_vertex(j, upper))
    elif edge_id < 8:
        # edges along y: 0-2, 1-3, 4-6, 5-7
        edge_id -= 4
        i = edge_id % 2 + edge_id // 2 * 4
        j = i + 2
        return wp.spatial_vector(get_box_vertex(i, upper), get_box_vertex(j, upper))
    # edges along z: 0-4, 1-5, 2-6, 3-7
    edge_id -= 8
    i = edge_id
    j = i + 4
    return wp.spatial_vector(get_box_vertex(i, upper), get_box_vertex(j, upper))


@wp.func
def get_plane_edge(edge_id: int, plane_width: float, plane_length: float):
    # get the edge of the plane given its ID (0-3)
    p0x = (2.0 * float(edge_id % 2) - 1.0) * plane_width
    p0z = (2.0 * float(edge_id // 2) - 1.0) * plane_length
    if edge_id == 0 or edge_id == 3:
        p1x = p0x
        p1z = -p0z
    else:
        p1x = -p0x
        p1z = p0z
    return wp.spatial_vector(wp.vec3(p0x, 0.0, p0z), wp.vec3(p1x, 0.0, p1z))


@wp.func
def closest_edge_coordinate_box(upper: wp.vec3, edge_a: wp.vec3, edge_b: wp.vec3, max_iter: int):
    # find point on edge closest to box, return its barycentric edge coordinate
    # Golden-section search
    a = float(0.0)
    b = float(1.0)
    h = b - a
    invphi = 0.61803398875  # 1 / phi
    invphi2 = 0.38196601125  # 1 / phi^2
    c = a + invphi2 * h
    d = a + invphi * h
    query = (1.0 - c) * edge_a + c * edge_b
    yc = box_sdf(upper, query)
    query = (1.0 - d) * edge_a + d * edge_b
    yd = box_sdf(upper, query)

    for k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = box_sdf(upper, query)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = box_sdf(upper, query)

    if yc < yd:
        return 0.5 * (a + d)
    return 0.5 * (c + b)


@wp.func
def closest_edge_coordinate_plane(
    plane_width: float,
    plane_length: float,
    edge_a: wp.vec3,
    edge_b: wp.vec3,
    max_iter: int,
):
    # find point on edge closest to plane, return its barycentric edge coordinate
    # Golden-section search
    a = float(0.0)
    b = float(1.0)
    h = b - a
    invphi = 0.61803398875  # 1 / phi
    invphi2 = 0.38196601125  # 1 / phi^2
    c = a + invphi2 * h
    d = a + invphi * h
    query = (1.0 - c) * edge_a + c * edge_b
    yc = plane_sdf(plane_width, plane_length, query)
    query = (1.0 - d) * edge_a + d * edge_b
    yd = plane_sdf(plane_width, plane_length, query)

    for k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = plane_sdf(plane_width, plane_length, query)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = plane_sdf(plane_width, plane_length, query)

    if yc < yd:
        return 0.5 * (a + d)
    return 0.5 * (c + b)


@wp.func
def closest_edge_coordinate_capsule(radius: float, half_height: float, edge_a: wp.vec3, edge_b: wp.vec3, max_iter: int):
    # find point on edge closest to capsule, return its barycentric edge coordinate
    # Golden-section search
    a = float(0.0)
    b = float(1.0)
    h = b - a
    invphi = 0.61803398875  # 1 / phi
    invphi2 = 0.38196601125  # 1 / phi^2
    c = a + invphi2 * h
    d = a + invphi * h
    query = (1.0 - c) * edge_a + c * edge_b
    yc = capsule_sdf(radius, half_height, query)
    query = (1.0 - d) * edge_a + d * edge_b
    yd = capsule_sdf(radius, half_height, query)

    for k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = capsule_sdf(radius, half_height, query)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = capsule_sdf(radius, half_height, query)

    if yc < yd:
        return 0.5 * (a + d)

    return 0.5 * (c + b)


@wp.func
def mesh_sdf(mesh: wp.uint64, point: wp.vec3, max_dist: float):
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    res = wp.mesh_query_point_sign_normal(mesh, point, max_dist, sign, face_index, face_u, face_v)

    if res:
        closest = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
        return wp.length(point - closest) * sign
    return max_dist


@wp.func
def closest_point_mesh(mesh: wp.uint64, point: wp.vec3, max_dist: float):
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    res = wp.mesh_query_point_sign_normal(mesh, point, max_dist, sign, face_index, face_u, face_v)

    if res:
        return wp.mesh_eval_position(mesh, face_index, face_u, face_v)
    # return arbitrary point from mesh
    return wp.mesh_eval_position(mesh, 0, 0.0, 0.0)


@wp.func
def closest_edge_coordinate_mesh(mesh: wp.uint64, edge_a: wp.vec3, edge_b: wp.vec3, max_iter: int, max_dist: float):
    # find point on edge closest to mesh, return its barycentric edge coordinate
    # Golden-section search
    a = float(0.0)
    b = float(1.0)
    h = b - a
    invphi = 0.61803398875  # 1 / phi
    invphi2 = 0.38196601125  # 1 / phi^2
    c = a + invphi2 * h
    d = a + invphi * h
    query = (1.0 - c) * edge_a + c * edge_b
    yc = mesh_sdf(mesh, query, max_dist)
    query = (1.0 - d) * edge_a + d * edge_b
    yd = mesh_sdf(mesh, query, max_dist)

    for k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = mesh_sdf(mesh, query, max_dist)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = mesh_sdf(mesh, query, max_dist)

    if yc < yd:
        return 0.5 * (a + d)
    return 0.5 * (c + b)


@wp.func
def volume_grad(volume: wp.uint64, p: wp.vec3):
    eps = 0.05  # TODO make this a parameter
    q = wp.volume_world_to_index(volume, p)

    # compute gradient of the SDF using finite differences
    dx = wp.volume_sample_f(volume, q + wp.vec3(eps, 0.0, 0.0), wp.Volume.LINEAR) - wp.volume_sample_f(
        volume, q - wp.vec3(eps, 0.0, 0.0), wp.Volume.LINEAR
    )
    dy = wp.volume_sample_f(volume, q + wp.vec3(0.0, eps, 0.0), wp.Volume.LINEAR) - wp.volume_sample_f(
        volume, q - wp.vec3(0.0, eps, 0.0), wp.Volume.LINEAR
    )
    dz = wp.volume_sample_f(volume, q + wp.vec3(0.0, 0.0, eps), wp.Volume.LINEAR) - wp.volume_sample_f(
        volume, q - wp.vec3(0.0, 0.0, eps), wp.Volume.LINEAR
    )

    return wp.normalize(wp.vec3(dx, dy, dz))

@wp.func
def soft_collision_filter(
    geo_filter: wp.array2d(dtype=wp.int32),
    particle_filter: wp.int32,
    face_index: wp.int32,

):
    if particle_filter > -1:
        if geo_filter[particle_filter][face_index]: 
            return True
    return False

@wp.kernel
def create_soft_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    body_X_wb: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    geo: ModelShapeGeometry,
    margin: float,
    soft_contact_max: int,
    # outputs
    soft_contact_count: wp.array(dtype=int),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_shape: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    cloth_reference_drag_particles: wp.array(dtype=int),

):
    particle_index, shape_index = wp.tid()
    if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    rigid_index = shape_body[shape_index]

    px = particle_x[particle_index]
    radius = particle_radius[particle_index]

    X_wb = wp.transform_identity()
    if rigid_index >= 0:
        X_wb = body_X_wb[rigid_index]

    X_bs = shape_X_bs[shape_index]

    X_ws = wp.transform_multiply(X_wb, X_bs)
    X_sw = wp.transform_inverse(X_ws)

    # transform particle position to shape local space
    x_local = wp.transform_point(X_sw, px)

    # geo description
    geo_type = geo.type[shape_index]
    geo_scale = geo.scale[shape_index]
    geo_thickness = geo.thickness[shape_index]
    geo_filter = geo.face_filter[shape_index]
    particle_filters = geo.particle_filter_ids[shape_index]
    particle_filter = particle_filters[particle_index]

    # evaluate shape sdf
    d = 1.0e6
    n = wp.vec3()
    v = wp.vec3()

    if geo_type == wp.sim.GEO_SPHERE:
        d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)
        n = sphere_sdf_grad(wp.vec3(), geo_scale[0], x_local)

    if geo_type == wp.sim.GEO_BOX:
        d = box_sdf(geo_scale, x_local)
        n = box_sdf_grad(geo_scale, x_local)

    if geo_type == wp.sim.GEO_CAPSULE:
        d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)
        n = capsule_sdf_grad(geo_scale[0], geo_scale[1], x_local)

    if geo_type == wp.sim.GEO_CYLINDER:
        d = cylinder_sdf(geo_scale[0], geo_scale[1], x_local)
        n = cylinder_sdf_grad(geo_scale[0], geo_scale[1], x_local)

    if geo_type == wp.sim.GEO_CONE:
        d = cone_sdf(geo_scale[0], geo_scale[1], x_local)
        n = cone_sdf_grad(geo_scale[0], geo_scale[1], x_local)

    if geo_type == wp.sim.GEO_MESH:
        mesh = geo.source[shape_index]

        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)

        # if wp.mesh_query_point_sign_normal(
        #     mesh, wp.cw_div(x_local, geo_scale), margin + radius, sign, face_index, face_u, face_v
        # ):
        if wp.mesh_query_point_sign_normal(
                mesh, wp.cw_div(x_local, geo_scale), 10000.0, sign, face_index, face_u, face_v
        ):
            if not soft_collision_filter(geo_filter, particle_filter, face_index):
                shape_p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
                shape_v = wp.mesh_eval_velocity(mesh, face_index, face_u, face_v)

                shape_p = wp.cw_mul(shape_p, geo_scale)
                shape_v = wp.cw_mul(shape_v, geo_scale)

                delta = x_local - shape_p
                
                d = wp.length(delta) * sign
                n = wp.normalize(delta) * sign
                v = shape_v
    
    if geo_type == wp.sim.GEO_SDF:
        volume = geo.source[shape_index]
        xpred_local = wp.volume_world_to_index(volume, wp.cw_div(x_local, geo_scale))
        nn = wp.vec3(0.0, 0.0, 0.0)
        d = wp.volume_sample_grad_f(volume, xpred_local, wp.Volume.LINEAR, nn)
        n = wp.normalize(nn)        

    if geo_type == wp.sim.GEO_PLANE:
        d = plane_sdf(geo_scale[0], geo_scale[1], x_local)
        n = wp.vec3(0.0, 1.0, 0.0)

    d = d - geo_thickness   # Apply collision thickness 
    if d > - margin and d < margin + radius: #if d < margin + radius:
        index = wp.atomic_add(soft_contact_count, 0, 1)

        if index < soft_contact_max:
            # compute contact point in body local space
            body_pos = wp.transform_point(X_bs, x_local - n * d)
            body_vel = wp.transform_vector(X_bs, v)

            world_normal = wp.transform_vector(X_ws, n)

            soft_contact_shape[index] = shape_index
            soft_contact_body_pos[index] = body_pos
            soft_contact_body_vel[index] = body_vel
            soft_contact_particle[index] = particle_index
            soft_contact_normal[index] = world_normal

    if d < -margin:
        cloth_reference_drag_particles[particle_index] = 2


@wp.kernel
def count_contact_points(
    contact_pairs: wp.array(dtype=int, ndim=2),
    geo: ModelShapeGeometry,
    # outputs
    contact_count: wp.array(dtype=int),
):
    tid = wp.tid()
    shape_a = contact_pairs[tid, 0]
    shape_b = contact_pairs[tid, 1]

    if shape_b == -1:
        actual_type_a = geo.type[shape_a]
        # ground plane
        actual_type_b = wp.sim.GEO_PLANE
    else:
        type_a = geo.type[shape_a]
        type_b = geo.type[shape_b]
        # unique ordering of shape pairs
        if type_a < type_b:
            actual_shape_a = shape_a
            actual_shape_b = shape_b
            actual_type_a = type_a
            actual_type_b = type_b
        else:
            actual_shape_a = shape_b
            actual_shape_b = shape_a
            actual_type_a = type_b
            actual_type_b = type_a

    # determine how many contact points need to be evaluated
    num_contacts = 0
    if actual_type_a == wp.sim.GEO_SPHERE:
        num_contacts = 1
    elif actual_type_a == wp.sim.GEO_CAPSULE:
        if actual_type_b == wp.sim.GEO_PLANE:
            if geo.scale[actual_shape_b][0] == 0.0 and geo.scale[actual_shape_b][1] == 0.0:
                num_contacts = 2  # vertex-based collision for infinite plane
            else:
                num_contacts = 2 + 4  # vertex-based collision + plane edges
        elif actual_type_b == wp.sim.GEO_MESH:
            num_contacts_a = 2
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            num_contacts_b = mesh_b.points.shape[0]
            num_contacts = num_contacts_a + num_contacts_b
        else:
            num_contacts = 2
    elif actual_type_a == wp.sim.GEO_BOX:
        if actual_type_b == wp.sim.GEO_BOX:
            num_contacts = 24
        elif actual_type_b == wp.sim.GEO_MESH:
            num_contacts_a = 8
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            num_contacts_b = mesh_b.points.shape[0]
            num_contacts = num_contacts_a + num_contacts_b
        elif actual_type_b == wp.sim.GEO_PLANE:
            if geo.scale[actual_shape_b][0] == 0.0 and geo.scale[actual_shape_b][1] == 0.0:
                num_contacts = 8  # vertex-based collision
            else:
                num_contacts = 8 + 4  # vertex-based collision + plane edges
        else:
            num_contacts = 8
    elif actual_type_a == wp.sim.GEO_MESH:
        mesh_a = wp.mesh_get(geo.source[actual_shape_a])
        num_contacts_a = mesh_a.points.shape[0]
        if actual_type_b == wp.sim.GEO_MESH:
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            num_contacts_b = mesh_b.points.shape[0]
        else:
            num_contacts_b = 0
        num_contacts = num_contacts_a + num_contacts_b
    elif actual_type_a == wp.sim.GEO_PLANE:
        return  # no plane-plane contacts
    else:
        print("count_contact_points: unsupported geometry type")
        print(actual_type_a)
        print(actual_type_b)

    wp.atomic_add(contact_count, 0, num_contacts)


@wp.kernel
def broadphase_collision_pairs(
    contact_pairs: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    geo: ModelShapeGeometry,
    collision_radius: wp.array(dtype=float),
    rigid_contact_max: int,
    rigid_contact_margin: float,
    # outputs
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point_id: wp.array(dtype=int),
):
    tid = wp.tid()
    shape_a = contact_pairs[tid, 0]
    shape_b = contact_pairs[tid, 1]

    rigid_a = shape_body[shape_a]
    if rigid_a == -1:
        X_ws_a = shape_X_bs[shape_a]
    else:
        X_ws_a = wp.transform_multiply(body_q[rigid_a], shape_X_bs[shape_a])
    rigid_b = shape_body[shape_b]
    if rigid_b == -1:
        X_ws_b = shape_X_bs[shape_b]
    else:
        X_ws_b = wp.transform_multiply(body_q[rigid_b], shape_X_bs[shape_b])

    type_a = geo.type[shape_a]
    type_b = geo.type[shape_b]
    # unique ordering of shape pairs
    if type_a < type_b:
        actual_shape_a = shape_a
        actual_shape_b = shape_b
        actual_type_a = type_a
        actual_type_b = type_b
        actual_X_ws_a = X_ws_a
        actual_X_ws_b = X_ws_b
    else:
        actual_shape_a = shape_b
        actual_shape_b = shape_a
        actual_type_a = type_b
        actual_type_b = type_a
        actual_X_ws_a = X_ws_b
        actual_X_ws_b = X_ws_a

    p_a = wp.transform_get_translation(actual_X_ws_a)
    if actual_type_b == wp.sim.GEO_PLANE:
        if actual_type_a == wp.sim.GEO_PLANE:
            return
        query_b = wp.transform_point(wp.transform_inverse(actual_X_ws_b), p_a)
        scale = geo.scale[actual_shape_b]
        closest = closest_point_plane(scale[0], scale[1], query_b)
        d = wp.length(query_b - closest)
        r_a = collision_radius[actual_shape_a]
        if d > r_a + rigid_contact_margin:
            return
    else:
        p_b = wp.transform_get_translation(actual_X_ws_b)
        d = wp.length(p_a - p_b) * 0.5 - 0.1
        r_a = collision_radius[actual_shape_a]
        r_b = collision_radius[actual_shape_b]
        if d > r_a + r_b + rigid_contact_margin:
            return

    # determine how many contact points need to be evaluated
    num_contacts = 0
    if actual_type_a == wp.sim.GEO_SPHERE:
        num_contacts = 1
    elif actual_type_a == wp.sim.GEO_CAPSULE:
        if actual_type_b == wp.sim.GEO_PLANE:
            if geo.scale[actual_shape_b][0] == 0.0 and geo.scale[actual_shape_b][1] == 0.0:
                num_contacts = 2  # vertex-based collision for infinite plane
            else:
                num_contacts = 2 + 4  # vertex-based collision + plane edges
        elif actual_type_b == wp.sim.GEO_MESH:
            num_contacts_a = 2
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            num_contacts_b = mesh_b.points.shape[0]
            num_contacts = num_contacts_a + num_contacts_b
            index = wp.atomic_add(contact_count, 0, num_contacts)
            if index + num_contacts - 1 >= rigid_contact_max:
                print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")
                return
            # allocate contact points from capsule A against mesh B
            for i in range(num_contacts_a):
                contact_shape0[index + i] = actual_shape_a
                contact_shape1[index + i] = actual_shape_b
                contact_point_id[index + i] = i
            # allocate contact points from mesh B against capsule A
            for i in range(num_contacts_b):
                contact_shape0[index + num_contacts_a + i] = actual_shape_b
                contact_shape1[index + num_contacts_a + i] = actual_shape_a
                contact_point_id[index + num_contacts_a + i] = i
            return
        else:
            num_contacts = 2
    elif actual_type_a == wp.sim.GEO_BOX:
        if actual_type_b == wp.sim.GEO_BOX:
            index = wp.atomic_add(contact_count, 0, 24)
            if index + 23 >= rigid_contact_max:
                print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")
                return
            # allocate contact points from box A against B
            for i in range(12):  # 12 edges
                contact_shape0[index + i] = shape_a
                contact_shape1[index + i] = shape_b
                contact_point_id[index + i] = i
            # allocate contact points from box B against A
            for i in range(12):
                contact_shape0[index + 12 + i] = shape_b
                contact_shape1[index + 12 + i] = shape_a
                contact_point_id[index + 12 + i] = i
            return
        elif actual_type_b == wp.sim.GEO_MESH:
            num_contacts_a = 8
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            num_contacts_b = mesh_b.points.shape[0]
            num_contacts = num_contacts_a + num_contacts_b
            index = wp.atomic_add(contact_count, 0, num_contacts)
            if index + num_contacts - 1 >= rigid_contact_max:
                print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")
                return
            # allocate contact points from box A against mesh B
            for i in range(num_contacts_a):
                contact_shape0[index + i] = actual_shape_a
                contact_shape1[index + i] = actual_shape_b
                contact_point_id[index + i] = i
            # allocate contact points from mesh B against box A
            for i in range(num_contacts_b):
                contact_shape0[index + num_contacts_a + i] = actual_shape_b
                contact_shape1[index + num_contacts_a + i] = actual_shape_a
                contact_point_id[index + num_contacts_a + i] = i
            return
        elif actual_type_b == wp.sim.GEO_PLANE:
            if geo.scale[actual_shape_b][0] == 0.0 and geo.scale[actual_shape_b][1] == 0.0:
                num_contacts = 8  # vertex-based collision
            else:
                num_contacts = 8 + 4  # vertex-based collision + plane edges
        else:
            num_contacts = 8
    elif actual_type_a == wp.sim.GEO_MESH:
        mesh_a = wp.mesh_get(geo.source[actual_shape_a])
        num_contacts_a = mesh_a.points.shape[0]
        num_contacts_b = 0
        if actual_type_b == wp.sim.GEO_MESH:
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            num_contacts_b = mesh_b.points.shape[0]
        elif actual_type_b != wp.sim.GEO_PLANE:
            print("broadphase_collision_pairs: unsupported geometry type for mesh collision")
            return
        num_contacts = num_contacts_a + num_contacts_b
        if num_contacts > 0:
            index = wp.atomic_add(contact_count, 0, num_contacts)
            if index + num_contacts - 1 >= rigid_contact_max:
                print("Mesh contact: Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")
                return
            # allocate contact points from mesh A against B
            for i in range(num_contacts_a):
                contact_shape0[index + i] = actual_shape_a
                contact_shape1[index + i] = actual_shape_b
                contact_point_id[index + i] = i
            # allocate contact points from mesh B against A
            for i in range(num_contacts_b):
                contact_shape0[index + num_contacts_a + i] = actual_shape_b
                contact_shape1[index + num_contacts_a + i] = actual_shape_a
                contact_point_id[index + num_contacts_a + i] = i
        return
    elif actual_type_a == wp.sim.GEO_PLANE:
        return  # no plane-plane contacts
    else:
        print("broadphase_collision_pairs: unsupported geometry type")

    if num_contacts > 0:
        index = wp.atomic_add(contact_count, 0, num_contacts)
        if index + num_contacts - 1 >= rigid_contact_max:
            print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")
            return
        # allocate contact points
        for i in range(num_contacts):
            contact_shape0[index + i] = actual_shape_a
            contact_shape1[index + i] = actual_shape_b
            contact_point_id[index + i] = i


@wp.kernel
def handle_contact_pairs(
    body_q: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    geo: ModelShapeGeometry,
    rigid_contact_margin: float,
    body_com: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point_id: wp.array(dtype=int),
    rigid_contact_count: wp.array(dtype=int),
    edge_sdf_iter: int,
    # outputs
    contact_body0: wp.array(dtype=int),
    contact_body1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_thickness: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= rigid_contact_count[0]:
        return
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return

    point_id = contact_point_id[tid]

    rigid_a = shape_body[shape_a]
    X_wb_a = wp.transform_identity()
    if rigid_a >= 0:
        X_wb_a = body_q[rigid_a]
    X_bs_a = shape_X_bs[shape_a]
    X_ws_a = wp.transform_multiply(X_wb_a, X_bs_a)
    X_sw_a = wp.transform_inverse(X_ws_a)
    X_bw_a = wp.transform_inverse(X_wb_a)
    geo_type_a = geo.type[shape_a]
    geo_scale_a = geo.scale[shape_a]
    min_scale_a = min(geo_scale_a)
    thickness_a = geo.thickness[shape_a]
    # is_solid_a = geo.is_solid[shape_a]

    rigid_b = shape_body[shape_b]
    X_wb_b = wp.transform_identity()
    if rigid_b >= 0:
        X_wb_b = body_q[rigid_b]
    X_bs_b = shape_X_bs[shape_b]
    X_ws_b = wp.transform_multiply(X_wb_b, X_bs_b)
    X_sw_b = wp.transform_inverse(X_ws_b)
    X_bw_b = wp.transform_inverse(X_wb_b)
    geo_type_b = geo.type[shape_b]
    geo_scale_b = geo.scale[shape_b]
    min_scale_b = min(geo_scale_b)
    thickness_b = geo.thickness[shape_b]
    # is_solid_b = geo.is_solid[shape_b]

    # fill in contact rigid body ids
    contact_body0[tid] = rigid_a
    contact_body1[tid] = rigid_b

    distance = 1.0e6
    u = float(0.0)

    if geo_type_a == wp.sim.GEO_SPHERE:
        p_a_world = wp.transform_get_translation(X_ws_a)
        if geo_type_b == wp.sim.GEO_SPHERE:
            p_b_world = wp.transform_get_translation(X_ws_b)
        elif geo_type_b == wp.sim.GEO_BOX:
            # contact point in frame of body B
            p_a_body = wp.transform_point(X_sw_b, p_a_world)
            p_b_body = closest_point_box(geo_scale_b, p_a_body)
            p_b_world = wp.transform_point(X_ws_b, p_b_body)
        elif geo_type_b == wp.sim.GEO_CAPSULE:
            half_height_b = geo_scale_b[1]
            # capsule B
            A_b = wp.transform_point(X_ws_b, wp.vec3(0.0, half_height_b, 0.0))
            B_b = wp.transform_point(X_ws_b, wp.vec3(0.0, -half_height_b, 0.0))
            p_b_world = closest_point_line_segment(A_b, B_b, p_a_world)
        elif geo_type_b == wp.sim.GEO_MESH:
            mesh_b = geo.source[shape_b]
            query_b_local = wp.transform_point(X_sw_b, p_a_world)
            face_index = int(0)
            face_u = float(0.0)
            face_v = float(0.0)
            sign = float(0.0)
            max_dist = (thickness_a + thickness_b + rigid_contact_margin) / geo_scale_b[0]
            res = wp.mesh_query_point_sign_normal(
                mesh_b, wp.cw_div(query_b_local, geo_scale_b), max_dist, sign, face_index, face_u, face_v
            )
            if res:
                shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
                shape_p = wp.cw_mul(shape_p, geo_scale_b)
                p_b_world = wp.transform_point(X_ws_b, shape_p)
            else:
                contact_shape0[tid] = -1
                contact_shape1[tid] = -1
                return
        elif geo_type_b == wp.sim.GEO_PLANE:
            p_b_body = closest_point_plane(geo_scale_b[0], geo_scale_b[1], wp.transform_point(X_sw_b, p_a_world))
            p_b_world = wp.transform_point(X_ws_b, p_b_body)
        else:
            print("Unsupported geometry type in sphere collision handling")
            print(geo_type_b)
            return
        diff = p_a_world - p_b_world
        normal = wp.normalize(diff)
        distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_BOX and geo_type_b == wp.sim.GEO_BOX:
        # edge-based box contact
        edge = get_box_edge(point_id, geo_scale_a)
        edge0_world = wp.transform_point(X_ws_a, wp.spatial_top(edge))
        edge1_world = wp.transform_point(X_ws_a, wp.spatial_bottom(edge))
        edge0_b = wp.transform_point(X_sw_b, edge0_world)
        edge1_b = wp.transform_point(X_sw_b, edge1_world)
        max_iter = edge_sdf_iter
        u = closest_edge_coordinate_box(geo_scale_b, edge0_b, edge1_b, max_iter)
        p_a_world = (1.0 - u) * edge0_world + u * edge1_world

        # find closest point + contact normal on box B
        query_b = wp.transform_point(X_sw_b, p_a_world)
        p_b_body = closest_point_box(geo_scale_b, query_b)
        p_b_world = wp.transform_point(X_ws_b, p_b_body)
        diff = p_a_world - p_b_world
        # use center of box A to query normal to make sure we are not inside B
        query_b = wp.transform_point(X_sw_b, wp.transform_get_translation(X_ws_a))
        normal = wp.transform_vector(X_ws_b, box_sdf_grad(geo_scale_b, query_b))
        distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_BOX and geo_type_b == wp.sim.GEO_CAPSULE:
        half_height_b = geo_scale_b[1]
        # capsule B
        # depending on point id, we query an edge from 0 to 0.5 or 0.5 to 1
        e0 = wp.vec3(0.0, -half_height_b * float(point_id % 2), 0.0)
        e1 = wp.vec3(0.0, half_height_b * float((point_id + 1) % 2), 0.0)
        edge0_world = wp.transform_point(X_ws_b, e0)
        edge1_world = wp.transform_point(X_ws_b, e1)
        edge0_a = wp.transform_point(X_sw_a, edge0_world)
        edge1_a = wp.transform_point(X_sw_a, edge1_world)
        max_iter = edge_sdf_iter
        u = closest_edge_coordinate_box(geo_scale_a, edge0_a, edge1_a, max_iter)
        p_b_world = (1.0 - u) * edge0_world + u * edge1_world
        # find closest point + contact normal on box A
        query_a = wp.transform_point(X_sw_a, p_b_world)
        p_a_body = closest_point_box(geo_scale_a, query_a)
        p_a_world = wp.transform_point(X_ws_a, p_a_body)
        diff = p_a_world - p_b_world
        # the contact point inside the capsule should already be outside the box
        normal = -wp.transform_vector(X_ws_a, box_sdf_grad(geo_scale_a, query_a))
        distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_BOX and geo_type_b == wp.sim.GEO_PLANE:
        plane_width = geo_scale_b[0]
        plane_length = geo_scale_b[1]
        if point_id < 8:
            # vertex-based contact
            p_a_body = get_box_vertex(point_id, geo_scale_a)
            p_a_world = wp.transform_point(X_ws_a, p_a_body)
            query_b = wp.transform_point(X_sw_b, p_a_world)
            p_b_body = closest_point_plane(plane_width, plane_length, query_b)
            p_b_world = wp.transform_point(X_ws_b, p_b_body)
            diff = p_a_world - p_b_world
            normal = wp.transform_vector(X_ws_b, wp.vec3(0.0, 1.0, 0.0))
            if plane_width > 0.0 and plane_length > 0.0:
                if wp.abs(query_b[0]) > plane_width or wp.abs(query_b[2]) > plane_length:
                    # skip, we will evaluate the plane edge contact with the box later
                    contact_shape0[tid] = -1
                    contact_shape1[tid] = -1
                    return
                # check whether the COM is above the plane
                # sign = wp.sign(wp.dot(wp.transform_get_translation(X_ws_a) - p_b_world, normal))
                # if sign < 0.0:
                #     # the entire box is most likely below the plane
                #     contact_shape0[tid] = -1
                #     contact_shape1[tid] = -1
                #     return
            # the contact point is within plane boundaries
            distance = wp.dot(diff, normal)
        else:
            # contact between box A and edges of finite plane B
            edge = get_plane_edge(point_id - 8, plane_width, plane_length)
            edge0_world = wp.transform_point(X_ws_b, wp.spatial_top(edge))
            edge1_world = wp.transform_point(X_ws_b, wp.spatial_bottom(edge))
            edge0_a = wp.transform_point(X_sw_a, edge0_world)
            edge1_a = wp.transform_point(X_sw_a, edge1_world)
            max_iter = edge_sdf_iter
            u = closest_edge_coordinate_box(geo_scale_a, edge0_a, edge1_a, max_iter)
            p_b_world = (1.0 - u) * edge0_world + u * edge1_world

            # find closest point + contact normal on box A
            query_a = wp.transform_point(X_sw_a, p_b_world)
            p_a_body = closest_point_box(geo_scale_a, query_a)
            p_a_world = wp.transform_point(X_ws_a, p_a_body)
            query_b = wp.transform_point(X_sw_b, p_a_world)
            if wp.abs(query_b[0]) > plane_width or wp.abs(query_b[2]) > plane_length:
                # ensure that the closest point is actually inside the plane
                contact_shape0[tid] = -1
                contact_shape1[tid] = -1
                return
            diff = p_a_world - p_b_world
            com_a = wp.transform_get_translation(X_ws_a)
            query_b = wp.transform_point(X_sw_b, com_a)
            if wp.abs(query_b[0]) > plane_width or wp.abs(query_b[2]) > plane_length:
                # the COM is outside the plane
                normal = wp.normalize(com_a - p_b_world)
            else:
                normal = wp.transform_vector(X_ws_b, wp.vec3(0.0, 1.0, 0.0))
            distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_CAPSULE and geo_type_b == wp.sim.GEO_CAPSULE:
        # find closest edge coordinate to capsule SDF B
        half_height_a = geo_scale_a[1]
        half_height_b = geo_scale_b[1]
        # edge from capsule A
        # depending on point id, we query an edge from 0 to 0.5 or 0.5 to 1
        e0 = wp.vec3(0.0, half_height_a * float(point_id % 2), 0.0)
        e1 = wp.vec3(0.0, -half_height_a * float((point_id + 1) % 2), 0.0)
        edge0_world = wp.transform_point(X_ws_a, e0)
        edge1_world = wp.transform_point(X_ws_a, e1)
        edge0_b = wp.transform_point(X_sw_b, edge0_world)
        edge1_b = wp.transform_point(X_sw_b, edge1_world)
        max_iter = edge_sdf_iter
        u = closest_edge_coordinate_capsule(geo_scale_b[0], geo_scale_b[1], edge0_b, edge1_b, max_iter)
        p_a_world = (1.0 - u) * edge0_world + u * edge1_world
        p0_b_world = wp.transform_point(X_ws_b, wp.vec3(0.0, half_height_b, 0.0))
        p1_b_world = wp.transform_point(X_ws_b, wp.vec3(0.0, -half_height_b, 0.0))
        p_b_world = closest_point_line_segment(p0_b_world, p1_b_world, p_a_world)
        diff = p_a_world - p_b_world
        normal = wp.normalize(diff)
        distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_CAPSULE and geo_type_b == wp.sim.GEO_MESH:
        # find closest edge coordinate to mesh SDF B
        half_height_a = geo_scale_a[1]
        # edge from capsule A
        # depending on point id, we query an edge from -h to 0 or 0 to h
        e0 = wp.vec3(0.0, -half_height_a * float(point_id % 2), 0.0)
        e1 = wp.vec3(0.0, half_height_a * float((point_id + 1) % 2), 0.0)
        edge0_world = wp.transform_point(X_ws_a, e0)
        edge1_world = wp.transform_point(X_ws_a, e1)
        edge0_b = wp.transform_point(X_sw_b, edge0_world)
        edge1_b = wp.transform_point(X_sw_b, edge1_world)
        max_iter = edge_sdf_iter
        max_dist = (rigid_contact_margin + thickness_a + thickness_b) / min_scale_b
        mesh_b = geo.source[shape_b]
        u = closest_edge_coordinate_mesh(
            mesh_b, wp.cw_div(edge0_b, geo_scale_b), wp.cw_div(edge1_b, geo_scale_b), max_iter, max_dist
        )
        p_a_world = (1.0 - u) * edge0_world + u * edge1_world
        query_b_local = wp.transform_point(X_sw_b, p_a_world)
        mesh_b = geo.source[shape_b]

        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        res = wp.mesh_query_point_sign_normal(
            mesh_b, wp.cw_div(query_b_local, geo_scale_b), max_dist, sign, face_index, face_u, face_v
        )

        if res:
            shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
            shape_p = wp.cw_mul(shape_p, geo_scale_b)
            p_b_world = wp.transform_point(X_ws_b, shape_p)
            p_a_world = closest_point_line_segment(edge0_world, edge1_world, p_b_world)
            # contact direction vector in world frame
            diff = p_a_world - p_b_world
            normal = wp.normalize(diff)
            distance = wp.dot(diff, normal)
        else:
            contact_shape0[tid] = -1
            contact_shape1[tid] = -1
            return

    elif geo_type_a == wp.sim.GEO_MESH and geo_type_b == wp.sim.GEO_CAPSULE:
        # vertex-based contact
        mesh = wp.mesh_get(geo.source[shape_a])
        body_a_pos = wp.cw_mul(mesh.points[point_id], geo_scale_a)
        p_a_world = wp.transform_point(X_ws_a, body_a_pos)
        # find closest point + contact normal on capsule B
        half_height_b = geo_scale_b[1]
        A_b = wp.transform_point(X_ws_b, wp.vec3(0.0, half_height_b, 0.0))
        B_b = wp.transform_point(X_ws_b, wp.vec3(0.0, -half_height_b, 0.0))
        p_b_world = closest_point_line_segment(A_b, B_b, p_a_world)
        diff = p_a_world - p_b_world
        # this is more reliable in practice than using the SDF gradient
        normal = wp.normalize(diff)
        distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_CAPSULE and geo_type_b == wp.sim.GEO_PLANE:
        plane_width = geo_scale_b[0]
        plane_length = geo_scale_b[1]
        if point_id < 2:
            # vertex-based collision
            half_height_a = geo_scale_a[1]
            side = float(point_id) * 2.0 - 1.0
            p_a_world = wp.transform_point(X_ws_a, wp.vec3(0.0, side * half_height_a, 0.0))
            query_b = wp.transform_point(X_sw_b, p_a_world)
            p_b_body = closest_point_plane(geo_scale_b[0], geo_scale_b[1], query_b)
            p_b_world = wp.transform_point(X_ws_b, p_b_body)
            diff = p_a_world - p_b_world
            if geo_scale_b[0] > 0.0 and geo_scale_b[1] > 0.0:
                normal = wp.normalize(diff)
            else:
                normal = wp.transform_vector(X_ws_b, wp.vec3(0.0, 1.0, 0.0))
            distance = wp.dot(diff, normal)
        else:
            # contact between capsule A and edges of finite plane B
            plane_width = geo_scale_b[0]
            plane_length = geo_scale_b[1]
            edge = get_plane_edge(point_id - 2, plane_width, plane_length)
            edge0_world = wp.transform_point(X_ws_b, wp.spatial_top(edge))
            edge1_world = wp.transform_point(X_ws_b, wp.spatial_bottom(edge))
            edge0_a = wp.transform_point(X_sw_a, edge0_world)
            edge1_a = wp.transform_point(X_sw_a, edge1_world)
            max_iter = edge_sdf_iter
            u = closest_edge_coordinate_capsule(geo_scale_a[0], geo_scale_a[1], edge0_a, edge1_a, max_iter)
            p_b_world = (1.0 - u) * edge0_world + u * edge1_world

            # find closest point + contact normal on capsule A
            half_height_a = geo_scale_a[1]
            p0_a_world = wp.transform_point(X_ws_a, wp.vec3(0.0, half_height_a, 0.0))
            p1_a_world = wp.transform_point(X_ws_a, wp.vec3(0.0, -half_height_a, 0.0))
            p_a_world = closest_point_line_segment(p0_a_world, p1_a_world, p_b_world)
            diff = p_a_world - p_b_world
            # normal = wp.transform_vector(X_ws_b, wp.vec3(0.0, 1.0, 0.0))
            normal = wp.normalize(diff)
            distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_MESH and geo_type_b == wp.sim.GEO_BOX:
        # vertex-based contact
        mesh = wp.mesh_get(geo.source[shape_a])
        body_a_pos = wp.cw_mul(mesh.points[point_id], geo_scale_a)
        p_a_world = wp.transform_point(X_ws_a, body_a_pos)
        # find closest point + contact normal on box B
        query_b = wp.transform_point(X_sw_b, p_a_world)
        p_b_body = closest_point_box(geo_scale_b, query_b)
        p_b_world = wp.transform_point(X_ws_b, p_b_body)
        diff = p_a_world - p_b_world
        # this is more reliable in practice than using the SDF gradient
        normal = wp.normalize(diff)
        if box_sdf(geo_scale_b, query_b) < 0.0:
            normal = -normal
        distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_BOX and geo_type_b == wp.sim.GEO_MESH:
        # vertex-based contact
        query_a = get_box_vertex(point_id, geo_scale_a)
        p_a_world = wp.transform_point(X_ws_a, query_a)
        query_b_local = wp.transform_point(X_sw_b, p_a_world)
        mesh_b = geo.source[shape_b]
        max_dist = (rigid_contact_margin + thickness_a + thickness_b) / min_scale_b
        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        res = wp.mesh_query_point_sign_normal(
            mesh_b, wp.cw_div(query_b_local, geo_scale_b), max_dist, sign, face_index, face_u, face_v
        )

        if res:
            shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
            shape_p = wp.cw_mul(shape_p, geo_scale_b)
            p_b_world = wp.transform_point(X_ws_b, shape_p)
            # contact direction vector in world frame
            diff_b = p_a_world - p_b_world
            normal = wp.normalize(diff_b) * sign
            distance = wp.dot(diff_b, normal)
        else:
            contact_shape0[tid] = -1
            contact_shape1[tid] = -1
            return

    elif geo_type_a == wp.sim.GEO_MESH and geo_type_b == wp.sim.GEO_MESH:
        # vertex-based contact
        mesh = wp.mesh_get(geo.source[shape_a])
        mesh_b = geo.source[shape_b]

        body_a_pos = wp.cw_mul(mesh.points[point_id], geo_scale_a)
        p_a_world = wp.transform_point(X_ws_a, body_a_pos)
        query_b_local = wp.transform_point(X_sw_b, p_a_world)

        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        min_scale = min(min_scale_a, min_scale_b)
        max_dist = (rigid_contact_margin + thickness_a + thickness_b) / min_scale

        res = wp.mesh_query_point_sign_normal(
            mesh_b, wp.cw_div(query_b_local, geo_scale_b), max_dist, sign, face_index, face_u, face_v
        )

        if res:
            shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
            shape_p = wp.cw_mul(shape_p, geo_scale_b)
            p_b_world = wp.transform_point(X_ws_b, shape_p)
            # contact direction vector in world frame
            diff_b = p_a_world - p_b_world
            normal = wp.normalize(diff_b) * sign
            distance = wp.dot(diff_b, normal)
        else:
            contact_shape0[tid] = -1
            contact_shape1[tid] = -1
            return

    elif geo_type_a == wp.sim.GEO_MESH and geo_type_b == wp.sim.GEO_PLANE:
        # vertex-based contact
        mesh = wp.mesh_get(geo.source[shape_a])
        body_a_pos = wp.cw_mul(mesh.points[point_id], geo_scale_a)
        p_a_world = wp.transform_point(X_ws_a, body_a_pos)
        query_b = wp.transform_point(X_sw_b, p_a_world)
        p_b_body = closest_point_plane(geo_scale_b[0], geo_scale_b[1], query_b)
        p_b_world = wp.transform_point(X_ws_b, p_b_body)
        diff = p_a_world - p_b_world
        normal = wp.transform_vector(X_ws_b, wp.vec3(0.0, 1.0, 0.0))
        distance = wp.length(diff)

        # if the plane is infinite or the point is within the plane we fix the normal to prevent intersections
        if (
            geo_scale_b[0] == 0.0
            and geo_scale_b[1] == 0.0
            or wp.abs(query_b[0]) < geo_scale_b[0]
            and wp.abs(query_b[2]) < geo_scale_b[1]
        ):
            normal = wp.transform_vector(X_ws_b, wp.vec3(0.0, 1.0, 0.0))
        else:
            normal = wp.normalize(diff)
        distance = wp.dot(diff, normal)
        # ignore extreme penetrations (e.g. when mesh is below the plane)
        if distance < -rigid_contact_margin:
            contact_shape0[tid] = -1
            contact_shape1[tid] = -1
            return

    else:
        print("Unsupported geometry pair in collision handling")
        return

    thickness = thickness_a + thickness_b
    d = distance - thickness
    if d < rigid_contact_margin:
        # transform from world into body frame (so the contact point includes the shape transform)
        contact_point0[tid] = wp.transform_point(X_bw_a, p_a_world)
        contact_point1[tid] = wp.transform_point(X_bw_b, p_b_world)
        contact_offset0[tid] = wp.transform_vector(X_bw_a, -thickness_a * normal)
        contact_offset1[tid] = wp.transform_vector(X_bw_b, thickness_b * normal)
        contact_normal[tid] = normal
        contact_thickness[tid] = thickness
        # wp.printf("distance: %f\tnormal: %.3f %.3f %.3f\tp_a_world: %.3f %.3f %.3f\tp_b_world: %.3f %.3f %.3f\n", distance, normal[0], normal[1], normal[2], p_a_world[0], p_a_world[1], p_a_world[2], p_b_world[0], p_b_world[1], p_b_world[2])
    else:
        contact_shape0[tid] = -1
        contact_shape1[tid] = -1


@wp.func
def closest_point_on_reference_shape(
        px: wp.vec3,
        shape_index: int,
        reference_shapes: wp.array(dtype=wp.uint64),
        reference_transform: wp.array(dtype=wp.transform),
        reference_scale: wp.array(dtype=wp.vec3),
):
    
    X_ws = reference_transform[shape_index]
    X_sw = wp.transform_inverse(X_ws)
    # transform particle position to shape local space
    x_local = wp.transform_point(X_sw, px)
    # geo description
    geo_scale = reference_scale[shape_index]
    # evaluate shape sdf
    mesh = reference_shapes[shape_index]
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    if wp.mesh_query_point(mesh, wp.cw_div(x_local, geo_scale), 10000.0, sign, face_index, face_u, face_v):
        shape_p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
        shape_p = wp.cw_mul(shape_p, geo_scale)
        shape_p = wp.transform_point(X_ws, shape_p)

        return shape_p
    return wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def find_intersecting_particles(
        edge_indices: wp.array(dtype=int),
        particle_shape: wp.uint64,
        particle_reference_label: wp.array(dtype=int),
        drag_label_pairs: wp.array(dtype=wp.vec2i),  
        drag_label_pairs_count: int,
        # outputs
        intersecting_particles: wp.array(dtype=int),
        drag_particles: wp.array(dtype=int),
        self_intersection_count: wp.array(dtype=int),
):
    """
    Dim: edge_count
    inputs = [
        model.edge_indices,
        state.particle_q,
        state.particle_qd,
        model.particle_radius,
        model.particle_flags,
        model.particle_shape,
        model.cloth_reference_margin,
    ],
    outputs=[
        model.intersecting_particles,
    ],
    """
    tid = wp.tid()
    idx_k = edge_indices[tid * 2 + 0]
    idx_l = edge_indices[tid * 2 + 1]
    
    mesh = wp.mesh_get(particle_shape)
    t = float(0.)
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()

    # # ray from k to l
    if wp.mesh_query_edge(particle_shape, idx_k, idx_l, t, face_u, face_v, sign, normal, face_index):
        f1 = mesh.indices[face_index * 3 + 0]
        f2 = mesh.indices[face_index * 3 + 1]
        f3 = mesh.indices[face_index * 3 + 2]

        # particle self intersection
        wp.atomic_add(self_intersection_count, 0, 1)

        # Add self-intersection to filter
        intersecting_particles[idx_l] = 1
        intersecting_particles[idx_k] = 1

        # --- Particle dragging ---
        # Intersecting particles are subject to dragging when they belong to different panels
        face_label = -1
        if particle_reference_label[f1] != -1:
            face_label = particle_reference_label[f1]
        elif particle_reference_label[f2] != -1:
            face_label = particle_reference_label[f2]
        elif particle_reference_label[f3] != -1:
            face_label = particle_reference_label[f3]

        edge_label = -1
        if particle_reference_label[idx_k] != -1:
            edge_label = particle_reference_label[idx_k]
        elif particle_reference_label[idx_l] != -1:
            edge_label = particle_reference_label[idx_l]

        # NOTE: Don't consider non-segmented sections
        if face_label != -1 and face_label != edge_label: 
            curr_pair, curr_pair_swap = wp.vec2i(face_label, edge_label), wp.vec2i(edge_label, face_label)
            is_drag_pair = int(0)
            for i in range(drag_label_pairs_count):
                pair = drag_label_pairs[i]
                if curr_pair == pair or curr_pair_swap == pair:
                    is_drag_pair = 1
                    break
            if is_drag_pair:
                drag_particles[idx_l] = 1
                drag_particles[idx_k] = 1


@wp.kernel
def count_self_intersections(
    edge_indices: wp.array(dtype=int),
    particle_shape: wp.uint64,
    # Outputs
    self_intersection_count: wp.array(dtype=int),
):
    tid = wp.tid()
    e_0idx = edge_indices[tid * 2 + 0]
    e_1idx = edge_indices[tid * 2 + 1]

    # Mesh-edge intersection
    t = float(0.)
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()

    if wp.mesh_query_edge(particle_shape, e_0idx, e_1idx, t, face_u, face_v, sign, normal, face_index):
        wp.atomic_add(self_intersection_count, 0, 1)


@wp.kernel
def count_body_cloth_intersections(
    edge_indices: wp.array(dtype=int),
    cloth_particle_shape: wp.uint64,
    geo: ModelShapeGeometry,
    body_shape_idx: int,
    # Outputs
    body_cloth_intersection_count: wp.array(dtype=int),
):
    tid = wp.tid()
    body_mesh_id = geo.source[body_shape_idx]
    e_0idx = edge_indices[tid * 2 + 0]
    e_1idx = edge_indices[tid * 2 + 1]

    # Mesh-edge intersection
    t = float(0.)
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()

    # Cloth mesh
    mesh = wp.mesh_get(cloth_particle_shape)
    o = mesh.points[e_0idx]
    d = mesh.points[e_1idx] - o  

    max_t = wp.length(d)  # == edge length
    d = wp.normalize(d)

    if wp.mesh_query_ray(body_mesh_id, o, d, max_t, t, face_u, face_v, sign, normal, face_index):
        wp.atomic_add(body_cloth_intersection_count, 0, 1)

    # DRAFT -- compilation errors?
    # out = wp.mesh_query_ray(body_mesh, o, d, max_t)
    # if out.result:
    #     wp.atomic_add(body_cloth_intersection_count, 0, 1)


@wp.func
def check_edge_exists(
    e0_idx: wp.int32,  
    e1_idx: wp.int32,
    edge_contact_pairs: wp.array(dtype=wp.vec4i),
    start: wp.int32,
    end: wp.int32
):
    """Check if edge is already present in the given section of edge pairs
        (as a second edge in pair)
    """
    for i in range(start, end):
        pair = edge_contact_pairs[i]
        if pair[2] == e0_idx and pair[3] == e1_idx:
            return True
    return False

@wp.func
def create_edge_pair_contact(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    p_a_idx: wp.int32,  # current edge
    p_b_idx: wp.int32,
    e0_idx: wp.int32,   # colliding with
    e1_idx: wp.int32,
    radius: float,
    gravity: wp.vec3,
    dt: float
):
    """Create a contact info for a given pair of edges"""
    # Parameters  # TODO External
    scr = radius * 0.5

    # Info
    p_a = particle_x[p_a_idx]
    v_a = particle_v[p_a_idx]
    w_a = particle_invmass[p_a_idx]

    p_b = particle_x[p_b_idx]
    v_b = particle_v[p_b_idx]
    w_b = particle_invmass[p_b_idx]

    p_a_next = p_a + v_a * dt + w_a * dt * dt * gravity
    p_b_next = p_b + v_b * dt + w_b * dt * dt * gravity

    e_0 = particle_x[e0_idx]
    v_0 = particle_v[e0_idx]
    w_0 = particle_invmass[e0_idx]

    e_1 = particle_x[e1_idx]
    v_1 = particle_v[e1_idx]
    w_1 = particle_invmass[e1_idx]

    # Next positions
    e0_next = e_0 + v_0 * dt + w_0 * dt * dt * gravity
    e1_next = e_1 + v_1 * dt + w_1 * dt * dt * gravity

    # Closest points
    out = wp.closest_point_edge_edge(p_a, p_b, e_0, e_1, 0.001)
    t_alpha, t_beta, dist = out[0], out[1], out[2]

    # If the intersection is too close to an endpoint, it's 
    # point-triangle collision, not edge-edge
    ab_length = wp.length(p_b - p_a)
    if (t_alpha * ab_length < radius or (1. - t_alpha) * ab_length < radius
            or t_beta * ab_length < radius or (1. - t_beta) * ab_length < radius
        ):
        return 0., 0., wp.vec3(0.)
    else:
        p_alpha = p_a * (1. - t_alpha) + p_b * t_alpha 
        p_beta = e_0 * (1. - t_beta) + e_1 * t_beta 
        lDir = wp.normalize(p_beta - p_alpha)  

        # TODOLOW More optimal arrangement of computations
        p_alpha_next = p_a_next * (1. - t_alpha) + p_b_next * t_alpha 
        p_beta_next = e0_next * (1. - t_beta) + e1_next * t_beta 

        d_alpha_proj = wp.dot((p_alpha_next - p_alpha), lDir)
        d_beta_proj = wp.dot((p_beta_next - p_beta), lDir)

        dist_after = dist - d_alpha_proj + d_beta_proj  

        if dist_after < 2. * radius + scr:  # Create contact only if comes close!

            return t_alpha, t_beta, lDir
        else:
            return 0., 0., wp.vec3(0.)

@wp.kernel
def create_self_edge_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_shape: wp.uint64,  
    intersecting_particles: wp.array(dtype=int),
    edge_indices: wp.array(dtype=int),
    edge_contact_max: int,
    gravity: wp.vec3,
    dt: float,
    # outputs
    edge_contact_count: wp.array(dtype=int),
    edge_contact_pairs: wp.array(dtype=wp.vec4i),
    edge_contact_filter: wp.array(dtype=bool),  
    edge_contact_normal: wp.array(dtype=wp.vec3)
):
    """Find edge-edge contact pairs"""
    # Current edge
    tid = wp.tid()
    p_a_idx = edge_indices[tid * 2 + 0]
    p_b_idx = edge_indices[tid * 2 + 1]

    if (intersecting_particles[p_a_idx] == 1 or intersecting_particles[p_b_idx] == 1):
        # Don't add self contacts for particles that are already intersecting
        # Allows not to force preserve the existing collisions
        return

    p_a = particle_x[p_a_idx]
    p_b = particle_x[p_b_idx]

    mid_p = (p_a + p_b) / 2.
    edge_len = wp.length(p_a - p_b)

    # Parameters
    radius = max(particle_radius[p_a_idx], particle_radius[p_b_idx])  
    max_distance = edge_len / 2. + radius * 3.

    # Iterate over closes faces
    mesh = wp.mesh_get(particle_shape)
    dist_vec = wp.vec3(max_distance, max_distance, max_distance)
    lower = mid_p - dist_vec
    upper = mid_p + dist_vec
    query = wp.mesh_query_aabb(particle_shape, lower, upper)
    face_index = int(0)

    # TODO Store only actually existing contacts, 
    # not all possibilities (+ filter)

    # Count max contacts
    max_count = int(0)
    while wp.mesh_query_aabb_next(query, face_index):
        max_count += 1
    max_count *= 3  # 3 edges in each face

    if max_count < 1:  # No contacts created # NOTE: unlikely scenario..
        return
    
    # Create contacts
    index = wp.atomic_add(edge_contact_count, 0, max_count)
    if index + max_count - 1 >= edge_contact_max:
        n_edges = float(edge_contact_max) / 3. / 250.
        printf("\n Number of edge-edge contacts (%d) exceeded limit (%d). Increase Model.edge_contact_max.", 
               int(float(index + max_count - 1) / 3. / n_edges), 
               int(float(edge_contact_max) / 3. / n_edges))
        return
    
    f_contact_index = int(0)
    query = wp.mesh_query_aabb(particle_shape, lower, upper)
    face_index = int(0)

    while wp.mesh_query_aabb_next(query, face_index):
        p0_idx = mesh.indices[face_index * 3 + 0]
        p1_idx = mesh.indices[face_index * 3 + 1]
        p2_idx = mesh.indices[face_index * 3 + 2]

        # Edge 0
        # NOTE: Edge existance checks with reversed order as the edges are traversed reveresly in the
        # neighbouring triangles
        if (p0_idx == p_a_idx or p0_idx == p_b_idx or p1_idx == p_a_idx or p1_idx == p_b_idx
                or check_edge_exists(p1_idx, p0_idx, edge_contact_pairs, index, index + f_contact_index)
            ):
            edge_contact_filter[index + f_contact_index + 0] = True
        else:
            p_alpha, p_beta, norm = create_edge_pair_contact(
                particle_x, 
                particle_v,
                particle_invmass,
                p_a_idx, 
                p_b_idx,
                p0_idx,
                p1_idx,
                radius,
                gravity,
                dt
            )
            if p_alpha < 1e-5:
                edge_contact_filter[index + f_contact_index + 0] = True
            else:
                edge_contact_normal[index + f_contact_index + 0] = norm
                edge_contact_pairs[index + f_contact_index + 0] = wp.vec4i(p_a_idx, p_b_idx, p0_idx, p1_idx)
        
        # Edge 1
        if (p2_idx == p_a_idx or p2_idx == p_b_idx or p1_idx == p_a_idx or p1_idx == p_b_idx
                or check_edge_exists(p2_idx, p1_idx, edge_contact_pairs, index, index + f_contact_index)
            ):
            edge_contact_filter[index + f_contact_index + 1] = True
        else:
            p_alpha, p_beta, norm = create_edge_pair_contact(
                particle_x, 
                particle_v,
                particle_invmass,
                p_a_idx, 
                p_b_idx,
                p1_idx,
                p2_idx,
                radius,
                gravity,
                dt
            )
            if p_alpha < 1e-5:
                edge_contact_filter[index + f_contact_index + 1] = True
            else:
                edge_contact_normal[index + f_contact_index + 1] = norm
                edge_contact_pairs[index + f_contact_index + 1] = wp.vec4i(p_a_idx, p_b_idx, p1_idx, p2_idx)
        
        # Edge 2
        if (p2_idx == p_a_idx or p2_idx == p_b_idx or p0_idx == p_a_idx or p0_idx == p_b_idx
                or check_edge_exists(p0_idx, p2_idx, edge_contact_pairs, index, index + f_contact_index)
            ):
            edge_contact_filter[index + f_contact_index + 2] = True
        else:
            p_alpha, p_beta, norm = create_edge_pair_contact(
                particle_x, 
                particle_v,
                particle_invmass,
                p_a_idx, 
                p_b_idx,
                p2_idx,
                p0_idx,
                radius,
                gravity,
                dt
            )  
            if p_alpha < 1e-5:
                edge_contact_filter[index + f_contact_index + 2] = True
            else:
                edge_contact_normal[index + f_contact_index + 2] = norm
                edge_contact_pairs[index + f_contact_index + 2] = wp.vec4i(p_a_idx, p_b_idx, p2_idx, p0_idx)

        f_contact_index += 3

@wp.kernel
def create_self_point_triangle_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_shape: wp.uint64,  
    intersecting_particles: wp.array(dtype=int),
    point_tri_contact_max: int,
    gravity: wp.vec3,
    dt: float,
    # outputs
    point_tri_contact_count: wp.array(dtype=int),
    point_tri_contact_pairs: wp.array(dtype=wp.vec2i),
    point_tri_contact_filter: wp.array(dtype=bool),  
    point_tri_contact_sidedness: wp.array(dtype=bool)
):
    """Collect point-triangle contact pairs: pair information and the side of the point w.r.t. triangle norm"""
    # Current particle
    tid = wp.tid()
    particle_index = tid

    if intersecting_particles[particle_index] == 1:
        # Don't add self contacts for particles that are already intersecting
        # Allows not to force preserve the existing collisions
        return

    p = particle_x[particle_index]
    vp = particle_v[particle_index]
    wpoint = particle_invmass[particle_index]
    radius = particle_radius[particle_index]  # NOTE == Thickness? 

    mesh = wp.mesh_get(particle_shape)

    # Parameters
    # TODO from outside
    max_distance = radius * 3.  #  NOTE: Following the recommendation from here: https://carmencincotti.com/2022-11-21/cloth-self-collisions/#the-instability-problem
    scr = radius * 0.5

    dist_vec = wp.vec3(max_distance, max_distance, max_distance)
    lower = p - dist_vec
    upper = p + dist_vec

    # Eval number of contacts
    query = wp.mesh_query_aabb(particle_shape, lower, upper)
    face_index = int(0)
    max_count = int(0)
    while wp.mesh_query_aabb_next(query, face_index):
        max_count += 1

    if max_count < 1:
        return  # no contacts (unlikely)
    
    # Create space for new contacts
    index = wp.atomic_add(point_tri_contact_count, 0, max_count)
    if index + max_count - 1 >= point_tri_contact_max:
        printf("\n Number of particle-triangle contacts (%d) exceeded limit (%d). Increase Model.point_tri_contact_max.", 
               int(float(index + max_count - 1) / float(particle_x.shape[0])), 
               int(float(point_tri_contact_max) / float(particle_x.shape[0])))
        return

    # Record contacts
    query = wp.mesh_query_aabb(particle_shape, lower, upper)
    face_index = int(0)
    f_contact_index = int(0)
    while wp.mesh_query_aabb_next(query, face_index):
        p0_idx = mesh.indices[face_index * 3 + 0]
        p1_idx = mesh.indices[face_index * 3 + 1]
        p2_idx = mesh.indices[face_index * 3 + 2]

        # Don't include its own face
        if p0_idx == tid or p1_idx == tid or p2_idx == tid:
            point_tri_contact_filter[index + f_contact_index] = True
        else: # Potential collision
            # Evaluate contact sidedness
            p0 = particle_x[p0_idx] 
            p1 = particle_x[p1_idx] 
            p2 = particle_x[p2_idx] 

            v0 = particle_v[p0_idx]
            v1 = particle_v[p1_idx]
            v2 = particle_v[p2_idx]

            w0 = particle_invmass[p0_idx]
            w1 = particle_invmass[p1_idx]
            w2 = particle_invmass[p2_idx]

            # Closest point
            bary_closest = triangle_closest_point_barycentric(p0, p1, p2, p)
            baryAlpha, baryBeta, baryGamma = bary_closest[0], bary_closest[1], bary_closest[2]
            pC = baryAlpha * p0 + baryBeta * p1 + baryGamma * p2

            # Check contact potential 
            # Project points: apply velocity
            # TODO Other external forces
            projP = p + vp * dt + wpoint * dt * dt * gravity
            projP0 = p0 + v0 * dt + w0 * dt * dt * gravity
            projP1 = p1 + v1 * dt + w1 * dt * dt * gravity
            projP2 = p2 + v2 * dt + w2 * dt * dt * gravity

            d = projP - p
            d0 = projP0 - p0
            d1 = projP1 - p1
            d2 = projP2 - p2
            dC = (d0 * baryAlpha) + (d1 * baryBeta) + (d2 * baryGamma)  
            lDir = wp.normalize(pC - p)

            dcProj = wp.dot(dC, lDir)
            dProj = wp.dot(d, lDir)

            distC = wp.length(pC - p)
            distC_after = distC - dProj + dcProj 

            if distC_after < 2. * radius + scr:
                # Evaluate sidedness
                normal = wp.cross(p1 - p0, p2 - p0) 
                n_hat = wp.normalize(normal) 

                # Asuming the current point-triangle relation is the correct one
                cosTheta = wp.dot(n_hat, p - p0)
        
                # cosTheta < 0 indicates that the triangle norm should be flipped 
                point_tri_contact_sidedness[index + f_contact_index] = (cosTheta < 0)  
                point_tri_contact_pairs[index + f_contact_index] = wp.vec2i(particle_index, face_index)

            else:
                point_tri_contact_filter[index + f_contact_index] = True
            
        f_contact_index += 1


@wp.kernel
def count_non_zero(
        array: wp.array(dtype=int),
        count: wp.array(dtype=int),
):
    tid = wp.tid()
    if array[tid] != 0:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def print_count(
        count: wp.array(dtype=int),
):
    wp.printf("Found %d intersecting particles\n", count[0])


def collide(model, state, dt, edge_sdf_iter: int = 10):
    """
    Generates contact points for the particles and rigid bodies in the model,
    to be used in the contact dynamics kernel of the integrator.

    Args:
        model: the model to be simulated
        state: the state of the model
        edge_sdf_iter: number of search iterations for finding closest contact points between edges and SDF
    """
    if model.particle_count and (model.cloth_reference_drag or model.global_collision_filter):
        model.cloth_reference_drag_particles.zero_()
        model.particle_self_intersection_particle.zero_()

    # generate soft contacts for particles and shapes except ground plane (last shape)
    if model.particle_count and model.shape_count > 1:
        # clear old count
        model.soft_contact_count.zero_()
        wp.launch(
            kernel=create_soft_contacts,
            dim=(model.particle_count, model.shape_count - 1),
            inputs=[
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_geo,
                model.soft_contact_margin,
                model.soft_contact_max,
            ],
            outputs=[
                model.soft_contact_count,
                model.soft_contact_particle,
                model.soft_contact_shape,
                model.soft_contact_body_pos,
                model.soft_contact_body_vel,
                model.soft_contact_normal,
                model.cloth_reference_drag_particles

            ],
            device=model.device,
        )

    # clear old count
    model.rigid_contact_count.zero_()

    if model.shape_contact_pair_count:
        wp.launch(
            kernel=broadphase_collision_pairs,
            dim=model.shape_contact_pair_count,
            inputs=[
                model.shape_contact_pairs,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_geo,
                model.shape_collision_radius,
                model.rigid_contact_max,
                model.rigid_contact_margin,
            ],
            outputs=[
                model.rigid_contact_count,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                model.rigid_contact_point_id,
            ],
            device=model.device,
            record_tape=False,
        )

    if model.ground and model.shape_ground_contact_pair_count:
        wp.launch(
            kernel=broadphase_collision_pairs,
            dim=model.shape_ground_contact_pair_count,
            inputs=[
                model.shape_ground_contact_pairs,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_geo,
                model.shape_collision_radius,
                model.rigid_contact_max,
                model.rigid_contact_margin,
            ],
            outputs=[
                model.rigid_contact_count,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                model.rigid_contact_point_id,
            ],
            device=model.device,
            record_tape=False,
        )

    if model.shape_contact_pair_count or model.ground and model.shape_ground_contact_pair_count:
        wp.launch(
            kernel=handle_contact_pairs,
            dim=model.rigid_contact_max,
            inputs=[
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_geo,
                model.rigid_contact_margin,
                model.body_com,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                model.rigid_contact_point_id,
                model.rigid_contact_count,
                edge_sdf_iter,
            ],
            outputs=[
                model.rigid_contact_body0,
                model.rigid_contact_body1,
                model.rigid_contact_point0,
                model.rigid_contact_point1,
                model.rigid_contact_offset0,
                model.rigid_contact_offset1,
                model.rigid_contact_normal,
                model.rigid_contact_thickness,
            ],
            device=model.device,
        )

    if model.particle_count and model.global_collision_filter:
        model.particle_global_self_intersection_count.zero_()
        wp.launch(
            kernel=find_intersecting_particles,
            dim=model.spring_count,
            inputs = [
                model.spring_indices,
                model.particle_shape.id,
                model.particle_reference_label,
                model.drag_label_pairs,
                model.drag_label_pairs_count
            ],
            outputs=[
                model.particle_self_intersection_particle,
                model.cloth_reference_drag_particles,
                model.particle_global_self_intersection_count,
            ],
            device=model.device,
        )

    # NOTE: After computing global intersection
    if model.particle_count and model.spring_count:
        model.edge_contact_count.zero_()
        model.edge_contact_filter.zero_()
        model.edge_contact_pairs.zero_()
        model.edge_contact_normal.zero_()

        wp.launch(
            kernel=create_self_edge_contacts,
            dim=model.spring_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.particle_inv_mass,
                model.particle_radius,
                model.particle_shape.id,
                model.particle_self_intersection_particle,  # TODO Under the cloth_drag condition!
                model.spring_indices,
                model.edge_contact_max,
                model.gravity,
                dt
            ],
            outputs=[
                model.edge_contact_count,
                model.edge_contact_pairs,
                model.edge_contact_filter,  
                model.edge_contact_normal
            ]
        )
    
    if model.particle_count:
        model.point_tri_contact_count.zero_()
        model.point_tri_contact_filter.zero_()
        model.point_tri_contact_sidedness.zero_()
        wp.launch(
            kernel=create_self_point_triangle_contacts,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.particle_inv_mass,
                model.particle_radius,
                model.particle_shape.id,
                model.particle_self_intersection_particle,
                model.point_tri_contact_max,
                model.gravity,
                dt
            ],
            outputs=[
                model.point_tri_contact_count,
                model.point_tri_contact_pairs,
                model.point_tri_contact_filter,  
                model.point_tri_contact_sidedness
            ]
        )