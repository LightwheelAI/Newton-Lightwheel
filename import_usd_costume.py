
from __future__ import annotations

import datetime
import os
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import warp as wp

from newton.utils import quat_between_axes
from newton import Axis
from newton import Mesh, ShapeFlags
from newton import ModelBuilder
from newton import JointMode
from collections import defaultdict, deque

try:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415
except ImportError as e:
    raise ImportError("Failed to import pxr. Please install USD (e.g. via `pip install usd-core`).") from e

def get_attribute(prim, name):
    return prim.GetAttribute(name)

def has_attribute(prim, name):
    attr = get_attribute(prim, name)
    return attr.IsValid() and attr.HasAuthoredValue()

def parse_float(prim, name, default=None):
    attr = get_attribute(prim, name)
    if not attr or not attr.HasAuthoredValue():
        return default
    val = attr.Get()
    if np.isfinite(val):
        return val
    return default

def parse_float_with_fallback(prims: Iterable[Usd.Prim], name: str, default: float = 0.0) -> float:
    ret = default
    for prim in prims:
        if not prim:
            continue
        attr = get_attribute(prim, name)
        if not attr or not attr.HasAuthoredValue():
            continue
        val = attr.Get()
        if np.isfinite(val):
            ret = val
            break
    return ret

def from_gfquat(gfquat):
    return wp.normalize(wp.quat(*gfquat.imaginary, gfquat.real))

def parse_quat(prim, name, default=None):
    attr = get_attribute(prim, name)
    if not attr or not attr.HasAuthoredValue():
        return default
    val = attr.Get()
    quat = from_gfquat(val)
    l = wp.length(quat)
    if np.isfinite(l) and l > 0.0:
        return quat
    return default

def parse_vec(prim, name, default=None):
    attr = get_attribute(prim, name)
    if not attr or not attr.HasAuthoredValue():
        return default
    val = attr.Get()
    if np.isfinite(val).all():
        return np.array(val, dtype=np.float32)
    return default

def parse_generic(prim, name, default=None):
    attr = get_attribute(prim, name)
    if not attr or not attr.HasAuthoredValue():
        return default
    return attr.Get()

def parse_xform(prim, invert_rotations: bool = True):
    xform = UsdGeom.Xform(prim)
    mat = np.array(xform.GetLocalTransformation(), dtype=np.float32)
    if invert_rotations:
        rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].T.flatten()))
    else:
        rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].flatten()))
    pos = mat[3, :3]
    return wp.transform(pos, rot)

def parse_scale(prim):
    xform = UsdGeom.Xform(prim)
    scale = np.ones(3, dtype=np.float32)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale = np.array(op.Get(), dtype=np.float32)
    return scale

def parse_world_matrix(prim):
    xformable = UsdGeom.Xformable(prim)
    world_mat = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(world_mat)

def transform_points(world_mat, points, global_scale, global_translate):
    rot = world_mat.T[:3, :3]
    trans = world_mat.T[:3, 3]

    world_points = global_scale * (points @ rot.T) + global_scale * trans + global_translate
    return world_points

def parse_usd_scene(
    builder: ModelBuilder,
    source,
    invert_rotations: bool = True,
    verbose: bool = True,
    global_scale = 1.0,
    global_translate = wp.vec3(0.0, 0.0, 0.0),
    use_static_collider: bool = False,
    skip_mesh_approximation: bool = False,
    mesh_maxhullvert: int = 64,
) -> dict[str, Any]:
    
    def parse_newton_properties(prim):
        property_name = ["_newton__collision_group", 
                         "_newton__collision_isVisible", 
                         "_newton__contact_ka",
                         "_newton__contact_kd",
                         "_newton__contact_ke",
                         "_newton__contact_kf",
                         "_newton__contact_thickness",
                         "_newton__material_density",
                         "_newton__material_dynamicFriction",
                         "_newton__material_restitution",
                         "_newton__material_staticFriction",
                         "_newton__collision_approximation",
                         "_newton__collision_approximationMethod"]
        
        parsed_properties = {}
        
        for property in property_name:
            if has_attribute(prim, property):
                property_name = property[property.find('__') + 2:]
                property_value = parse_generic(prim, property)
                parsed_properties[property_name] = property_value
        return parsed_properties
    
    if isinstance(source, str):
        stage = Usd.Stage.Open(source, Usd.Stage.LoadAll)
    else:
        stage = source

    mass_unit = 1.0
    try:
        if UsdPhysics.StageHasAuthoredKilogramsPerUnit(stage):
            mass_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    except Exception as e:
        if verbose:
            print(f"Failed to get mass unit: {e}")
    linear_unit = 1.0
    try:
        if UsdGeom.StageHasAuthoredMetersPerUnit(stage):
            linear_unit = UsdGeom.GetStageMetersPerUnit(stage)
    except Exception as e:
        if verbose:
            print(f"Failed to get linear unit: {e}")


    print("[lightwheel] parsing scene----------------------------") 
    scene = {}
    API_SCHEMA = "NewtonCollisionAPI"
    
    def add_mesh_to_builder(builder, static_body_id, mesh_obj, use_static_collider, cfg, key, approximation, approximation_method, approximation_query):
        shape_id = -1
        if not use_static_collider:
            body_id = builder.add_body(
                xform=wp.transform_identity(),
                mass=1.0,
                key=key
            )
            
            shape_id = builder.add_shape_mesh(
                body=body_id,
                xform=wp.transform_identity(),
                mesh=mesh_obj,
                scale=(1.0, 1.0, 1.0),
                cfg=cfg,
                key="SHAPE_" + key
            )
            print("Added dynamic mesh[{}] to scene...".format(len(mesh_obj.vertices)))
        
        else:
            shape_id = builder.add_shape_mesh(
                body=static_body_id,
                #xform=xform,
                xform=wp.transform_identity(),
                mesh=mesh_obj,
                scale=(1.0, 1.0, 1.0),
                cfg=cfg,
                key="SCENE_SHAPE_" + key
            )
            
            print("Added static mesh[{}] to scene...".format(len(mesh_obj.vertices)))
            
        if (approximation and approximation_query is not None and shape_id >= 0):
            pass
            approximation_method="coacd"
            if approximation_method not in approximation_query:
                approximation_query[approximation_method] = []          
            approximation_query[approximation_method].append(shape_id)
        
    mesh_list = []
    for prim in stage.Traverse():
        prim_name = parse_generic(prim, "primvars:file_name")
        if prim.IsA(UsdGeom.Mesh):
            if verbose:
                print("parsing {}...".format(prim_name))
            if not prim.GetAttribute(API_SCHEMA).IsValid():
                if verbose:
                    print(f"Skipping mesh {prim.GetPath()} without {API_SCHEMA} attribute")
                continue
            
            mesh = {}
            world_mat = parse_world_matrix(prim)
            xform = parse_xform(prim.GetParent())
            usd_mesh = UsdGeom.Mesh(prim)
            points = np.array(usd_mesh.GetPointsAttr().Get(), dtype = float)
            world_points = transform_points(world_mat, points, global_scale, global_translate)
            mesh["prim_name"] = prim_name
            mesh["world_mat"] = world_mat
            mesh["xform"] = xform
            mesh["points"] = world_points
            mesh["pointsCount"] = len(mesh["points"])
            mesh["faceVertexCounts"] = np.array(usd_mesh.GetFaceVertexCountsAttr().Get(), dtype = int)
            mesh["faceVertexIndices"] = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype = int)
            
            normals = usd_mesh.GetNormalsAttr().Get()
            if normals:
                mesh["normals"] = np.array(normals, dtype = float)
                
            mesh_obj = Mesh(
                vertices=mesh["points"], 
                indices=mesh["faceVertexIndices"],
                normals=mesh["normals"],
                maxhullvert=mesh_maxhullvert
            )
            
            newton_properties = parse_newton_properties(prim)
            shape_config = ModelBuilder.ShapeConfig(
                density=newton_properties["material_density"],
                ke=newton_properties["contact_ke"],
                kd=newton_properties["contact_kd"],
                kf=newton_properties["contact_kf"],
                ka=newton_properties["contact_ka"],
                mu=newton_properties["material_dynamicFriction"],
                restitution=newton_properties["material_restitution"],
                thickness=newton_properties["contact_thickness"],
                is_solid=True, 
                collision_group=newton_properties["collision_group"],
                collision_filter_parent=True,
                has_shape_collision=True,
                has_particle_collision=True,
                is_visible=newton_properties["collision_isVisible"],
            )
            
            if not skip_mesh_approximation:
                mesh_approximation = newton_properties["collision_approximation"]
                mesh_approximation_method = newton_properties["collision_approximationMethod"]
                mesh["mesh_approximation"] = mesh_approximation
                mesh["mesh_approximation_method"] = mesh_approximation_method

            mesh["mesh_obj"] = mesh_obj
            mesh["shape_config"] = shape_config

            if verbose:
                print("Mesh data:", mesh["prim_name"], mesh["pointsCount"])
            mesh_list.append(mesh)

    remesh_query = {}
    # breakpoint()
    
    if (use_static_collider):
        static_body_id = builder.add_body(
            xform=wp.transform_identity(),
            mass=0.0,
            key="SCENE_STATIC_BODY"
        )
    else:
        static_body_id = -1
        
    if len(mesh_list) > 0:
        for mesh_dict in mesh_list:
            mesh_obj = mesh_dict["mesh_obj"]
            shape_config = mesh_dict["shape_config"]
            key = mesh_dict["prim_name"]
            scene[key] = mesh_dict
            approximation = mesh_dict.get("mesh_approximation", False) 
            approximation_method = mesh_dict.get("mesh_approximation_method", "coacd")
            add_mesh_to_builder(builder, -1, mesh_obj, use_static_collider, shape_config, key, approximation, approximation_method, remesh_query)
            
    # approximate meshes
    for remeshing_method, shape_ids in remesh_query.items():
        if verbose:
            print(f"Approximating {len(shape_ids)} meshes with method '{remeshing_method}'")
        builder.approximate_meshes(method=remeshing_method, shape_indices=shape_ids)

    if verbose:
        print("builder: {} shapes ".format(builder.shape_count))
    return scene

def parse_usd_particles(
    builder: ModelBuilder,
    particle_key: str,
    source,
    global_scale = 1.0,
    global_translate = wp.vec3(0.0, 0.0, 0.0),
    invert_rotations: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    def parse_newton_properties(prim, key_list):
        # property_name = ["_newton__particle_flags", 
        #                  "_newton__particle_mass", 
        #                  "_newton__particle_qd",
        #                  "_newton__particle_radius"]
        
        parsed_properties = {}
        
        for property in key_list:
            if has_attribute(prim, property):
                key_list = property[property.find('__') + 2:]
                property_value = parse_generic(prim, property)
                parsed_properties[key_list] = property_value
        return parsed_properties
    
    if isinstance(source, str):
        stage = Usd.Stage.Open(source, Usd.Stage.LoadAll)
    else:
        stage = source
        
    print("[lightwheel] parsing particles----------------------------")
    particles = {}

    material_parameter = {}
    
    for prim in stage.Traverse():
        prim_name = prim.GetName()
        if prim.IsA(UsdGeom.Points) and particle_key in prim_name.lower():
            if verbose:
                print("parsing {}...".format(prim_name))
            points_prim = UsdGeom.Points(prim)
            points = np.array(points_prim.GetPointsAttr().Get(), dtype = float)
            world_mat = parse_world_matrix(prim)
            xform = parse_xform(prim.GetParent())
            world_points = transform_points(world_mat, points, global_scale, global_translate)
            particles["prim_name"] = prim_name
            particles["world_mat"] = world_mat
            particles["xform"] = xform
            particles["points"] = world_points
            particles["pointsCount"] = len(particles["points"])

            base_property_name_list = ["_newton__particle_flags", 
                "_newton__particle_mass", 
                "_newton__particle_qd",
                "_newton__particle_radius"
            ]
            material_name_list = [
                "_newton__particle_material_youngModulus",
                "_newton__particle_material_damping",
                "_newton__particle_material_poissonRatio",
                "_newton__particle_material_hardening",
                "_newton__particle_material_friction",
                "_newton__particle_material_yieldPressure",
                "_newton__particle_material_yieldStress",
                "_newton__particle_material_tensileYieldRatio"
            ]
            
            newton_properties = parse_newton_properties(prim, base_property_name_list)
            particles["flags"] = newton_properties["particle_flags"]
            particles["mass"] = newton_properties["particle_mass"]
            particles["qd"] = newton_properties["particle_qd"]
            particles["radius"] = newton_properties["particle_radius"]
            newton_mpm_material = parse_newton_properties(prim, material_name_list)
            particle_count = len(world_points)
            material_parameter["pt_count"] = particle_count
            material_parameter["mpm_material"] = newton_mpm_material
            # breakpoint()

            builder.add_particles(
                pos=world_points,
                vel=particles["qd"],
                mass=particles["mass"],
                radius=particles["radius"],
                flags=particles["flags"]
            )
            
    return material_parameter
