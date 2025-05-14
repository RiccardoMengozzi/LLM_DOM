import time, os, logging, cv2, shutil, gc, copy
import open3d as o3d
from tqdm import tqdm
import numpy as np
import genesis as gs
import xml.etree.ElementTree as ET
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.vis.camera import Camera
from scipy.spatial.transform import Rotation as R
from logging_formatter import ColoredFormatter

from SceneObject import CubeObject, CylinderObject, XMLObject
from ObjectManager import ObjectManager, Colors

logger = logging.getLogger("Perceiver_dataset")
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler()
formatter = ColoredFormatter(
    "[%(name)s] [%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


np.set_printoptions(
    precision=4,      # 4 cifre decimali
    suppress=True,    # niente notazione scientifica per piccoli valori
    threshold=10,     # tronca se più di 10 elementi
    edgeitems=2,      # mostra 2 elementi iniziali/finali quando tronca
    linewidth=75,     # max 75 caratteri per riga
    floatmode='fixed' # stampa sempre con punto fisso
)



_TOP_VIEW = [
    [0, 0.65, 3],  # camera pos
    [0, 0.65, 0],  # camera lookat
    30,  # camera fov
    [0, 1, 0],  # camera up
]

_LEFT_VIEW = [
    [1.0, 1.125, 0.75],  # camera pos
    [0.0, 1.125, 0.75],  # camera lookat
    30,  # camera fov
]

_FRONT_VIEW = [
    [0.0, 2.2, 0.75],  # camera pos
    [0.0, 0.0, 0.75],  # camera lookat
    50,  # camera fov
    [0, 0, 1],  # camera up
]

_ANGLE_VIEW = [
    [0.5, 1.5, 2.0],
    [0.0, 0.65, 0.8],
    50,
    [0, 0, 1],
]

_VIEW = _TOP_VIEW


_PROMPT_COMMANDS = [
    "pick",
    "grasp",
    "take",
]


NUMBER_OF_DATA = 3000
DT = 0.005
GUI = True # show renders
TRAIN_DATA_PERCENTAGE = 0.8

#SIM_TIME = 4.2  # seconds
SHOW_VIEWER = True
MIN_NUMBER_OF_OBJECTS = 4
MAX_NUMBER_OF_OBJECTS = 10

MAIN_TABLE_POS = (0.0, 0.0, 0.0)
OBJECT_TABLE_POS = (0.0, 0.75, 0.0)

CUBE_SIZE = 0.04
CYLINDER_HEIGHT = 0.04
CYLINDER_RADIUS = 0.02
OBJECT_MIN_DISTANCE = 0.05  # distance between surfaces of objects, not centers

WS_CENTER = [0.0, 0.6, 0.76]  # center of the workspace
WS_SIZE = 0.25

WH_CENTER = [-25.0, 0.0, 0.0]
WH_SIZE = 10.0

_TABLE_TEXTURES = [
    "fabric0.png",
    "metal0.png",
    "metal1.png",
    "metal2.png",
    "metal3.png",
    "stone0.png",
    "stone1.png",
    "stone2.png",
    "wood0.png",
    "wood1.png",
    "wood2.png",
    "wood3.png",
]


_TABLE_XML_FILE = "models/table/table.xml"
_CUP_XML_FILE = "models/cup/cup.urdf"
_PLATE_XML_FILE = "models/plate/plate.urdf"

def change_table_texture(xml_file, texture_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the texture element and change its value
    for texture in root.iter("texture"):
        file_attr = texture.get("file")
        if file_attr:
            texture.set("file", f"textures/{texture_file}")
            print(f"Updated texture file from '{file_attr}' to '{texture_file}'.")

    tree.write(xml_file)


def run():
    ########################## init ##########################
    gs.init(backend=gs.gpu)#, logging_level="warning")


    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            # !Important! Using dt >= 0.01 can cause instabilities in the physics contacts, use lower or use a lower dt in rigid_options.
            dt=DT,
        ),
        rigid_options=gs.options.RigidOptions(
            # dt=0.005,
            box_box_detection=False,  # True: Documentation says its slower but more stable, but it does not work well (bugged)
        ),
        show_viewer=SHOW_VIEWER,
    )

    cam = scene.add_camera(
        res=(1280, 960),
        pos=_VIEW[0],
        lookat=_VIEW[1],
        fov=_VIEW[2],
        up=_VIEW[3],
        GUI=GUI,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    change_table_texture(_TABLE_XML_FILE, 'wood1.png')

    table1: RigidEntity = scene.add_entity(
        gs.morphs.MJCF(
            file=_TABLE_XML_FILE,
            pos=MAIN_TABLE_POS,
        )
    )

    table2: RigidEntity = scene.add_entity(
        gs.morphs.MJCF(
            file=_TABLE_XML_FILE,
            pos=OBJECT_TABLE_POS,
        )
    )

    obj_manager = ObjectManager(scene)

    # Define the colors and sizes as dictionaries for easy access
    colors = {
        "red": (1, 0, 0),
        "blue": (0, 0, 1),
        "green": (0, 1, 0),
        "black": (0, 0, 0),
        "white": (1, 1, 1),
        "orange": (1, 0.5, 0),
        "yellow": (1, 1, 0),
        "purple": (0.5, 0, 0.5),
    }

    # Define the objects with their respective parameters
    cubes = [
        ("black_cube", colors["black"]),
        ("white_cube", colors["white"]),
    ]

    cylinders = [
        ("orange_cylinder", colors["orange"]),
        ("yellow_cylinder", colors["yellow"]),
    ]

    cups = [
        ("green_cup", colors["green"]),
        ("yellow_cup", colors["yellow"]),
    ]

    plates = [
        ("red_plate", colors["red"]),
    ]

    # Create the objects using loops
    cube_objects = [CubeObject(CUBE_SIZE, name=name, color=color) for name, color in cubes]
    cylinder_objects = [CylinderObject(CYLINDER_RADIUS, CYLINDER_HEIGHT, name=name, color=color) for name, color in cylinders]
    cup_objects = [XMLObject(_CUP_XML_FILE, name=name, size=0.12, color=color) for name, color in cups]
    plate_objects = [XMLObject(_PLATE_XML_FILE, name=name, size=0.3, color=color) for name, color in plates]

    # Combine all objects into a single list
    objects = cube_objects + cylinder_objects + cup_objects + plate_objects

    obj_manager.add_objects(objects)
    workspace = obj_manager.roi(
            WS_CENTER[0] - WS_SIZE,
            WS_CENTER[0] + WS_SIZE,
            WS_CENTER[1] - WS_SIZE,
            WS_CENTER[1] + WS_SIZE,
            WS_CENTER[2],
            WS_CENTER[2],
    )
    warehouse = obj_manager.roi(
            WH_CENTER[0] - WH_SIZE,
            WH_CENTER[0] + WH_SIZE,
            WH_CENTER[1] - WH_SIZE,
            WH_CENTER[1] + WH_SIZE,
            WH_CENTER[2],
            WH_CENTER[2],
    )


    success = obj_manager.spawn_all(workspace)
    if not success:
        return
    else:
        logger.info(f"{Colors.WHITE}All objects spawned successfully")


    franka: RigidEntity = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=np.array([0.0, 0.0, 0.76]),
            quat=np.array(R.from_euler("xyz", [np.pi / 2, 0.0, 0.0]).as_quat()),
        ),
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )

    ########################## build ##########################
    scene.build()

    qpos = np.array([0, -0.2, 0, -0.2, 0, 1.571, 0.785, 0.0, 0.0])
    franka.set_qpos(qpos)
    scene.step()

    voxel_size = 0.01  # lato del voxel in metri
    fov = cam.fov
    res = cam.res
    fov_rad = np.radians(fov)
    fy = 0.5 * res[1] / np.tan(fov_rad / 2)
    fx = fy  # Positive focal length
    cx, cy = res[0] / 2, res[1] / 2
    
    for i in range(100):
        rgb, depth, _, _ = cam.render(depth=True)
        rgb_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False,
        )


        # 3.1 Definizione degli intrinseci (esempio pinhole)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=rgb.shape[1],
            height=rgb.shape[0],
            fx=fx, fy=fy, cx=cx, cy=cy
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsics,
            extrinsic=np.eye(4),
            project_valid_depth_only=True
        )  # :contentReference[oaicite:6]{index=6}§
        
        pcd_center = pcd.get_center()
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)  # raggio in unità della scena
        world_frame = copy.deepcopy(cam_frame)
        T_cw = np.linalg.inv(gs.pos_lookat_up_to_T(cam.pos, cam.lookat, cam.up))
        print(T_cw)
        origin_wrt_cam = gs.transform_by_T(np.array([0.0, 0.0, 0.0]), T_cw)
        world_frame.translate(-T_cw[:3, 3])  # sposta al centro
        world_frame.rotate(T_cw[:3, :3])


        marker_frame = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)  # raggio in unità della scena
        marker_frame.paint_uniform_color([1.0, 0.0, 0.0])

        marker_center = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)  # raggio in unità della scena
        marker_center.translate(pcd_center)                                      # sposta al centro
        marker_center.paint_uniform_color([0.0, 1.0, 0.0])

        marker_origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)  # raggio in unità della scena
        marker_origin.translate(-origin_wrt_cam)                                      # sposta al centro
        marker_origin.paint_uniform_color([0.0, 0.0, 1.0])

        o3d.visualization.draw_geometries([pcd, cam_frame, world_frame, marker_frame, marker_center, marker_origin])




#         # Create the Axis-Aligned Bounding Box
#         aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

#         # Crop the point cloud
#         pcd = pcd.crop(aabb)



#         # Supponendo che 'pcd' sia la tua PointCloud
#         # Rot = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))  # Rotazione di 180° attorno all'asse X
#         # pcd.rotate(Rot, center=(0, 0, 0))

#         voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
#             pcd,
#             voxel_size=voxel_size,
#         )  # :contentReference[oaicite:8]{index=8}
#         print(len(voxel_grid.get_voxels()))
#         print(voxel_grid.get_center())
# # {
# # 	"class_name" : "ViewTrajectory",
# # 	"interval" : 29,
# # 	"is_loop" : false,
# # 	"trajectory" : 
# # 	[
# # 		{
# # 			"boundingbox_max" : [ 2.6021411873487637, 0.87084183854327257, 4.1917819976806641 ],
# # 			"boundingbox_min" : [ -2.6062050946077764, -1.9546600468347521, 1.1604272127151489 ],
# # 			"field_of_view" : 60.0,
# # 			"front" : [ 0.024065447281137999, 0.032447204837823256, -0.99918368338627872 ],
# # 			"lookat" : [ -0.077572157397282085, -0.087956002644674111, 2.7063003203128315 ],
# # 			"up" : [ 0.012841855772247373, -0.99940071472750547, -0.0321449551636988 ],
# # 			"zoom" : 0.2999999999999996
# # 		}
# # 	],
# # 	"version_major" : 1,
# # 	"version_minor" : 0
# # }

        # o3d.visualization.draw_geometries([pcd], 
        #                                   lookat=[ -0.077572157397282085, -0.087956002644674111, 2.7063003203128315],
        #                                   front=[ 0.024065447281137999, 0.032447204837823256, -0.99918368338627872 ],
        #                                   up=[ 0.012841855772247373, -0.99940071472750547, -0.0321449551636988 ],
        #                                   zoom=0.2999999999999996,
        #                                   )
                                          
                                         
                                            


        scene.step()


if __name__ == "__main__":
    run()