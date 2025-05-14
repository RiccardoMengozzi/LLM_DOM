import time, os, logging, cv2, shutil, gc
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

_VIEW = _FRONT_VIEW


_PROMPT_COMMANDS = [
    "pick",
    "grasp",
    "take",
]


NUMBER_OF_DATA = 3000
DT = 0.005
GUI = False # show renders
TRAIN_DATA_PERCENTAGE = 0.8

#SIM_TIME = 4.2  # seconds
SHOW_VIEWER = False
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






def update_cam_pose(cam: Camera, original_view) -> None:
    x_pos_noise = np.random.uniform(-0.2, 0.2)
    y_pos_noise = np.random.uniform(-0.2, 0.2)
    z_pos_noise = np.random.uniform(-0.5, 0.5)
    pos_noise = np.array([x_pos_noise, y_pos_noise, z_pos_noise])

    x_lookat_noise = np.random.uniform(-0.2, 0.2)
    y_lookat_noise = np.random.uniform(-0.2, 0.2)
    z_lookat_noise = np.random.uniform(-0.5, 0.5)
    lookat_noise = np.array([x_lookat_noise, y_lookat_noise, z_lookat_noise])

    random_up = np.random.uniform(-1, 1, size=3)
    random_up = random_up / np.linalg.norm(random_up)

    cam.set_pose(
        pos=original_view[0] + pos_noise,
        lookat=original_view[1] + lookat_noise,
        up=random_up,
    )

    cam.set_params(
        fov=original_view[2] + np.random.uniform(-30, 30),
    )


def from_world_to_screen(cam: Camera, P_world: np.ndarray):
    cam_pos = cam.pos
    lookat = cam.lookat
    fov = cam.fov
    res = cam.res
    up = cam.up

    T_wc = gs.pos_lookat_up_to_T(cam_pos, lookat, up)
    P_cam = gs.inv_transform_by_T(P_world, T_wc)

    # Compute intrinsics (vertical FOV)
    fov_rad = np.radians(fov)
    fy = 0.5 * res[1] / np.tan(fov_rad / 2)
    fx = fy  # Positive focal length
    cx, cy = res[0] / 2, res[1] / 2

    # Project to screen
    u = -fx * (P_cam[0] / P_cam[2]) + cx
    v = fy * (P_cam[1] / P_cam[2]) + cy

    return [u, v]


def get_random_prompt(objects: dict) -> str:
    # Generate a random prompt based on the positions of the objects
    object_names = list(objects.keys())
    random_object_name = np.random.choice(object_names)
    random_object_name_ = random_object_name.replace("_", " ")
    random_prompt_command = np.random.choice(_PROMPT_COMMANDS)
    prompt = f"{random_prompt_command} the {random_object_name_}"
    return prompt, random_object_name


def draw_text(
    img: np.ndarray, bbox: tuple, label: str, color: tuple, line_thickness: int = None
):
    """
    Disegna un bounding box in stile YOLO con etichetta su sfondo colorato.
    :param img: immagine BGR su cui disegnare
    :param bbox: (x1, y1, x2, y2)
    :param label: testo da mostrare, es. "person 0.87"
    :param color: tripla BGR es. (0,255,0)
    :param line_thickness: spessore bordi; se None, viene calcolato automaticamente
    """
    # Calcola spessore dinamico
    tl = line_thickness or max(1, int(round(0.002 * max(img.shape[0:2]))))
    x1, y1, x2, y2 = map(int, bbox)

    # Misura dimensione testo
    t_size = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, tl / 6, thickness=max(tl - 1, 1)
    )[0]
    # Coordinate sfondo etichetta
    c2 = x1 + t_size[0], y1 - t_size[1] - 3
    # Disegna sfondo etichetta
    cv2.rectangle(img, (x1, y1), c2, color, thickness=-1, lineType=cv2.LINE_AA)
    # Scrivi testo sopra lo sfondo
    cv2.putText(
        img,
        label,
        (x1, y1 - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        tl / 6,
        (255, 255, 255),
        thickness=max(tl - 1, 1),
        lineType=cv2.LINE_AA,
    )


def generate_dataset(
    scene: gs.Scene, 
    cam: Camera, 
    obj_manager: ObjectManager, 
    workspace: tuple[float, float, float, float, float, float], 
    warehouse: tuple[float, float, float, float, float, float],
    main_tables: list[XMLObject] = None,
    object_tables: list[XMLObject] = None,
    GUI: bool = False
) -> tuple:

    objects = obj_manager.objects
    train_start_idx = 0
    val_start_idx = 0
    for i, (main_table, object_table) in enumerate(zip(main_tables, object_tables)):
        frames = []
        pixel_coords = {}
        total_pixel_coords = []
        prompts = []
        obj_manager.reset_table_positions()
        obj_manager.select_table(main_table, MAIN_TABLE_POS)
        obj_manager.select_table(object_table, OBJECT_TABLE_POS)
        with tqdm(total=int(NUMBER_OF_DATA / len(main_tables)), desc=f"Generating data: {i+1}/{len(main_tables)}", unit="img") as pbar:
            for j in range(int(NUMBER_OF_DATA / len(main_tables))):
                counter = 0
                img_ok = False
                while not img_ok:
                    number_of_objects = np.random.randint(MIN_NUMBER_OF_OBJECTS, MAX_NUMBER_OF_OBJECTS)
                    success, objects = obj_manager.reset_positions_randomly(workspace, number_of_objects, warehouse)
                    if not success:
                        logger.error(f"{Colors.RED}Failed to reset positions")
                        exit()
                    update_cam_pose(cam, _TOP_VIEW)  # change the camera pose
                    scene.step()

                    rgb_img = cam.render()  # render the scene
                    frame = rgb_img[0]  # If using PyTorch/other tensor
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # BGR -> RGB conversion

                    frame_vis = frame.copy()

                    for object_name in objects:
                        world_pos = objects[object_name].entity.get_pos()
                        pixel_pos = from_world_to_screen(cam, world_pos.cpu().numpy())

                        pixel_pos = tuple(map(int, pixel_pos))
                        if (
                            0 <= pixel_pos[0] < frame.shape[1]
                            and 0 <= pixel_pos[1] < frame.shape[0]
                        ):
                            # Esempio di bounding box e confidenza
                            bbox = (
                                pixel_pos[0] - 15,
                                pixel_pos[1] - 15,
                                pixel_pos[0] + 15,
                                pixel_pos[1] + 15,
                            )
                            label_text = f"{object_name}"
                            color = (0, 0, 0)  # oppure scegli da una palette per classe
                            draw_text(frame_vis, bbox, label_text, color)

                            pixel_coords[object_name] = pixel_pos
                            # check if all objects are in image
                            counter += 1
                            if counter == len(objects):
                                img_ok = True
                                logger.info(f"{Colors.WHITE}Image {j} is ok")
                        else:
                            pixel_coords[object_name] = None
                            counter = 0
                            img_ok = False
                            logger.warning(
                                f"Pixel position {pixel_pos} is out of bounds for object '{object_name}'"
                            )
                            break

                prompt, object_name = get_random_prompt(objects)
                frames.append(frame)
                prompts.append(prompt)
                total_pixel_coords.append(pixel_coords[object_name])
                pbar.update(1)

                # if GUI:
                #     cv2.imshow("image", frame_vis)
                #     cv2.waitKey(1)

        data = list(zip(frames, prompts, total_pixel_coords))
        first_save = True if i == 0 else False
        train_data_length, val_data_length = save_data(data, train_start_idx, val_start_idx, cleanup=first_save)
        train_start_idx += train_data_length
        val_start_idx += val_data_length
        # Cleanup
        del frames
        del prompts
        del total_pixel_coords
        gc.collect()



def run() -> None:
    ########################## init ##########################
    gs.init(backend=gs.cpu, logging_level="warning")

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
    obj_manager = ObjectManager(scene)
    main_tables = []
    object_tables = []
    for i, texture in enumerate(_TABLE_TEXTURES):
        main_table = XMLObject(
            _TABLE_XML_FILE,
            name=f"main_table{i}",
            size=0.6,   
        )
        object_table = XMLObject(
            _TABLE_XML_FILE,
            name=f"object_table{i}",
            size=0.6,   
        )
        main_tables.append(main_table)
        object_tables.append(object_table)
        change_table_texture(_TABLE_XML_FILE, texture)
        obj_manager.add_table(main_table)
        obj_manager.add_table(object_table)
        obj_manager.spawn_table(main_table)
        obj_manager.spawn_table(object_table)  

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
        ("red_cube", colors["red"]),
        ("blue_cube", colors["blue"]),
        ("green_cube", colors["green"]),
        ("black_cube", colors["black"]),
        ("white_cube", colors["white"]),
    ]

    cylinders = [
        ("orange_cylinder", colors["orange"]),
        ("yellow_cylinder", colors["yellow"]),
        ("red_cylinder", colors["red"]),
        ("blue_cylinder", colors["blue"]),
        ("purple_cylinder", colors["purple"]),
        ("black_cylinder", colors["black"]),
        ("white_cylinder", colors["white"]),
    ]

    cups = [
        ("black_cup", colors["black"]),
        ("white_cup", colors["white"]),
        ("green_cup", colors["green"]),
        ("yellow_cup", colors["yellow"]),
        ("orange_cup", colors["orange"]),
        ("purple_cup", colors["purple"]),
    ]

    plates = [
        ("black_plate", colors["black"]),
        ("red_plate", colors["red"]),
        ("blue_plate", colors["blue"]),
        ("green_plate", colors["green"]),
        ("white_plate", colors["white"]),
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

    success = obj_manager.spawn_all(warehouse)
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

    ########################## run ##########################

    generate_dataset(
        scene=scene,
        cam=cam,
        obj_manager=obj_manager,
        workspace=workspace,
        warehouse=warehouse,
        main_tables=main_tables,
        object_tables=object_tables,
        GUI=GUI,
    )


    gs.destroy()


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


def save_data(data, train_idx, val_idx, cleanup=True):
    
    # Shuffle the data to ensure randomness
    np.random.shuffle(data)

    # Calculate the split index for 80% training data
    split_index = int(TRAIN_DATA_PERCENTAGE * len(data))

    # Split the data
    train_data = data[:split_index]
    val_data = data[split_index:]


    with tqdm(total=len(data), desc="Saving data", unit="img") as pbar:
        for subset, data, start_idx in zip(["train", "val"], [train_data, val_data], [train_idx, val_idx]):
            base_dir = os.path.join("perceiver_data", subset)
            dirs = {
                "images": os.path.join(base_dir, "images"),
                "prompts": os.path.join(base_dir, "prompts"),
                "coordinates": os.path.join(base_dir, "coordinates"),
            }

            # 1) Pulisci (o crea) le cartelle
            if cleanup:
                for d in dirs.values():
                    if os.path.isdir(d):
                        # elimina tutti i file/dir interni
                        for entry in os.listdir(d):
                            path = os.path.join(d, entry)
                            # rimuovi file o directory ricorsivamente
                            if os.path.isfile(path) or os.path.islink(path):
                                os.remove(path)
                            else:
                                shutil.rmtree(path)
                    else:
                        os.makedirs(d, exist_ok=True)

            # 2) Salvataggio dei dati
            for i, (frame, prompt, coord) in enumerate(data):
                # Definisci i percorsi
                image_path = os.path.join(dirs["images"], f"{i+start_idx}.jpg")
                prompt_path = os.path.join(dirs["prompts"], f"{i+start_idx}.txt")
                coord_path = os.path.join(dirs["coordinates"], f"{i+start_idx}.txt")

                # Salva immagine
                cv2.imwrite(image_path, frame)

                # Salva prompt
                with open(prompt_path, "w") as f:
                    f.write(f"{prompt}\n")

                # Salva coordinate
                with open(coord_path, "w") as f:
                    f.write(f"{coord[0]}, {coord[1]}\n")

                pbar.update(1)

    return len(train_data), len(val_data)



def shuffle_dataset(folder: str) -> None:
    """
    Shuffles triplets of files in `folder/images`, `folder/coordinates`, `folder/prompts`,
    renaming them in-place con nuovi indici casuali mantenendo i triplet accoppiati.
    
    :param folder: Path alla cartella radice contenente le sottocartelle.
    """
    # Definizione delle sottocartelle
    subdirs = {
        'images':  '.jpg',
        'coordinates': '.txt',
        'prompts': '.txt',
    }
    
    # 1) Costruiamo la lista di indici esistenti
    indices = []
    for name, ext in subdirs.items():
        path = os.path.join(folder, name)
        files = [f for f in os.listdir(path) if f.endswith(ext)]
        # Estraiamo gli indici numerici (prima di .jpg/.txt)
        idxs = sorted(int(os.path.splitext(f)[0]) for f in files)
        if not indices:
            indices = idxs
        else:
            # Verifichiamo che tutti i subdir contengano gli stessi indici
            assert indices == idxs, (
                f"I file in `{name}` non corrispondono agli indici nelle altre cartelle: "
                f"{indices} vs {idxs}"
            )
    
    # 2) Creiamo una permutazione casuale
    new_indices = indices.copy()
    np.random.shuffle(new_indices)  # O(n) :contentReference[oaicite:0]{index=0}
    
    # 3) Rinominiamo con nomi temporanei per evitare collisioni
    temp_suffix = ".tmp_shuffle"
    mapping = dict(zip(indices, new_indices))
    
    # Fase A: rinomina a .tmp_shuffle
    for old_idx, new_idx in mapping.items():
        for sub, ext in subdirs.items():
            src = os.path.join(folder, sub, f"{old_idx}{ext}")
            dst = os.path.join(folder, sub, f"{new_idx}{ext}{temp_suffix}")
            os.rename(src, dst)  # rinomina diretto :contentReference[oaicite:1]{index=1}

    # Fase B: togliamo il suffisso temporaneo
    for sub, ext in subdirs.items():
        path = os.path.join(folder, sub)
        for f in os.listdir(path):
            if f.endswith(temp_suffix):
                base = f[:-len(temp_suffix)]
                final = base  # già "new_idx.ext"
                os.rename(
                    os.path.join(path, f),
                    os.path.join(path, final)
                )

    print(f"Shuffle completed")

def main():
    run()
    shuffle_dataset("/home/mengo/gs_ws/perceiver_data/train")
    shuffle_dataset("/home/mengo/gs_ws/perceiver_data/val")



if __name__ == "__main__":
    main()
