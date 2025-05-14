import genesis as gs
import numpy as np
import logging, time
from scipy.spatial import KDTree
from typing import Optional, Tuple, List, Dict, Union
from genesis.engine.entities.rigid_entity import RigidEntity
from SceneObject import SceneObject, ObjectType
from logging_formatter import ColoredFormatter, Colors

logger = logging.getLogger("ObjectManager")
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler()
formatter = ColoredFormatter(
    "[%(name)s] [%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

MAX_TRIES = int(1e3)


class ObjectManager:
    """
    Manages SceneObject instances within a Genesis scene, handling
    addition, spawning, positioning, and collision-free placement.
    """

    def __init__(
        self, scene: gs.Scene, objects: Optional[Dict[str, SceneObject]] = None
    ) -> None:
        """
        Initialize ObjectManager.

        :param scene: Genesis Scene instance to manage objects within.
        :param objects: Optional initial mapping of names to SceneObject.
        """
        self.scene = scene
        self.objects: Dict[str, SceneObject] = objects.copy() if objects else {}
        self.n_objects = len(self.objects)
        self.tables: Dict[str, SceneObject] = {}
        self.table_positions: Dict[str, Tuple[float, float, float]] = {}
        self.table_positions_start = (20,20,0)
        self.current_used_tables = []



    def add_object(self, obj: SceneObject) -> None:
        """
        Add a single SceneObject to the manager.

        :param obj: SceneObject to register.
        :raises AssertionError: if an object with the same name already exists.
        """
        assert (
            obj.name not in self.objects
        ), f"Object with name {obj.name} already exists."
        self.objects[obj.name] = obj

    def add_objects(self, objs: List[SceneObject]) -> None:
        """
        Batch-add multiple SceneObject instances.

        :param objs: List of SceneObject to register.
        """
        for obj in objs:
            self.add_object(obj)

    def add_table(self, table: SceneObject) -> None:
        """
        Add a table object to the manager.

        :param obj: SceneObject to register.
        :raises AssertionError: if an object with the same name already exists.
        """
        assert (
            table.name not in self.tables
        ), f"Object with name {table.name} already exists."
        self.tables[table.name] = table
        self.table_positions[table.name] = self.table_positions_start
        self.table_positions_start = (
            self.table_positions_start[0],
            self.table_positions_start[1] + table.size,
            self.table_positions_start[2],
        )

    def add_tables(self, tables: List[SceneObject]) -> None:
        """
        Batch-add multiple SceneObject instances.

        :param tables: List of SceneObject to register.
        """
        for table in tables:
            self.add_table(table)

    def roi(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        min_z: float,
        max_z: float,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Set the region of interest (ROI) for object placement.
        :param min_x: Minimum x-coordinate.
        :param max_x: Maximum x-coordinate.
        :param min_y: Minimum y-coordinate.
        :param max_y: Maximum y-coordinate.
        :param min_z: Minimum z-coordinate.
        :param max_z: Maximum z-coordinate.
        """
        return (
            min_x,
            max_x,
            min_y,
            max_y,
            min_z,
            max_z,
        )
    
    def length(self) -> int:
        """
        Get the number of objects managed.
        :return: Number of SceneObject instances.
        """
        return len(self.objects)
    
    def create_dictionary(self, objs: List[SceneObject]) -> Dict[str, SceneObject]:
        """
        Create a dictionary mapping object names to SceneObject instances.

        :param objs: List of SceneObject to convert.
        :return: Dictionary of object names to SceneObject.
        """
        return {obj.name: obj for obj in objs}

        
    def spawn(
        self,
        obj: SceneObject,
        pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> RigidEntity:
        """
        Spawn a single object into the scene at specified pose.

        :param obj: SceneObject to spawn.
        :param pos: (x, y, z) position before applying object's z-offset.
        :param quat: (x, y, z, w) orientation quaternion.
        :return: The created RigidEntity.
        """
        logger.info(
            f"{Colors.WHITE}Spawning {Colors.GREEN}'{obj.name}'{Colors.WHITE} of type {Colors.GREEN}{obj.type}{Colors.WHITE} at {Colors.GREEN}{pos}{Colors.WHITE}"
        )
        return obj.spawn(self.scene, pos, quat)

    def _get_random_pos(
        self, roi: Tuple[float, float, float, float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Generate a random position within the region of interest.

        :param roi: (x_min, x_max, y_min, y_max, z_min, z_max).
        :return: Random (x, y, z) tuple.
        """
        return (
            gs.utils.random.uniform(roi[0], roi[1]),
            gs.utils.random.uniform(roi[2], roi[3]),
            gs.utils.random.uniform(roi[4], roi[5]),
        )

    def _does_object_collide(
        self,
        obj: SceneObject,
        candidate_pos: Tuple[float, float, float],
        other_positions: List[Tuple[float, float, float]],
        other_sizes: List[float],
        min_dist: float = 0.02,
    ) -> bool:
        """
        Check for overlap in XY plane between candidate position and existing objects.

        :param obj: SceneObject under consideration.
        :param candidate_pos: Proposed (x, y, z) or numpy array.
        :param other_positions: List of existing object positions.
        :param other_sizes: Corresponding object sizes.
        :param min_dist: Minimum clearance distance.
        :return: True if collision detected, False otherwise.
        """
        
        if not other_positions:
            return False
        x, y, _ = candidate_pos
        for (ox, oy, _), size in zip(other_positions, other_sizes):
            if (
                abs(x - ox) < (obj.size + size) / 2 + min_dist
                and abs(y - oy) < (obj.size + size) / 2 + min_dist
            ):
                return True
        return False

    def objects_in_roi(
        self, roi: Tuple[float, float, float, float, float, float]
    ) -> List[SceneObject]:
        """
        List all spawned objects whose positions lie within the ROI.

        :param roi: (x_min, x_max, y_min, y_max, z_min, z_max).
        :return: SceneObject list.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = roi
        if z_max - z_min < 1e-3:
            z_min -= 0.1
            z_max += 0.1
        found: List[SceneObject] = []
        for obj in self.objects.values():
            if obj.entity is None:
                continue
            px, py, pz = obj.get_pos()
            if x_min <= px <= x_max and y_min <= py <= y_max and z_min <= pz <= z_max:
                found.append(obj)
        return found

    def find_free_pos(
        self,
        obj: SceneObject,
        roi: Tuple[float, float, float, float, float, float],
        other_positions: Optional[List[Tuple[float, float, float]]] = None,
        other_sizes: Optional[List[float]] = None,
        max_tries: int = int(1e3),
    ) -> Optional[Tuple[float, float, float]]:
        """
        Attempt to find a non-colliding random position for an object within ROI.

        :param obj: SceneObject to place.
        :param roi: Region of interest (x_min, x_max, y_min, y_max, z_min, z_max).
        :param other_objects: Precomputed existing objects or None.
        :param other_positions: Precomputed existing positions or None.
        :param other_sizes: Corresponding sizes or None.
        :param max_tries: Maximum random attempts.
        :return: Free position or None on failure.
        """

        if len(other_positions) == 0:
            return self._get_random_pos(roi)

        if not hasattr(self, '_kd_tree') or len(other_positions) != len(self._kd_tree.data):
            pts = np.array(other_positions)[:, :2]  # solo XY
            self._kd_tree = KDTree(pts) if len(pts) > 0 else None

        for _ in range(max_tries):
            x = np.random.uniform(roi[0], roi[1])
            y = np.random.uniform(roi[2], roi[3])
            # Controllo collisione tramite KD-Tree
            if self._kd_tree:
                dist, idx = self._kd_tree.query([x, y], k=1)
                if dist < (obj.size + other_sizes[idx]) / 2 + 0.02:
                    continue
            return (x, y, np.random.uniform(roi[4], roi[5]))
        return None

    def _place_objects(self, 
                       objs: List[SceneObject], 
                       roi: Tuple[float, float, float, float, float, float], 
                       use_spawn: bool
                       ) -> bool:
        """
        Core logic to place objects either spawning or repositioning.

        :param objs: List of SceneObject to place.
        :param use_spawn: If True, call spawn; else call set_pos.
        :return: True if all placed, False otherwise.
        """
        placed_positions: List[Tuple[float, float, float]] = []
        placed_sizes: List[float] = []
        # Sort objects by size, largest first, easier fit
        sorted_objs = sorted(objs, key=lambda o: o.size, reverse=True)
        for obj in sorted_objs:
            pos = self.find_free_pos(obj, roi, placed_positions, placed_sizes, max_tries=MAX_TRIES)
            if pos is None:
                logger.warning(
                    f"Cannot place object '{obj.name}' in ROI: {roi}."
                )
                return False
            if use_spawn:
                self.spawn(obj, pos)
            else:
                obj.set_pos(pos)
            # Remember the position and size of the placed object
            placed_positions.append(pos)
            placed_sizes.append(obj.size)
        return True

    def spawn_subset(
        self,
        objs: List[SceneObject],
        roi: Tuple[float, float, float, float, float, float],
    ) -> bool:
        """
        Spawn a list of objects within specified or workspace ROI, largest-first.
        """
        return self._place_objects(objs, roi, use_spawn=True)

    def spawn_all(
        self,
        roi: Tuple[float, float, float, float, float, float],
    ) -> bool:
        """
        Spawn all managed objects in scene.
        """
        return self._place_objects(list(self.objects.values()), roi, use_spawn=True)

    def spawn_random(
        self,
        roi: Tuple[float, float, float, float, float, float],
        count: int = 1,
    ) -> Tuple[bool, List[SceneObject]]:
        """
        Spawn a random subset of managed objects.
        """
        assert 1 <= count <= len(self.objects), "Invalid number of objects to spawn."
        pool = list(self.objects.values())
        np.random.shuffle(pool)
        selected = pool[:count]
        return (
            self.spawn_subset(selected, roi),
            selected,
        )

    def change_pose(
        self,
        obj: SceneObject,
        pos: Optional[Tuple[float, float, float]] = None,
        quat: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """
        Update position and/or orientation of an existing object.
        """
        assert obj.name in self.objects, f"Object '{obj.name}' not managed."
        assert pos is not None or quat is not None, "Must specify pos or quat."
        if pos is not None:
            obj.entity.set_pos(pos)
        if quat is not None:
            obj.entity.set_quat(quat)

    def set_random_pos(
        self,
        obj: SceneObject,
        roi: Tuple[float, float, float, float, float, float],
    ) -> bool:
        """
        Assign a random collision-free position to an existing object.
        """
        assert obj.name in self.objects, f"Object '{obj.name}' not managed."
        pos = self.find_free_pos(
            obj, roi
        )
        assert pos is not None, f"Cannot find free position for '{obj.name}'."
        obj.set_pos(pos)
        return True

    def reset_positions_randomly(
        self,
        roi: Tuple[float, float, float, float, float, float],
        number_of_objects: Optional[int] = None,
        store_roi: Optional[Tuple[float, float, float, float, float, float] | None] = None,
    ) -> tuple[bool, List[SceneObject]]:
        """
        Reassign random positions to a subset or all objects, placing excess in warehouse if set.

        :param roi: Region of interest for object placement.
        :param number_of_objects: Number of objects to place in the scene.
        :param store_roi: Region of interest for storing excess objects.
        :return: True if all objects are placed successfully, False otherwise.
        """
        objs = list(self.objects.values())
        assert not(number_of_objects is not None and store_roi is None), "You specified a number of objects that is less than the total number of objects: you also need to specify a roi to store the rest of the objects."
        if number_of_objects is not None:
            assert 0 <= number_of_objects <= len(objs), "Invalid number_of_objects."

        if number_of_objects is None:
            number_of_objects = len(objs)

        for i in range(MAX_TRIES):
            # Move everything to warehouse first
            self._place_objects(objs, store_roi, use_spawn=False)

            np.random.shuffle(objs)
            selected = objs[:number_of_objects]
            rest = objs[number_of_objects:]
            logger.debug(
                f"Attempt {i+1} to place {len(selected)} objects in ROI: {roi}."
            )
            success = self._place_objects(selected, roi, use_spawn=False)
            if rest:
                success &= self._place_objects(rest, store_roi, use_spawn=False)
            if success:
                selected = self.create_dictionary(selected)
                return success, selected
        return False, []

    def spawn_table(self, table: SceneObject) -> None:
        """
        Spawn a single table in the scene.
        """
        table.spawn(self.scene, self.table_positions[table.name])

    def spawn_tables(self) -> None:
        """
        Spawn all registered tables in the scene.
        """
        for table in self.tables.values():
            table.spawn(self.scene, self.table_positions[table.name])

    def reset_table_positions(self) -> None:
        """
        Reset the positions of all tables to their initial state.
        """
        for table in self.current_used_tables:
            table.entity.set_pos(self.table_positions[table.name])
        self.current_used_tables = []

    def select_table(self,
                    table: SceneObject,
                    pos: Tuple[float, float, float],
                    ) -> None:
        # Move the selected table to the scene position
        table.entity.set_pos(pos)
        self.current_used_tables.append(table)
        

    