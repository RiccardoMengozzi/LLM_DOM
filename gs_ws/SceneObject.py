from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple

import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity


class ObjectType(Enum):
    DEFAULT = 0
    CUBE = 1
    CYLINDER = 2
    CUP = 3
    PLATE = 4


class SceneObject(ABC):
    """
    Base class for all scene objects, handling common logic for spawning
    and positioning an entity with a z-offset.
    """
    def __init__(self, name: str, color: Tuple[float, float, float], 
                 size: float, z_offset: float = 0.0,
                 obj_type: ObjectType = ObjectType.DEFAULT) -> None:
        self.name = name
        self.color = color
        self.size = size
        self.z_offset = z_offset
        self.type = obj_type
        self.entity: Optional[RigidEntity] = None

    @abstractmethod
    def _create_morph(self, pos: Tuple[float, float, float], 
                      quat: Tuple[float, float, float, float]) -> gs.morphs.Morph:
        """
        Return the Genesis morph for this object type,
        positioned at pos (with z already offset) and orientation quat.
        """
        ...

    def spawn(self,
              scene: gs.Scene,
              pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
              quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
              ) -> RigidEntity:
        """
        Adds the entity to the scene at (x, y, z + z_offset) with orientation quat.
        """
        # apply z-offset
        x, y, z = pos
        spawn_pos = (x, y, z + self.z_offset)

        morph = self._create_morph(pos=spawn_pos, quat=quat)
        if self.color is not None:
            self.entity = scene.add_entity(
                morph,
                surface=gs.options.surfaces.Default(color=self.color)
            )
        else:
            self.entity = scene.add_entity(morph)
        return self.entity

    def set_pos(self, pos: Tuple[float, float, float]) -> None:
        """
        Repositions the spawned entity to (x, y, z + z_offset).
        """
        if self.entity is None:
            raise ValueError(f"Entity for '{self.name}' is not spawned yet.")
        x, y, z = pos
        self.entity.set_pos((x, y, z + self.z_offset))

    def get_pos(self) -> Tuple[float, float, float]:
        """
        Returns current position of the entity in world coordinates.
        """
        if self.entity is None:
            raise ValueError(f"Entity for '{self.name}' is not spawned yet.")
        return tuple(self.entity.get_pos().cpu().numpy())


class CubeObject(SceneObject):
    def __init__(self,
                 size: float,
                 name: str = "cube",
                 color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
                 ) -> None:
        super().__init__(
            name=name,
            color=color,
            size=size,
            z_offset=size / 2,
            obj_type=ObjectType.CUBE
        )

    def _create_morph(self, pos: Tuple[float, float, float],
                      quat: Tuple[float, float, float, float]
                      ) -> gs.morphs.Box:
        return gs.morphs.Box(size=(self.size,) * 3, pos=pos, quat=quat)


class CylinderObject(SceneObject):
    def __init__(self,
                 radius: float,
                 height: float,
                 name: str = "cylinder",
                 color: Tuple[float, float, float] = (0.0, 1.0, 0.0)
                 ) -> None:
        diameter = radius * 2
        super().__init__(
            name=name,
            color=color,
            size=diameter,
            z_offset=height / 2,
            obj_type=ObjectType.CYLINDER
        )
        self.radius = radius
        self.height = height

    def _create_morph(self, pos: Tuple[float, float, float],
                      quat: Tuple[float, float, float, float]
                      ) -> gs.morphs.Cylinder:
        return gs.morphs.Cylinder(
            radius=self.radius,
            height=self.height,
            pos=pos,
            quat=quat
        )


class XMLObject(SceneObject):
    def __init__(
        self,
        file_path: str,
        name: str,
        size: float,
        color: Tuple[float, float, float] | None = None,
        obj_type: ObjectType = ObjectType.DEFAULT,
        z_offset: float = 0.0
    ) -> None:
        super().__init__(
            name=name,
            color=color,
            size=size,
            z_offset=z_offset,
            obj_type=obj_type
        )
        self.file_path = file_path

    def _create_morph(self, pos: Tuple[float, float, float],
                      quat: Tuple[float, float, float, float]
                      ):
        if self.file_path.endswith('.xml'):
            return gs.morphs.MJCF(
                file=self.file_path,
                pos=pos,
                quat=quat
            )
        elif self.file_path.endswith('.urdf'):
            return gs.morphs.URDF(
                file=self.file_path,
                pos=pos,
                quat=quat
            )

