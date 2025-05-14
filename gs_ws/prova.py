import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity
gs.init(backend=gs.cuda)


_CUP_XML_FILE = "models/cup/cup.urdf"

scene = gs.Scene(
    show_viewer=True, 
    rigid_options=gs.options.RigidOptions(
        box_box_detection=True,  # Needed otherwise boxes shake and jump on the table for some reason
    ),
)

plane = scene.add_entity(gs.morphs.Plane())
# franka = scene.add_entity(
#     gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),

# )
table = scene.add_entity(
    gs.morphs.MJCF(
        file="models/table/table.xml",
    )
)

white_cup : RigidEntity = scene.add_entity(
    gs.morphs.URDF(
        file=_CUP_XML_FILE,
        pos=(0.0,0.0,1.0)
    ),
    surface=gs.options.surfaces.Default(
        color=(1.0, 1.0, 1.0),
    ),
)

scene.build()

while True:
    scene.step()