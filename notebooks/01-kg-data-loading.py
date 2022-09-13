# %%
from pathlib import Path

import pyvista as pv

from fepredict import __file__ as filestr

# %%
module_dir = Path(filestr).parents[2]
del filestr


# %%
raw_data_dir = module_dir.joinpath("data", "raw")
simulation_dir = raw_data_dir / "9905863M00"

tibia_cartilage_lateral = pv.read_meshio(
    simulation_dir / "TIB_CART_LAT_G.inp", file_format="abaqus"
)
tibia_cartilage_lateral.plot()

# %% [markdown]
# # Data loading gameplan

# 1. Create point_data array mapping node numbers from _TIB_CART_LAT_G.inp_ to the 0-based
#    indexes of `tibia_cartilage_lateral`.
# 2. Preallocate point_data contact pressure array with zeros because contact pressure
#    is only output for surface nodes. Then, fill in the values you have.
# 3. Make your new point_data array the active scalar, then plot with colorbar.
# 4. Make an animation across all the frames to confirm that results make sense to
#    domain scientists (centralized blob on top of cartilage, without jerky movements)

# %%
