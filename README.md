# glautolab

```
 ██████╗ ██╗      █████╗    ═══ ██╗   ██╗████████╗ ██████╗ ██╗      █████╗ ██████╗
██╔════╝ ██║     ██╔══██╗   ═══ ██║   ██║╚══██╔══╝██╔═══██╗██║     ██╔══██╗██╔══██╗
██║  ███╗██║     ███████║   ═══ ██║   ██║   ██║   ██║   ██║██║     ███████║██████╔╝
██║   ██║██║     ██╔══██║   ═══ ██║   ██║   ██║   ██║   ██║██║     ██╔══██║██╔══██╗
╚██████╔╝███████╗██║  ██║   ═══ ╚██████╔╝   ██║   ╚██████╔╝███████╗██║  ██║██║  ██║
 ╚═════╝ ╚══════╝╚═╝  ╚═╝   ═══  ╚═════╝    ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
```

GLA-UTOLAB — OpenGL visualization and automation.

## Prerequisites

- **Python 3.10**
- **uv** (recommended) — [Astral’s uv](https://docs.astral.sh/uv/) for fast,

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add uv to your PATH and reload your shell:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
uv --version
```

### 2. Create and activate a virtual environment

```bash
uv venv .venv-glautolab --python 3.10
source .venv-glautolab/bin/activate
```

### 3. Install dependecies

```bash
uv sync --active
```

### 4. Run the visualization

From the project root:

```bash
cd vizModule
python openglModule_v2.py
```

## Coordinate frames and link convention

1. **Extrinsic to LiDAR**: The extrinsic gives you where the LiDAR origin is in camera coordinates (and the robot). In our case the LiDAR is the origin, so to draw the axes of the camera you need the inverse:

```python
# Inverse of the extrinsic: camera pose expressed in LiDAR frame (robot convention)
R_cam_in_lidar = ext.R_robot.T                          # (3, 3)
t_cam_in_lidar = (-ext.R_robot.T @ ext.t_robot).ravel() # (3,)
```

2. **Car links**: Links are defined as transform from one object to another. For example, base_link to LiDAR means:

   - base_link is 1 meter below the LiDAR:

   ```
   LiDAR
   ↑ 1m
   base_link
   ```

   So: **LiDAR is +1 m in Z relative to base_link**.

   Standard ROS coordinates:

   - **X** forward  
   - **Y** left  
   - **Z** up  

   Example `base_link_to_lidar`:

   ```yaml
   R:
     data: [1, 0, 0, 0, 1, 0, 0, 0, 1]
   t:
     data: [0, 0, 1.0]
   ```

   Chain: **center_link → base_link → lidar → camera**

## Clean up

To remove the environment and uv cache and start over:

```bash
deactivate
rm -rf .venv-glautolab
uv cache clean
```
