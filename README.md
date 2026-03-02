# glautolab

 ██████╗ ██╗      █████╗    ═══ ██╗   ██╗████████╗ ██████╗ ██╗      █████╗ ██████╗
██╔════╝ ██║     ██╔══██╗   ═══ ██║   ██║╚══██╔══╝██╔═══██╗██║     ██╔══██╗██╔══██╗
██║  ███╗██║     ███████║   ═══ ██║   ██║   ██║   ██║   ██║██║     ███████║██████╔╝
██║   ██║██║     ██╔══██║   ═══ ██║   ██║   ██║   ██║   ██║██║     ██╔══██║██╔══██╗
╚██████╔╝███████╗██║  ██║   ═══ ╚██████╔╝   ██║   ╚██████╔╝███████╗██║  ██║██║  ██║
 ╚═════╝ ╚══════╝╚═╝  ╚═╝   ═══  ╚═════╝    ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝

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

On Windows (PowerShell): `.venv-glautolab\Scripts\activate`

### 3. Install dependencies (CUDA 12.1 / GPU)

For GPU support with CUDA 12.1, install in this order:

```bash
uv pip uninstall numpy -y
uv pip install numpy==1.26.4

```

**CPU-only or after syncing from lockfile:** from the project root run:

```bash
uv sync
```

## Clean up

To remove the environment and uv cache and start over:

```bash
deactivate
rm -rf .venv-glautolab
uv cache clean
```
