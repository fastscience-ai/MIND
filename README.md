# LangChain MCP Adapters Setup

This guide explains how to set up a Conda environment, install JupyterLab, and install `langchain-mcp-adapters` for developing LangChain + MCP-based applications.

---

## 1. Create a Conda Environment

If you already have a suitable environment, you can skip this step.

```bash
conda create -n mcp python=3.10 -y
conda activate mcp
```


## 2. Upgrade pip and Build Tools
Before installing the packages, ensure your environment's build tools are up to date to avoid compatibility issues.

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 2. Install JupyterLab
Install JupyterLab to provide an interactive environment for testing your LangChain agents.

```bash
pip install jupyterlab
```

## 3. Install langchain-mcp-adapters
Install the adapter library required to connect LangChain to MCP servers.

```bash
pip install langchain-mcp-adapters
```

### 4. Verify Installation
Run the following script to confirm that the MCPToolkit is correctly installed and accessible.

```bash
python - <<EOF
from langchain_mcp_adapters import MCPToolkit
print("MCP adapter installed successfully")
```
