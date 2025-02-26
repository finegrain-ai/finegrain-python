# comfyui-finegrain

[Finegrain API](https://api.finegrain.ai/doc/) ComfyUI custom nodes

## Installation

### Requirements

1. Ensure you have **Python 3.12** or later installed.

2. Ensure you have Git installed.

### Comfy Registry installation

The nodes are published at https://registry.comfy.org/publishers/finegrain/nodes/comfyui-finegrain.

1. Ensure you have Comfy CLI installed.

2. Install the custom nodes using Comfy CLI:

```bash
comfy node registry-install comfyui-finegrain
```

The above command should automatically install the nodes' requirements.
If it somehow doesn't, you can manually install them with:

```bash
# ensure you activated the python virtual environment used by ComfyUI
pip install -r custom_nodes/comfyui-finegrain/requirements.txt
```

Alternatively if you installed ComfyUI via the "Windows Standalone archive", you can install the requirements with:

```shell
.\python_embeded\Scripts\pip.exe install hatchling
.\python_embeded\Scripts\pip.exe install -r .\ComfyUI\custom_nodes\comfyui-finegrain\requirements.txt
```

### Manual installation

To manually install the nodes, you may alternatively do the following:

1. Download an archive of the nodes by cliking the "Download Latest" button at
https://registry.comfy.org/publishers/finegrain/nodes/comfyui-finegrain

2. Extract the archive:

```bash
unzip -d custom_nodes/comfyui-finegrain comfyui-finegrain.zip
rm comfyui-finegrain.zip
```

3. Install the nodes' requirements:

```bash
pip install -r custom_nodes/comfyui-finegrain/requirements.txt
```

Alternatively if you installed ComfyUI via the "Windows Standalone archive", you can install the requirements with:

```shell
.\python_embeded\Scripts\pip.exe install hatchling
.\python_embeded\Scripts\pip.exe install -r .\ComfyUI\custom_nodes\comfyui-finegrain\requirements.txt
```

## Workflow examples

> [!Note]
> All the below workflow examples were made using comfyui-finegrain v0.2.0.

### Prompt to erase

Instantly remove any object, along with its shadows and reflections, just by naming it.

![Prompt to erase workflow](assets/erase.webp?raw=true)

[Download the Prompt to erase workflow](assets/erase.json)

### Prompt to cutout

Instantly isolate any object in a photo into a perfect cutout, just by naming it.

![Prompt to cutout workflow](assets/cutout.webp?raw=true)

[Download the Prompt to cutout workflow](assets/cutout.json)

### Prompt to recolor

Instantly change the color of any object in a photo, even through occlusions, just by naming it.

![Prompt to recolor workflow](assets/recolor.webp?raw=true)

[Download the Prompt to recolor workflow](assets/recolor.json)

### Swap

Replace any object in a photo with another, recreating shadows and reflections so naturally it looks like the new object was always there — perfectly preserved in every detail.

![Swap workflow](assets/swap.webp?raw=true)

[Download the Swap workflow](assets/swap.json)

### Blend

Seamlessly integrate any object into a scene, recreating shadows and reflections for a result so natural it looks like it was always there — perfectly preserved in every detail.

![Blend workflow](assets/blend.webp?raw=true)

[Download the Blend workflow](assets/blend.json)

### Generate packshot

Generate Packshot – Transform any mundane photo into a stunning white-background image with a perfectly natural shadow.

![Generate packshot workflow](assets/packshot.webp?raw=true)

[Download the Generate packshot workflow](assets/packshot.json)

### Remove background

Remove Background – Our pixel-perfect, high-resolution take on a classic, effortlessly extracting the main object from its background.

![Remove background workflow](assets/removebg.webp?raw=true)

[Download the Remove background workflow](assets/removebg.json)
