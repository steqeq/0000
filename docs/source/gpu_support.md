# GPU Support


|GPU|Architecture|[Support Level](#Description)|
|--------------|----------------|---------------|
|Instinct MI250|CDNA2|Full|
|Instinct MI210|CDNA2|Full|
|Radeon RX 6900 XT|RDNA2|Noncommercial|
|Radeon RX 6600|RDNA2|HIP|
|Radeon™ R9 Fury|Fiji|Community|

## Description
GPU support levels in ROCm:
 * Full - AMD provides full support for all software that is part of ROCm
 * Noncommercial - AMD enables all software that is part of ROCm. However, commercial usage is not supported.
 * HIP - AMD supports the HIP Runtime only for these products
 * Community - Packages distributed by AMD have dropped support for these GPUs or never enabled support for the GPUs. Builds from source are not disabled. AMD encourages the open source community to enable functionality for these cards.