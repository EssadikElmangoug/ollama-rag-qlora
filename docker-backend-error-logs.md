2025-12-23 13:40:28.527 | 
2025-12-23 13:40:28.527 | ==========
2025-12-23 13:40:28.527 | == CUDA ==
2025-12-23 13:40:28.527 | ==========
2025-12-23 13:40:28.529 | 
2025-12-23 13:40:28.529 | CUDA Version 12.1.0
2025-12-23 13:40:28.530 | 
2025-12-23 13:40:28.530 | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2025-12-23 13:40:28.531 | 
2025-12-23 13:40:28.531 | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2025-12-23 13:40:28.531 | By pulling and using the container, you accept the terms and conditions of this license:
2025-12-23 13:40:28.531 | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2025-12-23 13:40:28.531 | 
2025-12-23 13:40:28.531 | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2025-12-23 13:40:28.538 | 
2025-12-23 13:40:28.538 | WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
2025-12-23 13:40:28.538 |    Use the NVIDIA Container Toolkit to start this container with GPU support; see
2025-12-23 13:40:28.538 |    https://docs.nvidia.com/datacenter/cloud-native/ .
2025-12-23 13:40:28.538 | 
2025-12-23 13:40:28.538 | *************************
2025-12-23 13:40:28.538 | ** DEPRECATION NOTICE! **
2025-12-23 13:40:28.538 | *************************
2025-12-23 13:40:28.538 | THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
2025-12-23 13:40:28.538 |     https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md
2025-12-23 13:40:28.538 | 
2025-12-23 13:40:32.010 | /app/qlora_trainer.py:15: UserWarning: WARNING: Unsloth should be imported before [transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.
2025-12-23 13:40:32.010 | 
2025-12-23 13:40:32.010 | Please restructure your imports with 'import unsloth' at the top of your file.
2025-12-23 13:40:32.010 |   import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:40:32.011 | Traceback (most recent call last):
2025-12-23 13:40:32.011 |   File "/app/app.py", line 13, in <module>
2025-12-23 13:40:32.011 |     from qlora_trainer import QLoRATrainer
2025-12-23 13:40:32.011 |   File "/app/qlora_trainer.py", line 15, in <module>
2025-12-23 13:40:32.011 |     import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:40:32.011 |   File "/usr/local/lib/python3.10/dist-packages/unsloth/__init__.py", line 92, in <module>
2025-12-23 13:40:32.011 |     import unsloth_zoo
2025-12-23 13:40:32.011 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/__init__.py", line 146, in <module>
2025-12-23 13:40:32.011 |     from .device_type import (
2025-12-23 13:40:32.011 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 56, in <module>
2025-12-23 13:40:32.011 |     DEVICE_TYPE : str = get_device_type()
2025-12-23 13:40:32.011 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 46, in get_device_type
2025-12-23 13:40:32.011 |     raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
2025-12-23 13:40:32.011 | NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
2025-12-23 13:40:33.447 | 
2025-12-23 13:40:33.447 | ==========
2025-12-23 13:40:33.447 | == CUDA ==
2025-12-23 13:40:33.447 | ==========
2025-12-23 13:40:33.449 | 
2025-12-23 13:40:33.449 | CUDA Version 12.1.0
2025-12-23 13:40:33.450 | 
2025-12-23 13:40:33.450 | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2025-12-23 13:40:33.451 | 
2025-12-23 13:40:33.451 | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2025-12-23 13:40:33.451 | By pulling and using the container, you accept the terms and conditions of this license:
2025-12-23 13:40:33.451 | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2025-12-23 13:40:33.451 | 
2025-12-23 13:40:33.451 | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2025-12-23 13:40:33.456 | 
2025-12-23 13:40:33.456 | WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
2025-12-23 13:40:33.456 |    Use the NVIDIA Container Toolkit to start this container with GPU support; see
2025-12-23 13:40:33.456 |    https://docs.nvidia.com/datacenter/cloud-native/ .
2025-12-23 13:40:33.456 | 
2025-12-23 13:40:33.456 | *************************
2025-12-23 13:40:33.456 | ** DEPRECATION NOTICE! **
2025-12-23 13:40:33.456 | *************************
2025-12-23 13:40:33.456 | THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
2025-12-23 13:40:33.456 |     https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md
2025-12-23 13:40:33.456 | 
2025-12-23 13:40:36.843 | /app/qlora_trainer.py:15: UserWarning: WARNING: Unsloth should be imported before [transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.
2025-12-23 13:40:36.843 | 
2025-12-23 13:40:36.843 | Please restructure your imports with 'import unsloth' at the top of your file.
2025-12-23 13:40:36.843 |   import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:40:36.844 | Traceback (most recent call last):
2025-12-23 13:40:36.844 |   File "/app/app.py", line 13, in <module>
2025-12-23 13:40:36.844 |     from qlora_trainer import QLoRATrainer
2025-12-23 13:40:36.844 |   File "/app/qlora_trainer.py", line 15, in <module>
2025-12-23 13:40:36.844 |     import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:40:36.844 |   File "/usr/local/lib/python3.10/dist-packages/unsloth/__init__.py", line 92, in <module>
2025-12-23 13:40:36.844 |     import unsloth_zoo
2025-12-23 13:40:36.844 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/__init__.py", line 146, in <module>
2025-12-23 13:40:36.844 |     from .device_type import (
2025-12-23 13:40:36.844 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 56, in <module>
2025-12-23 13:40:36.844 |     DEVICE_TYPE : str = get_device_type()
2025-12-23 13:40:36.844 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 46, in get_device_type
2025-12-23 13:40:36.844 |     raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
2025-12-23 13:40:36.844 | NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
2025-12-23 13:40:38.127 | 
2025-12-23 13:40:38.127 | ==========
2025-12-23 13:40:38.127 | == CUDA ==
2025-12-23 13:40:38.127 | ==========
2025-12-23 13:40:38.129 | 
2025-12-23 13:40:38.129 | CUDA Version 12.1.0
2025-12-23 13:40:38.130 | 
2025-12-23 13:40:38.130 | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2025-12-23 13:40:38.130 | 
2025-12-23 13:40:38.130 | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2025-12-23 13:40:38.130 | By pulling and using the container, you accept the terms and conditions of this license:
2025-12-23 13:40:38.130 | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2025-12-23 13:40:38.130 | 
2025-12-23 13:40:38.130 | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2025-12-23 13:40:38.137 | 
2025-12-23 13:40:38.137 | WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
2025-12-23 13:40:38.137 |    Use the NVIDIA Container Toolkit to start this container with GPU support; see
2025-12-23 13:40:38.137 |    https://docs.nvidia.com/datacenter/cloud-native/ .
2025-12-23 13:40:38.137 | 
2025-12-23 13:40:38.137 | *************************
2025-12-23 13:40:38.137 | ** DEPRECATION NOTICE! **
2025-12-23 13:40:38.137 | *************************
2025-12-23 13:40:38.137 | THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
2025-12-23 13:40:38.137 |     https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md
2025-12-23 13:40:38.137 | 
2025-12-23 13:40:41.593 | /app/qlora_trainer.py:15: UserWarning: WARNING: Unsloth should be imported before [transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.
2025-12-23 13:40:41.593 | 
2025-12-23 13:40:41.593 | Please restructure your imports with 'import unsloth' at the top of your file.
2025-12-23 13:40:41.593 |   import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:40:41.594 | Traceback (most recent call last):
2025-12-23 13:40:41.594 |   File "/app/app.py", line 13, in <module>
2025-12-23 13:40:41.594 |     from qlora_trainer import QLoRATrainer
2025-12-23 13:40:41.594 |   File "/app/qlora_trainer.py", line 15, in <module>
2025-12-23 13:40:41.594 |     import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:40:41.594 |   File "/usr/local/lib/python3.10/dist-packages/unsloth/__init__.py", line 92, in <module>
2025-12-23 13:40:41.594 |     import unsloth_zoo
2025-12-23 13:40:41.594 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/__init__.py", line 146, in <module>
2025-12-23 13:40:41.595 |     from .device_type import (
2025-12-23 13:40:41.595 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 56, in <module>
2025-12-23 13:40:41.595 |     DEVICE_TYPE : str = get_device_type()
2025-12-23 13:40:41.595 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 46, in get_device_type
2025-12-23 13:40:41.595 |     raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
2025-12-23 13:40:41.595 | NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
2025-12-23 13:40:43.015 | 
2025-12-23 13:40:43.015 | ==========
2025-12-23 13:40:43.015 | == CUDA ==
2025-12-23 13:40:43.015 | ==========
2025-12-23 13:40:43.017 | 
2025-12-23 13:40:43.017 | CUDA Version 12.1.0
2025-12-23 13:40:43.017 | 
2025-12-23 13:40:43.017 | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2025-12-23 13:40:43.018 | 
2025-12-23 13:40:43.018 | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2025-12-23 13:40:43.018 | By pulling and using the container, you accept the terms and conditions of this license:
2025-12-23 13:40:43.018 | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2025-12-23 13:40:43.018 | 
2025-12-23 13:40:43.018 | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2025-12-23 13:40:43.025 | 
2025-12-23 13:40:43.025 | WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
2025-12-23 13:40:43.025 |    Use the NVIDIA Container Toolkit to start this container with GPU support; see
2025-12-23 13:40:43.025 |    https://docs.nvidia.com/datacenter/cloud-native/ .
2025-12-23 13:40:43.025 | 
2025-12-23 13:40:43.025 | *************************
2025-12-23 13:40:43.025 | ** DEPRECATION NOTICE! **
2025-12-23 13:40:43.025 | *************************
2025-12-23 13:40:43.025 | THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
2025-12-23 13:40:43.025 |     https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md
2025-12-23 13:40:43.025 | 
2025-12-23 13:40:46.443 | /app/qlora_trainer.py:15: UserWarning: WARNING: Unsloth should be imported before [transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.
2025-12-23 13:40:46.443 | 
2025-12-23 13:40:46.443 | Please restructure your imports with 'import unsloth' at the top of your file.
2025-12-23 13:40:46.443 |   import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:40:46.444 | Traceback (most recent call last):
2025-12-23 13:40:46.444 |   File "/app/app.py", line 13, in <module>
2025-12-23 13:40:46.444 |     from qlora_trainer import QLoRATrainer
2025-12-23 13:40:46.444 |   File "/app/qlora_trainer.py", line 15, in <module>
2025-12-23 13:40:46.444 |     import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:40:46.444 |   File "/usr/local/lib/python3.10/dist-packages/unsloth/__init__.py", line 92, in <module>
2025-12-23 13:40:46.445 |     import unsloth_zoo
2025-12-23 13:40:46.445 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/__init__.py", line 146, in <module>
2025-12-23 13:40:46.445 |     from .device_type import (
2025-12-23 13:40:46.445 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 56, in <module>
2025-12-23 13:40:46.445 |     DEVICE_TYPE : str = get_device_type()
2025-12-23 13:40:46.445 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 46, in get_device_type
2025-12-23 13:40:46.445 |     raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
2025-12-23 13:40:46.445 | NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
2025-12-23 13:40:48.333 | 
2025-12-23 13:40:48.333 | ==========
2025-12-23 13:40:48.333 | == CUDA ==
2025-12-23 13:40:48.333 | ==========
2025-12-23 13:40:48.335 | 
2025-12-23 13:40:48.335 | CUDA Version 12.1.0
2025-12-23 13:40:48.336 | 
2025-12-23 13:40:48.336 | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2025-12-23 13:40:48.337 | 
2025-12-23 13:40:48.337 | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2025-12-23 13:40:48.337 | By pulling and using the container, you accept the terms and conditions of this license:
2025-12-23 13:40:48.337 | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2025-12-23 13:40:48.337 | 
2025-12-23 13:40:48.337 | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2025-12-23 13:40:48.344 | 
2025-12-23 13:40:48.344 | WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
2025-12-23 13:40:48.344 |    Use the NVIDIA Container Toolkit to start this container with GPU support; see
2025-12-23 13:40:48.344 |    https://docs.nvidia.com/datacenter/cloud-native/ .
2025-12-23 13:40:48.344 | 
2025-12-23 13:40:48.344 | *************************
2025-12-23 13:40:48.344 | ** DEPRECATION NOTICE! **
2025-12-23 13:40:48.344 | *************************
2025-12-23 13:40:48.344 | THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
2025-12-23 13:40:48.344 |     https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md
2025-12-23 13:40:48.344 | 
2025-12-23 13:40:51.729 | /app/qlora_trainer.py:15: UserWarning: WARNING: Unsloth should be imported before [transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.
2025-12-23 13:40:51.729 | 
2025-12-23 13:40:51.729 | Please restructure your imports with 'import unsloth' at the top of your file.
2025-12-23 13:40:51.729 |   import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:40:51.730 | Traceback (most recent call last):
2025-12-23 13:40:51.730 |   File "/app/app.py", line 13, in <module>
2025-12-23 13:40:51.730 |     from qlora_trainer import QLoRATrainer
2025-12-23 13:40:51.730 |   File "/app/qlora_trainer.py", line 15, in <module>
2025-12-23 13:40:51.730 |     import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:40:51.730 |   File "/usr/local/lib/python3.10/dist-packages/unsloth/__init__.py", line 92, in <module>
2025-12-23 13:40:51.730 |     import unsloth_zoo
2025-12-23 13:40:51.730 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/__init__.py", line 146, in <module>
2025-12-23 13:40:51.730 |     from .device_type import (
2025-12-23 13:40:51.730 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 56, in <module>
2025-12-23 13:40:51.730 |     DEVICE_TYPE : str = get_device_type()
2025-12-23 13:40:51.730 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 46, in get_device_type
2025-12-23 13:40:51.730 |     raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
2025-12-23 13:40:51.730 | NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
2025-12-23 13:40:54.386 | 
2025-12-23 13:40:54.386 | ==========
2025-12-23 13:40:54.386 | == CUDA ==
2025-12-23 13:40:54.386 | ==========
2025-12-23 13:40:54.388 | 
2025-12-23 13:40:54.389 | CUDA Version 12.1.0
2025-12-23 13:40:54.390 | 
2025-12-23 13:40:54.390 | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2025-12-23 13:40:54.391 | 
2025-12-23 13:40:54.391 | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2025-12-23 13:40:54.391 | By pulling and using the container, you accept the terms and conditions of this license:
2025-12-23 13:40:54.391 | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2025-12-23 13:40:54.391 | 
2025-12-23 13:40:54.391 | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2025-12-23 13:40:54.398 | 
2025-12-23 13:40:54.398 | WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
2025-12-23 13:40:54.398 |    Use the NVIDIA Container Toolkit to start this container with GPU support; see
2025-12-23 13:40:54.398 |    https://docs.nvidia.com/datacenter/cloud-native/ .
2025-12-23 13:40:54.398 | 
2025-12-23 13:40:54.398 | *************************
2025-12-23 13:40:54.398 | ** DEPRECATION NOTICE! **
2025-12-23 13:40:54.398 | *************************
2025-12-23 13:40:54.398 | THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
2025-12-23 13:40:54.398 |     https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md
2025-12-23 13:40:54.398 | 
2025-12-23 13:40:57.825 | /app/qlora_trainer.py:15: UserWarning: WARNING: Unsloth should be imported before [transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.
2025-12-23 13:40:57.825 | 
2025-12-23 13:40:57.825 | Please restructure your imports with 'import unsloth' at the top of your file.
2025-12-23 13:40:57.825 |   import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:40:57.826 | Traceback (most recent call last):
2025-12-23 13:40:57.826 |   File "/app/app.py", line 13, in <module>
2025-12-23 13:40:57.826 |     from qlora_trainer import QLoRATrainer
2025-12-23 13:40:57.826 |   File "/app/qlora_trainer.py", line 15, in <module>
2025-12-23 13:40:57.826 |     import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:40:57.826 |   File "/usr/local/lib/python3.10/dist-packages/unsloth/__init__.py", line 92, in <module>
2025-12-23 13:40:57.826 |     import unsloth_zoo
2025-12-23 13:40:57.826 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/__init__.py", line 146, in <module>
2025-12-23 13:40:57.826 |     from .device_type import (
2025-12-23 13:40:57.826 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 56, in <module>
2025-12-23 13:40:57.826 |     DEVICE_TYPE : str = get_device_type()
2025-12-23 13:40:57.826 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 46, in get_device_type
2025-12-23 13:40:57.826 |     raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
2025-12-23 13:40:57.826 | NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
2025-12-23 13:41:02.074 | 
2025-12-23 13:41:02.074 | ==========
2025-12-23 13:41:02.074 | == CUDA ==
2025-12-23 13:41:02.074 | ==========
2025-12-23 13:41:02.075 | 
2025-12-23 13:41:02.075 | CUDA Version 12.1.0
2025-12-23 13:41:02.076 | 
2025-12-23 13:41:02.076 | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2025-12-23 13:41:02.077 | 
2025-12-23 13:41:02.077 | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2025-12-23 13:41:02.077 | By pulling and using the container, you accept the terms and conditions of this license:
2025-12-23 13:41:02.077 | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2025-12-23 13:41:02.077 | 
2025-12-23 13:41:02.077 | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2025-12-23 13:41:02.083 | 
2025-12-23 13:41:02.083 | WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
2025-12-23 13:41:02.083 |    Use the NVIDIA Container Toolkit to start this container with GPU support; see
2025-12-23 13:41:02.083 |    https://docs.nvidia.com/datacenter/cloud-native/ .
2025-12-23 13:41:02.083 | 
2025-12-23 13:41:02.083 | *************************
2025-12-23 13:41:02.083 | ** DEPRECATION NOTICE! **
2025-12-23 13:41:02.083 | *************************
2025-12-23 13:41:02.083 | THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
2025-12-23 13:41:02.083 |     https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md
2025-12-23 13:41:02.083 | 
2025-12-23 13:41:05.441 | /app/qlora_trainer.py:15: UserWarning: WARNING: Unsloth should be imported before [transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.
2025-12-23 13:41:05.441 | 
2025-12-23 13:41:05.441 | Please restructure your imports with 'import unsloth' at the top of your file.
2025-12-23 13:41:05.441 |   import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:41:05.442 | Traceback (most recent call last):
2025-12-23 13:41:05.442 |   File "/app/app.py", line 13, in <module>
2025-12-23 13:41:05.442 |     from qlora_trainer import QLoRATrainer
2025-12-23 13:41:05.442 |   File "/app/qlora_trainer.py", line 15, in <module>
2025-12-23 13:41:05.442 |     import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:41:05.442 |   File "/usr/local/lib/python3.10/dist-packages/unsloth/__init__.py", line 92, in <module>
2025-12-23 13:41:05.442 |     import unsloth_zoo
2025-12-23 13:41:05.442 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/__init__.py", line 146, in <module>
2025-12-23 13:41:05.443 |     from .device_type import (
2025-12-23 13:41:05.443 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 56, in <module>
2025-12-23 13:41:05.443 |     DEVICE_TYPE : str = get_device_type()
2025-12-23 13:41:05.443 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 46, in get_device_type
2025-12-23 13:41:05.443 |     raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
2025-12-23 13:41:05.443 | NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
2025-12-23 13:41:12.886 | 
2025-12-23 13:41:12.886 | ==========
2025-12-23 13:41:12.886 | == CUDA ==
2025-12-23 13:41:12.886 | ==========
2025-12-23 13:41:12.888 | 
2025-12-23 13:41:12.888 | CUDA Version 12.1.0
2025-12-23 13:41:12.889 | 
2025-12-23 13:41:12.889 | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2025-12-23 13:41:12.889 | 
2025-12-23 13:41:12.889 | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2025-12-23 13:41:12.889 | By pulling and using the container, you accept the terms and conditions of this license:
2025-12-23 13:41:12.889 | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2025-12-23 13:41:12.889 | 
2025-12-23 13:41:12.889 | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2025-12-23 13:41:12.895 | 
2025-12-23 13:41:12.895 | WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
2025-12-23 13:41:12.895 |    Use the NVIDIA Container Toolkit to start this container with GPU support; see
2025-12-23 13:41:12.895 |    https://docs.nvidia.com/datacenter/cloud-native/ .
2025-12-23 13:41:12.895 | 
2025-12-23 13:41:12.895 | *************************
2025-12-23 13:41:12.895 | ** DEPRECATION NOTICE! **
2025-12-23 13:41:12.895 | *************************
2025-12-23 13:41:12.895 | THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
2025-12-23 13:41:12.895 |     https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md
2025-12-23 13:41:12.895 | 
2025-12-23 13:41:16.325 | /app/qlora_trainer.py:15: UserWarning: WARNING: Unsloth should be imported before [transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.
2025-12-23 13:41:16.325 | 
2025-12-23 13:41:16.325 | Please restructure your imports with 'import unsloth' at the top of your file.
2025-12-23 13:41:16.325 |   import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:41:16.327 | Traceback (most recent call last):
2025-12-23 13:41:16.327 |   File "/app/app.py", line 13, in <module>
2025-12-23 13:41:16.327 |     from qlora_trainer import QLoRATrainer
2025-12-23 13:41:16.327 |   File "/app/qlora_trainer.py", line 15, in <module>
2025-12-23 13:41:16.327 |     import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:41:16.327 |   File "/usr/local/lib/python3.10/dist-packages/unsloth/__init__.py", line 92, in <module>
2025-12-23 13:41:16.327 |     import unsloth_zoo
2025-12-23 13:41:16.327 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/__init__.py", line 146, in <module>
2025-12-23 13:41:16.327 |     from .device_type import (
2025-12-23 13:41:16.327 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 56, in <module>
2025-12-23 13:41:16.327 |     DEVICE_TYPE : str = get_device_type()
2025-12-23 13:41:16.327 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 46, in get_device_type
2025-12-23 13:41:16.327 |     raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
2025-12-23 13:41:16.327 | NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
2025-12-23 13:41:30.168 | 
2025-12-23 13:41:30.168 | ==========
2025-12-23 13:41:30.168 | == CUDA ==
2025-12-23 13:41:30.168 | ==========
2025-12-23 13:41:30.170 | 
2025-12-23 13:41:30.170 | CUDA Version 12.1.0
2025-12-23 13:41:30.171 | 
2025-12-23 13:41:30.171 | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2025-12-23 13:41:30.172 | 
2025-12-23 13:41:30.172 | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2025-12-23 13:41:30.172 | By pulling and using the container, you accept the terms and conditions of this license:
2025-12-23 13:41:30.172 | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2025-12-23 13:41:30.172 | 
2025-12-23 13:41:30.172 | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2025-12-23 13:41:30.177 | 
2025-12-23 13:41:30.178 | WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
2025-12-23 13:41:30.178 |    Use the NVIDIA Container Toolkit to start this container with GPU support; see
2025-12-23 13:41:30.178 |    https://docs.nvidia.com/datacenter/cloud-native/ .
2025-12-23 13:41:30.178 | 
2025-12-23 13:41:30.178 | *************************
2025-12-23 13:41:30.178 | ** DEPRECATION NOTICE! **
2025-12-23 13:41:30.178 | *************************
2025-12-23 13:41:30.178 | THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
2025-12-23 13:41:30.178 |     https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md
2025-12-23 13:41:30.178 | 
2025-12-23 13:41:33.500 | /app/qlora_trainer.py:15: UserWarning: WARNING: Unsloth should be imported before [transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.
2025-12-23 13:41:33.501 | 
2025-12-23 13:41:33.501 | Please restructure your imports with 'import unsloth' at the top of your file.
2025-12-23 13:41:33.501 |   import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:41:33.502 | Traceback (most recent call last):
2025-12-23 13:41:33.502 |   File "/app/app.py", line 13, in <module>
2025-12-23 13:41:33.502 |     from qlora_trainer import QLoRATrainer
2025-12-23 13:41:33.502 |   File "/app/qlora_trainer.py", line 15, in <module>
2025-12-23 13:41:33.502 |     import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:41:33.502 |   File "/usr/local/lib/python3.10/dist-packages/unsloth/__init__.py", line 92, in <module>
2025-12-23 13:41:33.502 |     import unsloth_zoo
2025-12-23 13:41:33.502 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/__init__.py", line 146, in <module>
2025-12-23 13:41:33.502 |     from .device_type import (
2025-12-23 13:41:33.502 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 56, in <module>
2025-12-23 13:41:33.502 |     DEVICE_TYPE : str = get_device_type()
2025-12-23 13:41:33.502 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 46, in get_device_type
2025-12-23 13:41:33.502 |     raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
2025-12-23 13:41:33.502 | NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
2025-12-23 13:42:00.139 | 
2025-12-23 13:42:00.139 | ==========
2025-12-23 13:42:00.139 | == CUDA ==
2025-12-23 13:42:00.139 | ==========
2025-12-23 13:42:00.142 | 
2025-12-23 13:42:00.142 | CUDA Version 12.1.0
2025-12-23 13:42:00.142 | 
2025-12-23 13:42:00.142 | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2025-12-23 13:42:00.143 | 
2025-12-23 13:42:00.143 | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
2025-12-23 13:42:00.143 | By pulling and using the container, you accept the terms and conditions of this license:
2025-12-23 13:42:00.143 | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
2025-12-23 13:42:00.143 | 
2025-12-23 13:42:00.143 | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
2025-12-23 13:42:00.148 | 
2025-12-23 13:42:00.148 | WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
2025-12-23 13:42:00.148 |    Use the NVIDIA Container Toolkit to start this container with GPU support; see
2025-12-23 13:42:00.148 |    https://docs.nvidia.com/datacenter/cloud-native/ .
2025-12-23 13:42:00.148 | 
2025-12-23 13:42:00.148 | *************************
2025-12-23 13:42:00.148 | ** DEPRECATION NOTICE! **
2025-12-23 13:42:00.148 | *************************
2025-12-23 13:42:00.148 | THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
2025-12-23 13:42:00.148 |     https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md
2025-12-23 13:42:00.148 | 
2025-12-23 13:42:03.734 | /app/qlora_trainer.py:15: UserWarning: WARNING: Unsloth should be imported before [transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.
2025-12-23 13:42:03.734 | 
2025-12-23 13:42:03.734 | Please restructure your imports with 'import unsloth' at the top of your file.
2025-12-23 13:42:03.734 |   import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:42:03.735 | Traceback (most recent call last):
2025-12-23 13:42:03.736 |   File "/app/app.py", line 13, in <module>
2025-12-23 13:42:03.736 |     from qlora_trainer import QLoRATrainer
2025-12-23 13:42:03.736 |   File "/app/qlora_trainer.py", line 15, in <module>
2025-12-23 13:42:03.736 |     import unsloth  # This triggers the patching and cache creation early
2025-12-23 13:42:03.736 |   File "/usr/local/lib/python3.10/dist-packages/unsloth/__init__.py", line 92, in <module>
2025-12-23 13:42:03.736 |     import unsloth_zoo
2025-12-23 13:42:03.736 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/__init__.py", line 146, in <module>
2025-12-23 13:42:03.736 |     from .device_type import (
2025-12-23 13:42:03.736 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 56, in <module>
2025-12-23 13:42:03.736 |     DEVICE_TYPE : str = get_device_type()
2025-12-23 13:42:03.736 |   File "/usr/local/lib/python3.10/dist-packages/unsloth_zoo/device_type.py", line 46, in get_device_type
2025-12-23 13:42:03.736 |     raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
2025-12-23 13:42:03.736 | NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.