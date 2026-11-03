#!/usr/bin/env python3
"""Bind q36's Vulkan calls to the statically linked PS5 RADV ICD."""
from pathlib import Path
import re
import sys

source, output = map(Path, sys.argv[1:])
names = sorted(set(re.findall(r'\b(vk[A-Z]\w*)\s*\(', source.read_text())))
names.remove('vkCreateInstance')
names.remove('vkGetInstanceProcAddr')
(output / 'vk_functions.inc').write_text(''.join(f'Q36_VK({n})\n' for n in names))
header = '''#pragma once
#include <vulkan/vulkan.h>
PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance, const char *);
VkResult q36_ps5_create_instance(const VkInstanceCreateInfo *, const VkAllocationCallbacks *, VkInstance *);
#define vkGetInstanceProcAddr vk_icdGetInstanceProcAddr
#define vkCreateInstance q36_ps5_create_instance
'''
for name in names:
    header += f'extern PFN_{name} q36_ps5_{name};\n#define {name} q36_ps5_{name}\n'
(output / 'dispatch.h').write_text(header)
