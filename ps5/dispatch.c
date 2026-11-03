#include "dispatch.h"
#include <stdio.h>

#define Q36_VK(name) PFN_##name q36_ps5_##name;
#include "vk_functions.inc"
#undef Q36_VK

VkResult q36_ps5_create_instance(const VkInstanceCreateInfo *info,
                                const VkAllocationCallbacks *allocator,
                                VkInstance *instance) {
    PFN_vkCreateInstance create = (PFN_vkCreateInstance)
        vk_icdGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    if (!create) return VK_ERROR_INCOMPATIBLE_DRIVER;
    VkResult rc = create(info, allocator, instance);
    if (rc != VK_SUCCESS) return rc;

    /* An instance lookup also returns the device dispatch trampolines. */
    int missing = 0;
#define Q36_VK(name) \
    q36_ps5_##name = (PFN_##name)vk_icdGetInstanceProcAddr(*instance, #name); \
    if (!q36_ps5_##name) { fprintf(stderr, "q36: RADV missing %s\n", #name); missing = 1; }
#include "vk_functions.inc"
#undef Q36_VK
    if (!missing) return VK_SUCCESS;
    if (vkDestroyInstance) vkDestroyInstance(*instance, allocator);
    *instance = VK_NULL_HANDLE;
    return VK_ERROR_INCOMPATIBLE_DRIVER;
}
