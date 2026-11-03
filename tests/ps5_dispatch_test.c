#include "dispatch.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static VkResult create_result;
static const char *missing;
static int destroyed, lookups;

static void unused_entry(void) {}

static VkResult create_instance(const VkInstanceCreateInfo *info,
                                const VkAllocationCallbacks *allocator,
                                VkInstance *instance) {
    (void)info;
    (void)allocator;
    if (create_result == VK_SUCCESS) *instance = (VkInstance)(uintptr_t)1;
    return create_result;
}

static void destroy_instance(VkInstance instance, const VkAllocationCallbacks *allocator) {
    (void)allocator;
    assert(instance == (VkInstance)(uintptr_t)1);
    destroyed++;
}

PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance instance, const char *name) {
    lookups++;
    if (!strcmp(name, "vkCreateInstance")) {
        assert(instance == VK_NULL_HANDLE);
        return (PFN_vkVoidFunction)create_instance;
    }
    assert(instance == (VkInstance)(uintptr_t)1);
    if (missing && !strcmp(name, missing)) return NULL;
    if (!strcmp(name, "vkDestroyInstance")) return (PFN_vkVoidFunction)destroy_instance;
    return unused_entry;
}

int main(void) {
    VkInstance instance = VK_NULL_HANDLE;
    create_result = VK_ERROR_OUT_OF_HOST_MEMORY;
    assert(vkCreateInstance(NULL, NULL, &instance) == VK_ERROR_OUT_OF_HOST_MEMORY);
    assert(lookups == 1 && instance == VK_NULL_HANDLE && destroyed == 0);
    create_result = VK_SUCCESS;
    missing = "vkQueueSubmit";
    assert(vkCreateInstance(NULL, NULL, &instance) == VK_ERROR_INCOMPATIBLE_DRIVER);
    assert(instance == VK_NULL_HANDLE && destroyed == 1);
    missing = NULL;
    assert(vkCreateInstance(NULL, NULL, &instance) == VK_SUCCESS);
    assert(instance != VK_NULL_HANDLE);
#define Q36_VK(name) assert(q36_ps5_##name != NULL);
#include "vk_functions.inc"
#undef Q36_VK
    vkDestroyInstance(instance, NULL);
    assert(destroyed == 2);
    puts("PS5 Vulkan dispatch tests passed");
    return 0;
}
