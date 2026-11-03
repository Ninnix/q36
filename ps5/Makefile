# Reuse the engine's shader recipes without touching its host build outputs.
include $(ROOT)/Makefile
VPATH := $(ROOT)
.DEFAULT_GOAL := objects
PS5_FLAGS := -O2 -march=znver2 -std=c11 -D_GNU_SOURCE -fno-finite-math-only \
    -ffunction-sections -fdata-sections -Wall -Wextra -Wno-unreachable-code-generic-assoc \
    -I$(ROOT) -I. -I$(HEADERS)
PS5_SOURCES := q36 q36_cli q36_vulkan q36_image q36_ssd q36_prompt_prefix q36_distributed linenoise
PS5_OBJECTS := $(addprefix engine-,$(addsuffix .o,$(PS5_SOURCES)))

.PHONY: objects
objects: $(PS5_OBJECTS) main.o dispatch.o

engine-%.o: $(ROOT)/%.c
	$(CC) $(PS5_FLAGS) -c $< -o $@

engine-q36_cli.o: $(ROOT)/q36_cli.c
	$(CC) $(PS5_FLAGS) -Dmain=q36_cli_main -c $< -o $@

engine-q36_vulkan.o: $(ROOT)/q36_vulkan.c dispatch.h $(VULKAN_SHADERS)
	$(CC) $(PS5_FLAGS) -include dispatch.h -c $< -o $@

main.o: $(ROOT)/ps5/main.c
	$(CC) $(PS5_FLAGS) -DQ36_REVISION='"$(REVISION)"' -c $< -o $@

dispatch.o: $(ROOT)/ps5/dispatch.c dispatch.h vk_functions.inc
	$(CC) $(PS5_FLAGS) -c $< -o $@
