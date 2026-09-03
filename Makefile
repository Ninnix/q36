CC ?= cc
CXX ?= c++
GLSLC ?= ./glslc
.DEFAULT_GOAL := all
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
# Release-safe Apple Silicon baseline. Local-only builds may override this with
# NATIVE_CPU_FLAG=-mcpu=native, but published binaries must remain M1-capable.
NATIVE_CPU_FLAG ?= -mcpu=apple-m1
MACOSX_DEPLOYMENT_TARGET ?= 11.0
DARWIN_MIN_FLAG := -mmacosx-version-min=$(MACOSX_DEPLOYMENT_TARGET)
else
NATIVE_CPU_FLAG ?= -march=native
DARWIN_MIN_FLAG :=
endif

CFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
LLAMA_CPP_DIR ?= llama.cpp
LLAMA_BUILD_DIR ?= $(LLAMA_CPP_DIR)/build
REFERENCE_MODEL ?= gguf/Qwen3.6-35B-A3B-Q8_0.gguf
QWEN27B_MODEL ?= gguf/Qwen3.8-27B-UD-IQ3_S.gguf
QWEN27B_PROMPT ?= tests/long_context_story_prompt.txt
QWEN27B_GEN ?= 128
QWEN27B_GATE_GEN ?= 32
QWEN27B_LLAMA_PROMPT ?= 64
QWEN27B_LLAMA_BENCH ?= $(LLAMA_CPP_DIR)/build-vulkan/bin/llama-bench
SAMPLING_TEST := tests/test_sampling
OPENROUTER_MODEL ?= qwen/qwen3.6-35b-a3b
OPENROUTER_OUT ?= gguf-tools/quality-testing/local/openrouter
OPENROUTER_COUNT ?= 100
LLAMA_INCLUDE := -I"$(LLAMA_CPP_DIR)/include" -I"$(LLAMA_CPP_DIR)/ggml/include"
LLAMA_LIBS := \
	"$(LLAMA_BUILD_DIR)/src/libllama.a" \
	"$(LLAMA_BUILD_DIR)/ggml/src/libggml.a" \
	"$(LLAMA_BUILD_DIR)/ggml/src/libggml-cpu.a" \
	"$(LLAMA_BUILD_DIR)/ggml/src/libggml-base.a"
LLAMA_LDLIBS := -lstdc++ -fopenmp
CPU_CFLAGS := $(filter-out -ffast-math,$(CFLAGS)) $(DARWIN_MIN_FLAG) -D_GNU_SOURCE -fno-finite-math-only -DQ36_NO_GPU
GPU_CFLAGS := $(CFLAGS) $(DARWIN_MIN_FLAG) -D_GNU_SOURCE -fno-finite-math-only
VULKAN_CFLAGS ?=
LDLIBS ?= -lm -pthread
GPU_LDLIBS := $(LDLIBS) -ldl -lvulkan
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
METAL_LDFLAGS := $(LDFLAGS) $(DARWIN_MIN_FLAG)
METAL_SRCS := $(wildcard metal/*.metal)
VULKAN_SHADERS := \
	vulkan/matmul_f16.spv \
	vulkan/vision_matmul_f16.spv \
	vulkan/vision_attention.spv \
	vulkan/matmul_f32.spv \
	vulkan/matmul_f32_fast.spv \
	vulkan/add.spv \
	vulkan/directional_steering.spv \
	vulkan/add_rms_norm.spv \
	vulkan/rms_norm.spv \
	vulkan/swiglu.spv \
	vulkan/swiglu_q8_k.spv \
	vulkan/rope_qwen.spv \
	vulkan/rope_qwen_mrope.spv \
	vulkan/rms_norm_rope_qwen.spv \
	vulkan/rms_norm_rope_kv_qwen.spv \
	vulkan/copy_rows.spv \
	vulkan/recur_window.spv \
	vulkan/conv_silu.spv \
	vulkan/recur_conv_silu_decode.spv \
	vulkan/recur_norm_gate.spv \
	vulkan/recur_norm_gate_q8_k.spv \
	vulkan/delta_qk.spv \
	vulkan/delta_qkv.spv \
	vulkan/delta_gates.spv \
	vulkan/quantize_q8_0.spv \
	vulkan/quantize_q8_k.spv \
	vulkan/predequant_b16.spv \
	vulkan/matmul_q8_0.spv \
	vulkan/matmul_q8_0_q36.spv \
	vulkan/matmul_q8_0_f32b.spv \
	vulkan/matmul_q8_0_f32b_pair.spv \
	vulkan/shared_gate_up_decode.spv \
	vulkan/shared_down_tail_decode.spv \
	vulkan/matmul_q8_0_mm.spv \
	vulkan/matmul_q8_0_mm_f16.spv \
	vulkan/matmul_q8_0_mm_f16_out32.spv \
	vulkan/matmul_q8_0_f32b_nx.spv \
	vulkan/matmul_q8_0_decode.spv \
	vulkan/matmul_q8_0_decode_q36.spv \
	vulkan/matmul_q8_0_decode_b64.spv \
	vulkan/matmul_kquant.spv \
	vulkan/matmul_kquant_mmq.spv \
	vulkan/matmul_q5k_mmq.spv \
	vulkan/matmul_q6k_mmq.spv \
	vulkan/matmul_q5k_mmq_fast.spv \
	vulkan/matmul_q6k_mmq_fast.spv \
	vulkan/delta_net.spv \
	vulkan/delta_net_fast.spv \
	vulkan/delta_net_cols.spv \
	vulkan/delta_net_cols_f16.spv \
	vulkan/delta_net_decode.spv \
	vulkan/delta_net_decode_reg.spv \
	vulkan/delta_net_decode_reg_f16.spv \
	vulkan/attn_scores.spv \
	vulkan/attn_post.spv \
	vulkan/attn_reduce.spv \
	vulkan/attn_decode_fused.spv \
	vulkan/attn_decode_split.spv \
	vulkan/attn_prefill_qtile.spv \
	vulkan/attn_prefill_qtile2.spv \
	vulkan/attn_combine.spv \
	vulkan/moe_gate_up.spv \
	vulkan/router_topk.spv \
	vulkan/moe_tiles.spv \
	vulkan/kv_store.spv \
	vulkan/kv_store_quant.spv \
	vulkan/rms_norm_rope_kv_qwen_quant.spv \
	vulkan/top2.spv \
	vulkan/moe_gate_up_f32b.spv \
	vulkan/moe_gate_up_decode.spv \
	vulkan/moe_down_q2k_f32b.spv \
	vulkan/moe_down_q2k_sum_decode.spv \
	vulkan/moe_gate_up_q4k_f32b.spv \
	vulkan/moe_down_q4k_sum_decode.spv \
	vulkan/moe_gate_up_gemm.spv \
	vulkan/moe_down_gemm.spv \
	vulkan/moe_matvec.spv \
	vulkan/moe_matvec_fast.spv \
	vulkan/dense_iq3_xxs_decode.spv \
	vulkan/dense_iq3_xxs_decode_r4.spv \
	vulkan/dense_iq3_xxs_mmq.spv \
	vulkan/dense_iq3_s_decode.spv \
	vulkan/dense_iq3_s_decode_full.spv \
	vulkan/dense_iq3_s_decode_r4.spv \
	vulkan/dense_iq3_s_decode_r1.spv \
	vulkan/dense_iq3_s_mmq.spv \
	vulkan/dense_iq3_s_mmq_r4.spv \
	vulkan/dense_iq4_xs.spv \
	vulkan/dense_iq4_xs_decode.spv \
	vulkan/dense_iq4_xs_mmq.spv \
	vulkan/dense_iq1_m.spv \
	vulkan/dense_extra_decode.spv \
	vulkan/dense_extra_mmq.spv \
	vulkan/dense_kquant_mmq.spv \
	vulkan/dense_kquant_decode.spv \
	vulkan/dense_q4k_decode.spv \
	vulkan/dense_q5k_decode.spv \
	vulkan/moe_reduce.spv \
	vulkan/ffn_tail.spv

vulkan/moe_matvec_fast.spv: vulkan/moe_matvec_fast.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/top2.spv: vulkan/top2.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/recur_norm_gate.spv: vulkan/recur_norm_gate.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/recur_norm_gate_q8_k.spv: vulkan/recur_norm_gate_q8_k.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/swiglu_q8_k.spv: vulkan/swiglu_q8_k.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq3_xxs_decode.spv: vulkan/dense_iq3_xxs_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq3_xxs_decode_r4.spv: vulkan/dense_iq3_xxs_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DQ36_ROWS=4 -DQ36_BOUNDS=0 -o $@ $<

vulkan/dense_iq3_xxs_mmq.spv: vulkan/dense_iq3_xxs_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq3_s_decode.spv: vulkan/dense_iq3_s_decode.comp q36_iq3s_grid_values.inc
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq3_s_decode_full.spv: vulkan/dense_iq3_s_decode.comp q36_iq3s_grid_values.inc
	$(GLSLC) -O --target-env=vulkan1.1 -DQ36_BOUNDS=0 -o $@ $<

vulkan/dense_iq3_s_decode_r4.spv: vulkan/dense_iq3_s_decode.comp q36_iq3s_grid_values.inc
	$(GLSLC) -O --target-env=vulkan1.1 -DQ36_ROWS=4 -DQ36_BOUNDS=0 -o $@ $<

vulkan/dense_iq3_s_decode_r1.spv: vulkan/dense_iq3_s_decode.comp q36_iq3s_grid_values.inc
	$(GLSLC) -O --target-env=vulkan1.1 -DQ36_ROWS=1 -DQ36_BOUNDS=0 -o $@ $<

vulkan/dense_iq3_s_mmq.spv: vulkan/dense_iq3_s_mmq.comp q36_iq3s_grid_values.inc
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq3_s_mmq_r4.spv: vulkan/dense_iq3_s_mmq.comp q36_iq3s_grid_values.inc
	$(GLSLC) -O --target-env=vulkan1.1 -DQ36_BM=4 -DQ36_BK=64 -o $@ $<

vulkan/dense_iq4_xs.spv: vulkan/dense_iq4_xs.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq4_xs_decode.spv: vulkan/dense_iq4_xs_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DQ36_BOUNDS=0 -o $@ $<

vulkan/dense_iq4_xs_mmq.spv: vulkan/dense_iq4_xs_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq1_m.spv: vulkan/dense_iq1_m.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_extra_decode.spv: vulkan/dense_extra_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_extra_mmq.spv: vulkan/dense_extra_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/predequant_b16.spv: vulkan/predequant_b16.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_kquant_mmq.spv: vulkan/dense_kquant_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_kquant_decode.spv: vulkan/dense_kquant_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_q4k_decode.spv: vulkan/dense_kquant_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DQ36_Q4K_ONLY=1 -DQ36_BOUNDS=0 -o $@ $<

vulkan/dense_q5k_decode.spv: vulkan/dense_kquant_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DQ36_Q5K_ONLY=1 -DQ36_BOUNDS=0 -o $@ $<

vulkan/matmul_q8_0_f32b.spv: vulkan/matmul_q8_0_f32b.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q8_0_f32b_pair.spv: vulkan/matmul_q8_0_f32b_pair.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/shared_gate_up_decode.spv: vulkan/shared_gate_up_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/shared_down_tail_decode.spv: vulkan/shared_down_tail_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q8_0_mm.spv: vulkan/matmul_q8_0_mm.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q8_0_mm_f16.spv: vulkan/matmul_q8_0_mm_f16.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q8_0_mm_f16_out32.spv: vulkan/matmul_q8_0_mm_f16_out32.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q8_0_f32b_nx.spv: vulkan/matmul_q8_0_f32b_nx.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/moe_gate_up_f32b.spv: vulkan/moe_gate_up_f32b.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/moe_gate_up_decode.spv: vulkan/moe_gate_up_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/moe_down_q2k_f32b.spv: vulkan/moe_down_q2k_f32b.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/moe_down_q2k_sum_decode.spv: vulkan/moe_down_q2k_sum_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/moe_gate_up_q4k_f32b.spv: vulkan/moe_gate_up_q4k_f32b.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/moe_down_q4k_sum_decode.spv: vulkan/moe_down_q4k_sum_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/moe_gate_up_gemm.spv: vulkan/moe_gate_up_gemm.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/moe_down_gemm.spv: vulkan/moe_down_gemm.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/delta_net_cols.spv: vulkan/delta_net_cols.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/delta_net_cols_f16.spv: vulkan/delta_net_cols.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DQ36_STATE_F16=1 -o $@ $<

vulkan/delta_net_decode_reg.spv: vulkan/delta_net_decode_reg.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/delta_net_decode_reg_f16.spv: vulkan/delta_net_decode_reg.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DQ36_STATE_F16=1 -o $@ $<

vulkan/attn_decode_fused.spv: vulkan/attn_decode_fused.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/attn_decode_split.spv: vulkan/attn_decode_split.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/attn_prefill_qtile.spv: vulkan/attn_prefill_qtile.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/attn_prefill_qtile2.spv: vulkan/attn_prefill_qtile2.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

# Vulkan is the default backend. CORE_OBJS holds the GPU engine; CPU_CORE_OBJS
# is the -DQ36_NO_GPU reference build selected by `make cpu`.
CORE_OBJS := q36_gpu_core.o q36_vulkan.o q36_image.o
METAL_CORE_OBJS := q36_gpu_core_metal.o q36_metal.o q36_image.o
CPU_CORE_OBJS := q36_cpu.o q36_image.o

.PHONY: all help cpu gpu vulkan vulkan-generic vulkan-bc250 metal q36-quality-score test test-metal test-metal-model test-qwen27b-metal-parity test-qwen27b-vulkan-parity test-qwen27b-llama-parity test-quick test-all test-unit test-vulkan test-streaming test-mtp test-model test-session-batch test-server-live test-server-live-metal test-server-live-metal-ssd test-server-batching test-server-batching-metal test-server-batching-metal-ssd test-release release-build-check release-build-check-metal benchmark-gate benchmark-session-batch benchmark-qwen27b benchmark-qwen27b-gate benchmark-qwen27b-metal-gate benchmark-qwen27b-vulkan benchmark-qwen27b-metal benchmark-qwen27b-llama test-reference test-reference-local test-vectors-local reference-openrouter test-llama test-llama-long test-llama-batch test-llama-all clean

all: q36 q36-server q36-bench q36-agent q36-eval q36_test

help:
	@echo "Q36 build targets:"
	@echo "  make              Build generic Vulkan with automatic BC-250 fast-path selection (default)"
	@echo "  make vulkan-generic  Build generic Vulkan for runtime capability detection"
	@echo "  make vulkan-bc250    Build Vulkan and require an AMD BC-250 at runtime"
	@echo "  make metal        Build the same binaries with the Metal backend (macOS)"
	@echo "  make test-metal   Build Metal and run its model-independent unit and kernel tests"
	@echo "  make test-metal-model  Run Metal model, parity, streaming, and state tests"
	@echo "  make test-qwen27b-metal-parity  Run short CPU/Metal parity for the IQ3_S 27B model"
	@echo "  make test-qwen27b-vulkan-parity  Run short CPU/Vulkan parity for the IQ3_S 27B model"
	@echo "  make test-qwen27b-llama-parity  Run short llama.cpp/CPU parity for the IQ3_S 27B model"
	@echo "  make cpu          Build CPU-only ./q36, ./q36-server, ./q36-bench, ./q36-agent, ./q36-eval, and ./q36_test"
	@echo "  make q36-quality-score  Build the OpenRouter/Q36 local scorer"
	@echo "  make test         Build and run tests"
	@echo "  make test-release Run the complete model, Vulkan, reference, streaming, benchmark, and build gates"
	@echo "  make benchmark-session-batch  Benchmark old, 1/2/4/8-slot, and ordered-fallback server decode"
	@echo "  make benchmark-qwen27b  Benchmark q36 at 128, 2048, and 4096 prompt tokens, then llama.cpp at its fitting context"
	@echo "  make benchmark-qwen27b-gate  Quick 128/2048 comparison gate with 32 decode tokens"
	@echo "  make benchmark-qwen27b-metal-gate  Quick Metal 128/2048 gate with 32 decode tokens"
	@echo "  make test-reference       Compare Q36 CPU against tracked llama.cpp results"
	@echo "  make test-reference-local Compare Q36 CPU against ignored local results"
	@echo "  make test-vectors-local   Capture ignored local llama.cpp results"
	@echo "  make reference-openrouter Capture ignored official API results"
	@echo "  make test-llama           Build optional live CPU-vs-llama.cpp tests"
	@echo "  make clean        Remove build outputs"

gpu: vulkan-generic

vulkan: vulkan-generic

vulkan-generic:
	$(MAKE) -B all VULKAN_CFLAGS=

vulkan-bc250:
	$(MAKE) -B all VULKAN_CFLAGS=-DQ36_VULKAN_REQUIRE_BC250

metal: q36_cli_metal.o q36_server.o q36_bench.o q36_agent.o q36_eval.o q36_help.o q36_kvstore.o q36_ssd.o q36_web.o linenoise.o rax.o q36_test_metal.o q36_gpu_core_metal_test.o $(METAL_CORE_OBJS)
	$(CC) $(METAL_LDFLAGS) -o q36 q36_cli_metal.o q36_ssd.o linenoise.o $(METAL_CORE_OBJS) $(METAL_LDLIBS)
	$(CC) $(METAL_LDFLAGS) -o q36-server q36_server.o rax.o q36_ssd.o $(METAL_CORE_OBJS) $(METAL_LDLIBS)
	$(CC) $(METAL_LDFLAGS) -o q36-bench q36_bench.o q36_ssd.o $(METAL_CORE_OBJS) $(METAL_LDLIBS)
	$(CC) $(METAL_LDFLAGS) -o q36-agent q36_agent.o q36_help.o q36_kvstore.o q36_ssd.o q36_web.o linenoise.o $(METAL_CORE_OBJS) $(METAL_LDLIBS)
	$(CC) $(METAL_LDFLAGS) -o q36-eval q36_eval.o q36_help.o q36_ssd.o $(METAL_CORE_OBJS) $(METAL_LDLIBS)
	$(CC) $(METAL_LDFLAGS) -o q36_test q36_test_metal.o rax.o q36_ssd.o q36_image.o q36_gpu_core_metal_test.o q36_metal.o $(METAL_LDLIBS)

q36: q36_cli.o q36_ssd.o linenoise.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ q36_cli.o q36_ssd.o linenoise.o $(CORE_OBJS) $(GPU_LDLIBS)

q36-server: q36_server.o rax.o q36_ssd.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ q36_server.o rax.o q36_ssd.o $(CORE_OBJS) $(GPU_LDLIBS)

q36-bench: q36_bench.o q36_ssd.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ q36_bench.o q36_ssd.o $(CORE_OBJS) $(GPU_LDLIBS)

q36-agent: q36_agent.o q36_help.o q36_kvstore.o q36_ssd.o q36_web.o linenoise.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ q36_agent.o q36_help.o q36_kvstore.o q36_ssd.o q36_web.o linenoise.o $(CORE_OBJS) $(GPU_LDLIBS)

q36-eval: q36_eval.o q36_help.o q36_ssd.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ q36_eval.o q36_help.o q36_ssd.o $(CORE_OBJS) $(GPU_LDLIBS)

q36_test: q36_test.o rax.o q36_ssd.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ q36_test.o rax.o q36_ssd.o $(CORE_OBJS) $(GPU_LDLIBS)

q36-quality-score: gguf-tools/quality-testing/score_openrouter

gguf-tools/quality-testing/score_openrouter: gguf-tools/quality-testing/score_openrouter.o q36_ssd.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -I. -o $@ gguf-tools/quality-testing/score_openrouter.o q36_ssd.o $(CORE_OBJS) $(GPU_LDLIBS)

# `make cpu` rebuilds the same binary names from the -DQ36_NO_GPU objects.
cpu: q36_cli_cpu.o linenoise_cpu.o q36_server_cpu.o rax_cpu.o q36_bench_cpu.o q36_agent_cpu.o q36_eval_cpu.o q36_help_cpu.o q36_kvstore_cpu.o q36_ssd_cpu.o q36_web_cpu.o q36_test_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CPU_CFLAGS) -o q36 q36_cli_cpu.o q36_ssd_cpu.o linenoise_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CPU_CFLAGS) -o q36-server q36_server_cpu.o rax_cpu.o q36_ssd_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CPU_CFLAGS) -o q36-bench q36_bench_cpu.o q36_ssd_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CPU_CFLAGS) -o q36-agent q36_agent_cpu.o q36_help_cpu.o q36_kvstore_cpu.o q36_ssd_cpu.o q36_web_cpu.o linenoise_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CPU_CFLAGS) -o q36-eval q36_eval_cpu.o q36_help_cpu.o q36_ssd_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CPU_CFLAGS) -o q36_test q36_test_cpu.o rax_cpu.o q36_ssd_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

# --- GPU (default) objects ---
q36_gpu_core.o: q36.c q36.h q36_image.h q36_gpu.h q36_quant.h q36_ssd.h q36_iq_tables.h q36_iq3s_grid_values.inc q36_streaming_hotlist.inc
	$(CC) $(GPU_CFLAGS) -c -o $@ q36.c

q36_gpu_core_metal.o: q36.c q36.h q36_image.h q36_gpu.h q36_quant.h q36_ssd.h q36_iq_tables.h q36_iq3s_grid_values.inc q36_streaming_hotlist.inc
	$(CC) $(GPU_CFLAGS) -DQ36_METAL -c -o $@ q36.c

q36_gpu_core_metal_test.o: q36.c q36.h q36_gpu.h q36_quant.h q36_ssd.h q36_iq_tables.h q36_iq3s_grid_values.inc q36_streaming_hotlist.inc
	$(CC) $(GPU_CFLAGS) -DQ36_METAL -DQ36_METAL_TEST_COMPAT -c -o $@ q36.c

q36_cli_metal.o: q36_cli.c q36.h q36_ssd.h linenoise.h
	$(CC) $(GPU_CFLAGS) -DQ36_METAL -c -o $@ q36_cli.c

q36_vulkan.o: q36_vulkan.c q36_gpu.h q36_quant.h q36_iq2_tables_vulkan.inc q36_iq_tables.h q36_iq3s_grid_values.inc $(VULKAN_SHADERS)
	$(CC) $(GPU_CFLAGS) $(VULKAN_CFLAGS) -c -o $@ q36_vulkan.c

q36_metal.o: q36_metal.m q36_gpu.h q36_quant.h q36_ssd.h $(METAL_SRCS)
	$(CC) $(GPU_CFLAGS) -fobjc-arc -c -o $@ q36_metal.m

q36_cli.o: q36_cli.c q36.h q36_ssd.h linenoise.h
	$(CC) $(GPU_CFLAGS) -c -o $@ q36_cli.c

q36_server.o: q36_server.c q36.h rax.h
	$(CC) $(GPU_CFLAGS) -c -o $@ q36_server.c

q36_bench.o: q36_bench.c q36.h
	$(CC) $(GPU_CFLAGS) -c -o $@ q36_bench.c

q36_agent.o: q36_agent.c q36.h q36_help.h q36_kvstore.h q36_ssd.h q36_web.h linenoise.h
	$(CC) $(GPU_CFLAGS) -c -o $@ q36_agent.c

q36_eval.o: q36_eval.c q36.h q36_help.h q36_ssd.h
	$(CC) $(GPU_CFLAGS) -c -o $@ q36_eval.c

q36_help.o: q36_help.c q36_help.h q36.h
	$(CC) $(GPU_CFLAGS) -c -o $@ q36_help.c

q36_kvstore.o: q36_kvstore.c q36_kvstore.h q36.h
	$(CC) $(GPU_CFLAGS) -c -o $@ q36_kvstore.c

q36_ssd.o: q36_ssd.c q36_ssd.h
	$(CC) $(GPU_CFLAGS) -c -o $@ q36_ssd.c

q36_web.o: q36_web.c q36_web.h
	$(CC) $(GPU_CFLAGS) -c -o $@ q36_web.c

q36_test.o: tests/q36_test.c q36_server.c q36.h rax.h q36_gpu.h
	$(CC) $(GPU_CFLAGS) -Wno-unused-function -c -o $@ tests/q36_test.c

q36_test_metal.o: tests/q36_test.c q36_server.c q36.h rax.h q36_gpu.h
	$(CC) $(GPU_CFLAGS) -DQ36_METAL -Wno-unused-function -c -o $@ tests/q36_test.c

gguf-tools/quality-testing/score_openrouter.o: gguf-tools/quality-testing/score_openrouter.c q36.h q36_ssd.h
	$(CC) $(GPU_CFLAGS) -I. -c -o $@ gguf-tools/quality-testing/score_openrouter.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(GPU_CFLAGS) -c -o $@ linenoise.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(GPU_CFLAGS) -c -o $@ rax.c

vulkan/%.spv: vulkan/%.comp
	$(GLSLC) -O -o $@ $<

vulkan/matmul_f32_fast.spv: vulkan/matmul_f32_fast.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q8_0.spv: vulkan/matmul_q8_0.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q5k_mmq_fast.spv: vulkan/matmul_q5k_mmq_fast.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q6k_mmq_fast.spv: vulkan/matmul_q6k_mmq_fast.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q5k_mmq.spv: vulkan/matmul_q5k_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q6k_mmq.spv: vulkan/matmul_q6k_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

# --- CPU-only objects (-DQ36_NO_GPU) ---
q36_cpu.o: q36.c q36.h q36_image.h q36_gpu.h q36_quant.h q36_ssd.h q36_iq_tables.h q36_iq3s_grid_values.inc q36_streaming_hotlist.inc
	$(CC) $(CPU_CFLAGS) -c -o $@ q36.c

q36_image.o: q36_image.c q36_image.h third_party/iris/jpeg.h third_party/iris/png.h
	$(CC) $(GPU_CFLAGS) -c -o $@ q36_image.c

q36_cli_cpu.o: q36_cli.c q36.h q36_ssd.h linenoise.h
	$(CC) $(CPU_CFLAGS) -c -o $@ q36_cli.c

q36_server_cpu.o: q36_server.c q36.h rax.h
	$(CC) $(CPU_CFLAGS) -c -o $@ q36_server.c

q36_bench_cpu.o: q36_bench.c q36.h
	$(CC) $(CPU_CFLAGS) -c -o $@ q36_bench.c

q36_agent_cpu.o: q36_agent.c q36.h q36_help.h q36_kvstore.h q36_ssd.h q36_web.h linenoise.h
	$(CC) $(CPU_CFLAGS) -c -o $@ q36_agent.c

q36_eval_cpu.o: q36_eval.c q36.h q36_help.h q36_ssd.h
	$(CC) $(CPU_CFLAGS) -c -o $@ q36_eval.c

q36_help_cpu.o: q36_help.c q36_help.h q36.h
	$(CC) $(CPU_CFLAGS) -c -o $@ q36_help.c

q36_kvstore_cpu.o: q36_kvstore.c q36_kvstore.h q36.h
	$(CC) $(CPU_CFLAGS) -c -o $@ q36_kvstore.c

q36_ssd_cpu.o: q36_ssd.c q36_ssd.h
	$(CC) $(CPU_CFLAGS) -c -o $@ q36_ssd.c

q36_web_cpu.o: q36_web.c q36_web.h
	$(CC) $(CPU_CFLAGS) -c -o $@ q36_web.c

q36_test_cpu.o: tests/q36_test.c q36_server.c q36.h rax.h
	$(CC) $(CPU_CFLAGS) -Wno-unused-function -c -o $@ tests/q36_test.c

q36_sampling_test_core.o: q36.c q36.h q36_gpu.h q36_quant.h q36_ssd.h q36_iq_tables.h q36_iq3s_grid_values.inc q36_streaming_hotlist.inc
	$(CC) $(CPU_CFLAGS) -Wno-unused-function -DQ36_TEST_HOOKS -c -o $@ q36.c

tests/test_sampling.o: tests/test_sampling.c q36.h
	$(CC) $(CPU_CFLAGS) -DQ36_TEST_HOOKS -I. -c -o $@ $<

$(SAMPLING_TEST): tests/test_sampling.o q36_sampling_test_core.o q36_ssd_cpu.o q36_image.o
	$(CC) $(LDFLAGS) $(DARWIN_MIN_FLAG) -o $@ $^ $(LDLIBS)

linenoise_cpu.o: linenoise.c linenoise.h
	$(CC) $(CPU_CFLAGS) -c -o $@ linenoise.c

rax_cpu.o: rax.c rax.h rax_malloc.h
	$(CC) $(CPU_CFLAGS) -c -o $@ rax.c

test: all q36_agent_test $(SAMPLING_TEST)
	./q36-eval --self-test-extractors
	./q36_agent_test
	./tests/test_sampling
	./q36_test --quant-primitives --ssd-cache-shrink --qwen-tool-call-format --vector-fixtures --server

test-quick: test

test-unit: test

test-metal: metal q36_agent_test_metal $(SAMPLING_TEST)
	./q36-eval --self-test-extractors
	./q36_agent_test_metal
	./tests/test_sampling
	./q36_test --quant-primitives --ssd-cache-shrink --qwen-tool-call-format --vector-fixtures --server
	./q36_test --vulkan-kernels

test-metal-model: metal
	@echo "=== Metal model-dependent tests (requires q36moe.gguf) ==="
	./q36_test --tool-call-quality --qwen-tool-call-quality --thinking-generation --kv-cache-save-restore --session-sync-resume --gpu-cpu-parity --vulkan-fusion-parity --ssd-streaming-parity --mtp-verifier

test-qwen27b-metal-parity: metal
	./q36_test --dense-quant-model-rows --gpu-cpu-parity --model $(QWEN27B_MODEL) --case short

test-qwen27b-vulkan-parity: q36_test
	./q36_test --vulkan-cpu-parity --model $(QWEN27B_MODEL) --case short

test-qwen27b-llama-parity: q36_llama_test
	./q36_llama_test --llama-parity-seq --model $(QWEN27B_MODEL) --case short

test-vulkan: q36_test
	./q36_test --vulkan-kernels

test-streaming: q36_test
	./q36_test --ssd-streaming-parity

test-mtp: q36_test
	./q36_test --mtp-verifier

test-all: test test-vulkan

test-model: q36_test
	@echo "=== Model-dependent tests (requires q36moe.gguf) ==="
	./q36_test --tool-call-quality --qwen-tool-call-quality --thinking-generation --kv-cache-save-restore --session-sync-resume --gpu-cpu-parity --vulkan-fusion-parity --vulkan-queue-parity --mtp-verifier

test-session-batch: q36_test
	./q36_test --vulkan-session-batch
	Q36_TEST_BATCH_KV=f16 ./q36_test --vulkan-session-batch
	Q36_TEST_BATCH_KV=ssd ./q36_test --vulkan-session-batch

test-server-live: q36-server
	./tests/server_live_smoke.sh

test-server-live-metal: metal
	./tests/server_live_smoke.sh --metal

test-server-live-metal-ssd: metal
	./tests/server_live_smoke.sh --metal --ssd-streaming

test-server-batching: q36-server
	./tests/server_batching.sh

test-server-batching-metal: metal
	./tests/server_batching.sh --metal

test-server-batching-metal-ssd: metal
	./tests/server_batching.sh --metal --ssd-streaming

benchmark-session-batch: q36-server
	./tests/benchmark_server_batching.sh

benchmark-gate: q36-bench
	./tests/release_bench_gate.sh

benchmark-qwen27b: benchmark-qwen27b-vulkan benchmark-qwen27b-llama

benchmark-qwen27b-gate: q36-bench
	./q36-bench --vulkan -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 128 --ctx-max 128 --ctx-alloc 512 --prefill-chunk 128 --gen-tokens $(QWEN27B_GATE_GEN)
	./q36-bench --vulkan -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 2048 --ctx-max 2048 --ctx-alloc 2304 --gen-tokens $(QWEN27B_GATE_GEN)

benchmark-qwen27b-metal-gate: metal
	./q36-bench --metal -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 128 --ctx-max 128 --ctx-alloc 512 --prefill-chunk 128 --gen-tokens $(QWEN27B_GATE_GEN)
	./q36-bench --metal -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 2048 --ctx-max 2048 --ctx-alloc 2304 --gen-tokens $(QWEN27B_GATE_GEN)

benchmark-qwen27b-vulkan: q36-bench
	./q36-bench --vulkan -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 128 --ctx-max 128 --ctx-alloc 512 --prefill-chunk 128 --gen-tokens $(QWEN27B_GEN)
	./q36-bench --vulkan -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 2048 --ctx-max 2048 --ctx-alloc 2304 --gen-tokens $(QWEN27B_GEN)
	./q36-bench --vulkan -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 4096 --ctx-max 4096 --ctx-alloc 4352 --gen-tokens $(QWEN27B_GEN)

benchmark-qwen27b-metal: metal
	./q36-bench --metal -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 128 --ctx-max 128 --ctx-alloc 512 --prefill-chunk 128 --gen-tokens $(QWEN27B_GEN)
	./q36-bench --metal -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 2048 --ctx-max 2048 --ctx-alloc 2304 --gen-tokens $(QWEN27B_GEN)
	./q36-bench --metal -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 4096 --ctx-max 4096 --ctx-alloc 4352 --gen-tokens $(QWEN27B_GEN)

benchmark-qwen27b-llama:
	$(QWEN27B_LLAMA_BENCH) -m $(QWEN27B_MODEL) -p $(QWEN27B_LLAMA_PROMPT) -n 0 -r 1 -b $(QWEN27B_LLAMA_PROMPT) -ub $(QWEN27B_LLAMA_PROMPT) -fa off --no-warmup

release-build-check:
	$(MAKE) -B CFLAGS='$(CFLAGS) -Werror' VULKAN_CFLAGS= all
	$(MAKE) -B CFLAGS='$(CFLAGS) -Werror' VULKAN_CFLAGS=-DQ36_VULKAN_REQUIRE_BC250 all
	$(MAKE) -B CFLAGS='$(filter-out -ffast-math,$(CFLAGS)) -Werror' cpu
	$(MAKE) -B CFLAGS='$(CFLAGS) -Werror' VULKAN_CFLAGS= all

release-build-check-metal:
	$(MAKE) -B CFLAGS='$(CFLAGS) -Werror' metal

test-release:
	$(MAKE) test
	$(MAKE) test-vulkan
	$(MAKE) test-model
	$(MAKE) test-server-live
	$(MAKE) test-reference
	$(MAKE) test-streaming
	$(MAKE) benchmark-gate
	$(MAKE) release-build-check

test-kv-cache: q36_test
	./q36_test --kv-cache-save-restore

test-thinking: q36_test
	./q36_test --thinking-generation

test-qwen-tool: q36_test
	./q36_test --qwen-tool-call-quality --qwen-tool-call-format

test-tool: q36_test
	./q36_test --tool-call-quality --qwen-tool-call-quality

test-reference: q36_reference_test
	./q36_reference_test --vector-fixtures --logprob-vectors

test-reference-local: q36_reference_test
	@test -f tests/test-vectors/local/llama.vec || { echo "missing tests/test-vectors/local/llama.vec; run make test-vectors-local"; exit 2; }
	Q36_TEST_VECTOR_FILE=tests/test-vectors/local/llama.vec ./q36_reference_test --logprob-vectors

test-vectors-refresh: .opencode-tmp/llama_qwen_logprobs_capture
	python3 tests/test-vectors/fetch_llama_vectors.py --model $(REFERENCE_MODEL) --threads 8 --capture-bin .opencode-tmp/llama_qwen_logprobs_capture

test-vectors-local: .opencode-tmp/llama_qwen_logprobs_capture
	python3 tests/test-vectors/fetch_llama_vectors.py --out tests/test-vectors/local --model $(REFERENCE_MODEL) --threads 8 --capture-bin .opencode-tmp/llama_qwen_logprobs_capture

reference-openrouter:
	python3 gguf-tools/quality-testing/collect_openrouter.py --model $(OPENROUTER_MODEL) --out $(OPENROUTER_OUT) --count $(OPENROUTER_COUNT) --max-tokens 24 --top-logprobs 20 --provider-order parasail/fp8 --require-parameters --reasoning-effort none

test-llama: q36_llama_test
	./q36_llama_test --llama-parity-seq

test-llama-long: q36_llama_test
	./q36_llama_test --llama-parity-seq-long

test-llama-batch: q36_llama_test
	./q36_llama_test --llama-parity-batch-loose

test-llama-all: q36_llama_test
	./q36_llama_test --llama-parity-seq --llama-parity-seq-long

test-server: q36_test
	./q36_test --server

test-long: q36_test
	./q36_test --long-context

.opencode-tmp/llama_qwen_logprobs_capture: tests/test-vectors/llama_qwen_logprobs_capture.c
	@mkdir -p .opencode-tmp
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(LLAMA_INCLUDE) \
		-o $@ tests/test-vectors/llama_qwen_logprobs_capture.c \
		$(LLAMA_LIBS) $(LDLIBS) $(LLAMA_LDLIBS)

q36_llama_test.o: tests/q36_test.c q36_server.c q36.h rax.h q36_gpu.h
	$(CC) $(CPU_CFLAGS) $(LLAMA_INCLUDE) -DQ36_WITH_LLAMA -Wno-unused-function -c -o $@ tests/q36_test.c

q36_llama_test: q36_llama_test.o rax_cpu.o q36_ssd_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CPU_CFLAGS) -o $@ q36_llama_test.o rax_cpu.o q36_ssd_cpu.o $(CPU_CORE_OBJS) $(LLAMA_LIBS) $(LDLIBS) $(LLAMA_LDLIBS)

q36_reference_test: q36_test_cpu.o rax_cpu.o q36_ssd_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CPU_CFLAGS) -o $@ q36_test_cpu.o rax_cpu.o q36_ssd_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

q36_agent_test.o: tests/q36_agent_test.c q36_agent.c q36.h q36_help.h q36_kvstore.h q36_ssd.h q36_web.h linenoise.h
	$(CC) $(GPU_CFLAGS) -Wno-unused-function -c -o $@ tests/q36_agent_test.c

q36_agent_test: q36_agent_test.o q36_help.o q36_kvstore.o q36_ssd.o q36_web.o linenoise.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ q36_agent_test.o q36_help.o q36_kvstore.o q36_ssd.o q36_web.o linenoise.o $(CORE_OBJS) $(GPU_LDLIBS)

q36_agent_test_metal.o: tests/q36_agent_test.c q36_agent.c q36.h q36_help.h q36_kvstore.h q36_ssd.h q36_web.h linenoise.h
	$(CC) $(GPU_CFLAGS) -DQ36_METAL -Wno-unused-function -c -o $@ tests/q36_agent_test.c

q36_agent_test_metal: q36_agent_test_metal.o q36_help.o q36_kvstore.o q36_ssd.o q36_web.o linenoise.o $(METAL_CORE_OBJS)
	$(CC) $(METAL_LDFLAGS) -o $@ q36_agent_test_metal.o q36_help.o q36_kvstore.o q36_ssd.o q36_web.o linenoise.o $(METAL_CORE_OBJS) $(METAL_LDLIBS)

clean:
	rm -f q36 q36-server q36-bench q36-agent q36-eval q36_test q36_reference_test q36_llama_test q36_agent_test q36_agent_test_metal $(SAMPLING_TEST) tests/test_sampling.o gguf-tools/quality-testing/score_openrouter gguf-tools/quality-testing/score_openrouter.o *.o $(VULKAN_SHADERS)
