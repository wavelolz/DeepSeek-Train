#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

struct MoEArgs {
	std::int64_t top_k = 5;
	std::int64_t num_experts = 64;
};

struct DeepseekV3args {
	// Model args
	std::int64_t max_seq_len = 4096 * 4;
	std::int64_t vocab_size = 102400; // valid range of token ID
	std::int64_t dim = 2048;
	std::int64_t inter_dim = 10944;
	std::int64_t moe_inter_dim = 1408;
	std::int64_t n_layers = 27;
	std::int64_t n_dense_layers = 1;
	std::int64_t n_heads = 16;

	// Deepseek MLA args
	std::int64_t q_lora_rank = 0;
	std::int64_t kv_lora_rank = 512;
	std::int64_t qk_nope_head_dim = 128;
	std::int64_t qk_rope_head_dim = 64;
	std::int64_t v_head_dim = 128;

	// RoPE config
	std::int64_t rope_base = 10000;

	// YaRN args
	std::int64_t original_seq_len = 4096;
	std::int64_t beta_fast = 32;
	std::int64_t beta_slow = 1;
	float scale_factor = 40;

	std::tuple<std::int64_t, std::int64_t> get_nparams_and_flops(
		torch::nn::Module& model,
		std::int64_t seq_len
	) const;
};
