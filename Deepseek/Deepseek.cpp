#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
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

	std::tuple <std::int64_t, std::int64_t>
		get_nparams_and_flops(
			torch::nn::Module & model,
			std::int64_t seq_len
		) const
	{
		std::int64_t nparams_dense = 0;
		std::int64_t nparams_embedding = 0;
		std::int64_t nparams_moe_router = 0;
		std::int64_t nparams_shared_expert = 0;
		std::int64_t nparams_experts = 0;

		for (const auto& item : model.named_parameters()) {
			const std::string& name = item.key();
			const torch::Tensor& p = item.value();

			if (name.find("embedding") != std::string::npos) {
				nparams_dense += p.numel();
				nparams_embedding += p.numel();
			}
			else if (name.find("moe.shared_experts") != std::string::npos) {
				nparams_shared_expert += p.numel();
			}
			else if (name.find("moe.router") != std::string::npos) {
				nparams_moe_router += p.numel();
			}
			else if (name.find("moe.experts") != std::string::npos) {
				nparams_experts += p.numel();
			}
			else {
				nparams_dense += p.numel();
			}
		}


		MoEArgs moe_args;

		std::int64_t nparams_sparse = nparams_moe_router + nparams_shared_expert + nparams_experts;
		std::int64_t nparams = nparams_dense + nparams_sparse;

		std::int64_t nparams_sparse_active = (
			nparams_moe_router + nparams_shared_expert + (nparams_experts * moe_args.top_k / moe_args.num_experts)
			);

		std::clog
			<< "Total number of dense parameters" << nparams_dense
			<< "Total number of sparse parameters" << nparams_sparse
			<< "Total number of active parameters" << nparams_sparse_active;

		const std::int64_t head_dims = (
			qk_nope_head_dim + qk_rope_head_dim + v_head_dim
			);

		const std::int64_t flops_per_token = (
			6 * (nparams - nparams_embedding) + 6 * (n_layers + n_heads + head_dims + seq_len)
			);

		return { nparams, flops_per_token };


	}
};