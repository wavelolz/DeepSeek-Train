#pragma once

#include <torch/torch.h>

struct DeepseekV3args;

torch::Tensor precompute_freqs_cis(const DeepseekV3args& args);

torch::Tensor apply_rope(
	const torch::Tensor& x,
	const torch::Tensor& freq_cis
);
