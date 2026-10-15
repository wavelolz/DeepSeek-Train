#include "rope.h"

#include "Deepseek.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <utility>

namespace {
	double find_correction_val(
		double rotations,
		std::int64_t original_context_length,
		std::int64_t dim,
		double base
	) {
		// calculate frequency based on rotations
		return dim * (std::log(original_context_length / (2.0 * std::numbers::pi * rotations))) / (2.0 * std::log(base));
	}


	std::pair<std::int64_t, std::int64_t> find_correction_range(
		double beta_fast,
		double beta_slow,
		std::int64_t original_context_length,
		std::int64_t dim,
		double base
	) {
		auto low = static_cast<std::int64_t>(std::floor(find_correction_val(beta_fast, original_context_length, dim, base))); // calculate alpah
		auto high = static_cast<std::int64_t>(std::ceil(find_correction_val(beta_slow, original_context_length, dim, base))); // calculate beta

		low = std::max<std::int64_t>(low, 0);
		high = std::min<std::int64_t>(high, dim - 1);

		return { low, high };

	}


	torch::Tensor linear_ramp_function(double low, double high, std::int64_t dim) {
		if (low == high) {
			high += 0.0001;
		}

		const auto options = torch::TensorOptions().dtype(torch::kFloat32);
		auto index = (torch::arange(dim, options) - low) / (high - low);
		return torch::clamp(index, 0.0, 1.0);
	}
} // namespace

torch::Tensor precompute_freqs_cis(const DeepseekV3args& args) {
	const auto dim = args.qk_rope_head_dim;
	const auto seqlen = args.max_seq_len;
	const auto base = args.rope_base;
	const auto beta_fast = args.beta_fast;
	const auto beta_slow = args.beta_slow;
	const auto original_seq_len = args.original_seq_len;
	const auto scale = args.scale_factor;


	// calculate basic RoPE angle
	const auto options = torch::TensorOptions().dtype(torch::kFloat32);
	auto indices = torch::arange(0, dim, 2, options);
	auto freqs = 1.0 / torch::pow(static_cast<double>(base), indices / static_cast<double>(dim));

	// if context length exceeds training context length then apply Yarn
	if (seqlen > args.original_seq_len) {
		auto [low, high] = find_correction_range(beta_fast, beta_slow, original_seq_len, dim, base);
		auto smooth = 1 - linear_ramp_function(low, high, dim / 2);
		freqs = freqs / scale * (1 - smooth) + smooth * freqs;
	}

	auto t = torch::arange(seqlen, options);

	// compute the outer product to get the angle
	auto angles = torch::outer(t, freqs);

	// apply rotation
	auto freq_cis = torch::polar(torch::ones_like(angles), angles);

	return freq_cis;

}

torch::Tensor apply_rope(const torch::Tensor& x, const torch::Tensor& freq_cis) {

	const auto input_type = x.scalar_type();
	// x: [batch, seq_len, head, head_dim]
	// convert x to [batch, seq_len, head, head_dim//2, 2]
	auto x_pairs = x.to(torch::kFloat32).contiguous().view({
		x.size(0), x.size(1), x.size(2), -1, 2
		});

	// convert to [batch, seq_len, head, head_dim//2] but in complex form
	auto x_complex = torch::view_as_complex(x_pairs);

	// broadcast frequency matrix to [1, seq_len, 1, head_dim//2]
	auto frequency = freq_cis.view({
		1, freq_cis.size(0), 1, freq_cis.size(1)
		});

	// apply rotation matrix
	auto rotated = x_complex * frequency;
	
	// turn back to real matrix
	auto y = torch::view_as_real(rotated).flatten(3);

	return y.to(input_type);
}
