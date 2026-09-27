#include <cstdint>

struct DeepseekV3args {
	std::int64_t max_seq_len = 4096 * 4;
	std::int64_t vocab_size = 102400;
	std::int64_t dim = 2048;
	std::int64_t inter_dim = 10944;
	std::int64_t n_layers = 27;
	std::int64_t n_dense_layers = 1;
	std::int64_t n_heads = 16;

};