#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.
// Max. ncols_dst for which the fused (gate/GLU, bias) MMVQ kernels are instantiated. Fusion at ncols_dst 2-4 is only
// enabled on RDNA4 (ggml_cuda_should_fuse_mul_mat_vec_q), so CUDA builds keep 1 and do not compile the extra kernels.
#if defined(GGML_USE_HIP)
#define MMVQ_MAX_FUSED_NCOLS_DST 4
#else
#define MMVQ_MAX_FUSED_NCOLS_DST 1
#endif // defined(GGML_USE_HIP)

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// q8_1 activation cache (HIP only, see ggml_cuda_q8_cache in common.cuh). All of these are no-ops
// unless the cache is active, i.e. inside ggml_cuda_graph_evaluate_and_capture.
void ggml_cuda_q8_cache_begin(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, bool allow);
void ggml_cuda_q8_cache_end(ggml_backend_cuda_context & ctx);
// Nodes [first, last] have executed as one block: drop entries (from earlier blocks) whose F32 source they may have overwritten.
void ggml_cuda_q8_cache_after_block(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int first, int last);
void ggml_cuda_q8_cache_set_block(ggml_backend_cuda_context & ctx, int first);
// FWHT: q8_1 destination for the contiguous F32 output dst if an MMVQ in this graph will read it, else nullptr.
// The caller must write block_q8_1 in flat element order (identical to quantize_q8_1 of dst).
void * ggml_cuda_q8_cache_fwht_dst(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
