#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "../tester/utils.h"

/**
 * @brief Find the k-th largest element in a vector using CUDA.
 * 
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed. 
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */
//用双调函数，j=pass_size是比较的距离，k=stage_size是当前处理的双调序列的长度
//descending是排序方向，false=升序, true=降序,因为要找第k大，所以总是降序
template <typename T>
 __global__ void kernel(T *data, int j, int k, bool descending){

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int ixj = i ^ j;//按位异或，计算出i要进行比较的伙伴元素

  if(ixj > i){//避免重复，只让索引小的线程作比较
    bool direction = ((i & k) == 0);//确定本地排序方向，false=升序, true=降序，当前长度k相当于一个掩码，把序列划分成前后两半
    if(descending) {
      direction = !direction;
    }
    T val_i = data[i];
    T val_ixj = data[ixj];
    if((val_i > val_ixj) == direction){//direction：false=降序, true=升序
      data[i] = val_ixj;
      data[ixj] = val_i;
    }
  }
}

 template <typename T>
T kthLargest(const std::vector<T>& h_input, size_t k) {
  const size_t n= h_input.size();
  if(k == 0 || k > n || n == 0){
    return T(-100);
  }

  size_t m = 1;
  while(m < n) m <<= 1;
  std::vector<T> h_padded_input(m);//必须填充到2的幂次方
  std::copy(h_input.begin(), h_input.end(), h_padded_input.begin());
  std::fill(h_padded_input.begin() + n, h_padded_input.end(), std::numeric_limits<T>::lowest());//用这种数据类型的最小值填充剩余的m-n个位置，降序排列不会影响前n个元素

  T* d_input = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, m * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_input, h_padded_input.data(), m * sizeof(T), cudaMemcpyHostToDevice));

  if(m > 1){
    const unsigned int threads = 256;
    const unsigned int blocks = (m + threads - 1) / threads;
    for(int stage_size =2; stage_size <= m; stage_size <<= 1){
      for(int pass_size = stage_size >> 1; pass_size > 0; pass_size >>= 1){
        kernel<<<blocks, threads>>>(d_input, pass_size, stage_size, true);
        CUDA_CHECK(cudaGetLastError());
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  T result;
  CUDA_CHECK(cudaMemcpy(&result, d_input + (k - 1),sizeof(T), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  return result;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
//编译时确定
constexpr int Br = 32;
constexpr int Bc = 32;
constexpr int max_head_dim = 128;

template <typename T>
__global__ void flashAttentionKernel(
    const T* q_ptr, const T* k_ptr, const T* v_ptr, T* o_ptr,
    int target_seq_len, int src_seq_len, int query_heads, int kv_heads, int head_dim, bool is_causal) {
    
    const int head_group_size = query_heads / kv_heads;
    const float scale = rsqrtf(static_cast<float>(head_dim));//float平方根倒数函数
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int row_block_start = blockIdx.x * Br;
    const int kv_head_idx = head_idx / head_group_size;

    q_ptr += batch_idx * target_seq_len * query_heads * head_dim + head_idx * head_dim;
    k_ptr += batch_idx * src_seq_len * kv_heads * head_dim + kv_head_idx * head_dim;
    v_ptr += batch_idx * src_seq_len * kv_heads * head_dim + kv_head_idx * head_dim;
    o_ptr += batch_idx * target_seq_len * query_heads * head_dim + head_idx * head_dim;
    
    __shared__ T q_tile[Br][max_head_dim];
    __shared__ T k_tile[Bc][max_head_dim];
    __shared__ T v_tile[Bc][max_head_dim];
    T o_acc[max_head_dim];//分配在每个线程的寄存器中，用来累加最后的输出
    T l_i = 0.0f;
    T m_i = -INFINITY;

    for (int i = 0; i < head_dim; ++i) o_acc[i] = 0.0f;

    //加载Q的tile
    for (int i = threadIdx.x; i < Br * head_dim; i += blockDim.x) {//线程0负责0.32.64个元素
        int r = i / head_dim;
        int c = i % head_dim;
        if (row_block_start + r < target_seq_len) {
            q_tile[r][c] = q_ptr[(row_block_start + r) * query_heads * head_dim + c];
        }
    }
    __syncthreads();

    //核心循环，加载K.V
    for (int j_start = 0; j_start < src_seq_len; j_start += Bc) {//外层循环，步长Bc=32
        for (int i = threadIdx.x; i < Bc * head_dim; i += blockDim.x) {
            int r = i / head_dim;
            int c = i % head_dim;
            if (j_start + r < src_seq_len) {
                k_tile[r][c] = k_ptr[(j_start + r) * kv_heads * head_dim + c];
                v_tile[r][c] = v_ptr[(j_start + r) * kv_heads * head_dim + c];
            }
        }
        __syncthreads();

        if (threadIdx.x < Br) {
            const int r_idx = threadIdx.x;
            const int global_r_idx = row_block_start + r_idx;//当前线程处理的Q行在整个张量的全局行索引
            //计算局部注意力分数
            if (global_r_idx < target_seq_len) {
                T m_block = -INFINITY;
                T scores[Bc];//寄存器数组

                for (int c_idx = 0; c_idx < Bc; ++c_idx) {
                    const int global_c_idx = j_start + c_idx;
                    T score = 0.0f;
                    if (global_c_idx < src_seq_len) {
                        for (int d = 0; d < head_dim; ++d) {//取线程对应负责那行和共享内存Bc=32行算点积
                            score += q_tile[r_idx][d] * k_tile[c_idx][d];
                        }
                        score *= scale;
                        if (is_causal && global_r_idx < global_c_idx) {
                            score = -INFINITY;
                        }
                    } else {
                        score = -INFINITY;
                    }
                    scores[c_idx] = score;
                    if (score > m_block) m_block = score;
                }
                
                T m_new = fmaxf(m_i, m_block);
                T l_block = 0.0f;
                T scale_o = expf(m_i - m_new);
                for (int d = 0; d < head_dim; ++d) o_acc[d] *= scale_o;//因子去衰减之前已经累加的输出o_acc

                for (int c_idx = 0; c_idx < Bc; ++c_idx) {
                    T p_val = (scores[c_idx] > -INFINITY) ? expf(scores[c_idx] - m_new) : 0.0f;
                    if (j_start + c_idx < src_seq_len) {
                        l_block += p_val;
                        for (int d = 0; d < head_dim; ++d) {
                            o_acc[d] += p_val * v_tile[c_idx][d];
                        }
                    }
                }
                
                l_i = l_i * scale_o + l_block;
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < Br) {
        int r_idx = threadIdx.x;
        int global_r_idx = row_block_start + r_idx;
        
        if (global_r_idx < target_seq_len) {
            T inv_l = (l_i > 0) ? (1.0f / l_i) : 0.0f;//确保输出零向量而不是NaN
            for (int d = 0; d < head_dim; ++d) {
                o_ptr[global_r_idx * query_heads * head_dim + d] = o_acc[d] * inv_l;
            }
        }
    }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
                      T *d_q, *d_k, *d_v, *d_o;
    CUDA_CHECK(cudaMalloc(&d_q, h_q.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v, h_v.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_o, h_o.size() * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(T), cudaMemcpyHostToDevice));
    
    dim3 gridDim( (target_seq_len + Br - 1) / Br, query_heads, batch_size );
    dim3 blockDim(Br); // Make sure the number of threads matches the tile size Br

    flashAttentionKernel<<<gridDim, blockDim>>>(
        d_q, d_k, d_v, d_o,
        target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, h_o.size() * sizeof(T), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));       
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int>&, size_t);
template float kthLargest<float>(const std::vector<float>&, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);