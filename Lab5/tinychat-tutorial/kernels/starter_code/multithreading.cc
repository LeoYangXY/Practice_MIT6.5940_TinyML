#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"
struct multithreading_thread_args {
    int start, end;
    const struct matmul_params* params;
};
static void* multithreading_worker_func(void* args) {
    struct multithreading_thread_args* mat_args = (struct multithreading_thread_args*)args;
    const struct matmul_params* params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = mat_args->start; col < mat_args->end; col++) {
            float acc = 0;
            // Compute each block
            for (int ch = 0; ch < k;) {
                // pointer of the int4 weights
                uint8_t* w_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                // pointer of the int8 activation
                const signed char* a_int8 = &A->int8_data_ptr[row * k + ch];
                // scale of weight
                float s_w = params->scales[(col * k + ch) / block_size];
                // scale of activation
                float s_a = params->A_scales[(row * k + ch) / block_size];
                // scales of the second block
                float s_w_2nd = params->scales[(col * k + ch) / block_size + 1];
                float s_a_2nd = params->A_scales[(row * k + ch) / block_size + 1];
                // order of weights with QM_x86:
                // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w62,w63)
                // QM_ARM order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)
                //               |--|
                //               4 bits
                //               |------|
                //               8 bits (byte)
                //            low|----------------------------------------------------------|high
                //               0                         256 bit
                // process 32 bytes of weigths (256 bit) = 2 blocks
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum = 0, intermediate_sum_2nd = 0;
                for (int qj = 0; qj < 32; qj++) {
                    // decode a packed byte into two int8 in the range of (-8, 7)
                    uint8_t packed_int4_0 = w_int4[qj];
                    signed char w_de_0 = (packed_int4_0 & 0x0F) - 8.0;
                    signed char w_de_16 = (packed_int4_0 >> 4) - 8.0;
                    // int8 multiply and accumulate operation
                    intermediate_sum += a_int8[qj] * w_de_0;
                    intermediate_sum_2nd += a_int8[qj + 32] * w_de_16;
                }
                // dequantize the sum into floating point
                acc += (float)intermediate_sum * s_a * s_w;
                acc += (float)intermediate_sum_2nd * s_a_2nd * s_w_2nd;
                ch += block_size * 2;
            }
            C->data_ptr[row * n + col] = acc;
        }
    }
    return NULL;
}

namespace matmul {
void MatmulOperator::mat_mul_multithreading(struct matmul_params* params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;

    const int num_thread = 4;
    pthread_t thread_pool[num_thread];
    struct multithreading_thread_args threads_args[num_thread];

    //思想很简单，就是每个thread负责读 A 的所有行，读 B 的自己负责的那些行，最后算出C中自己负责的部分
    //只是在multithreading_worker_func中要有对应的处理，上面的函数是：
    //for (int row = 0; row < m; row++) { for (int col = mat_args->start; col < mat_args->end; col++)}} 这样的
    //然后在对于多线程编程的一个固定方式就是创建一个threadPool，方便进行join这样的线程同步
    const int col_per_thread = (n + num_thread - 1) / num_thread;
    for (int t = 0; t < num_thread; t++) {
        threads_args[t].start = t * col_per_thread;
        threads_args[t].end = (t + 1) * col_per_thread;
        if (threads_args[t].end > n) {
            threads_args[t].end = n;
        }
        threads_args[t].params = params;

        // &thread_pool[t]：把新线程的句柄存到数组第 t 个位置
        // NULL：线程默认属性
        // multithreading_worker_func：线程启动后执行的函数
        // &threads_args[t]：传给这个线程的参数（start/end/params）
        int rc = pthread_create(&thread_pool[t], NULL, multithreading_worker_func, &threads_args[t]);
        assert(rc == 0);
    }

    for (int t = 0; t < num_thread; t++) {
        int rc = pthread_join(thread_pool[t], NULL);
        assert(rc == 0);
    }
};
}  // namespace matmul
