#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

struct multithreading_loop_unrolling_thread_args {
    int start, end;
    const struct matmul_params *params;
};

static void *multithreading_loop_unrolling_worker_func(void *args) {
    struct multithreading_loop_unrolling_thread_args *mat_args =
        (struct multithreading_loop_unrolling_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;

    int m = C->row, n = C->column, k = A->column;
    for (int row = 0; row < m; row++) {
        for (int col = mat_args->start; col < mat_args->end; col += 4) {
            float acc0 = 0;
            float acc1 = 0;
            float acc2 = 0;
            float acc3 = 0;

            for (int ch = 0; ch < k;) {
                const signed char *A_int8 = &A->int8_data_ptr[row * k + ch];
                uint8_t *B0_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                uint8_t *B1_int4 = &B->int4_data_ptr[((col + 1) * k + ch) / 2];
                uint8_t *B2_int4 = &B->int4_data_ptr[((col + 2) * k + ch) / 2];
                uint8_t *B3_int4 = &B->int4_data_ptr[((col + 3) * k + ch) / 2];
                float scale_A_block0 = params->A_scales[(row * k + ch) / block_size];
                float scale_B0_block0 = params->scales[(col * k + ch) / block_size];
                float scale_B1_block0 = params->scales[((col + 1) * k + ch) / block_size];
                float scale_B2_block0 = params->scales[((col + 2) * k + ch) / block_size];
                float scale_B3_block0 = params->scales[((col + 3) * k + ch) / block_size];
                float scale_A_block1 = params->A_scales[(row * k + ch) / block_size + 1];
                float scale_B0_block1 = params->scales[(col * k + ch) / block_size + 1];
                float scale_B1_block1 = params->scales[((col + 1) * k + ch) / block_size + 1];
                float scale_B2_block1 = params->scales[((col + 2) * k + ch) / block_size + 1];
                float scale_B3_block1 = params->scales[((col + 3) * k + ch) / block_size + 1];
                int intermediate_sum0_block0 = 0, intermediate_sum1_block0 = 0, intermediate_sum2_block0 = 0,
                    intermediate_sum3_block0 = 0;
                int intermediate_sum0_block1 = 0, intermediate_sum1_block1 = 0, intermediate_sum2_block1 = 0,
                    intermediate_sum3_block1 = 0;
                for (int qj = 0; qj < 32; qj++) {
                    uint8_t packed_B0 = B0_int4[qj];
                    uint8_t packed_B1 = B1_int4[qj];
                    uint8_t packed_B2 = B2_int4[qj];
                    uint8_t packed_B3 = B3_int4[qj];
                    signed char B0_block0_cur = (packed_B0 & 0x0F) - 8;
                    signed char B0_block1_cur = (packed_B0 >> 4) - 8;
                    signed char B1_block0_cur = (packed_B1 & 0x0F) - 8;
                    signed char B1_block1_cur = (packed_B1 >> 4) - 8;
                    signed char B2_block0_cur = (packed_B2 & 0x0F) - 8;
                    signed char B2_block1_cur = (packed_B2 >> 4) - 8;
                    signed char B3_block0_cur = (packed_B3 & 0x0F) - 8;
                    signed char B3_block1_cur = (packed_B3 >> 4) - 8;

                    signed char A_block0_cur = A_int8[qj];
                    signed char A_block1_cur = A_int8[qj + block_size];

                    intermediate_sum0_block0 += A_block0_cur * B0_block0_cur;
                    intermediate_sum1_block0 += A_block0_cur * B1_block0_cur;
                    intermediate_sum2_block0 += A_block0_cur * B2_block0_cur;
                    intermediate_sum3_block0 += A_block0_cur * B3_block0_cur;
                    intermediate_sum0_block1 += A_block1_cur * B0_block1_cur;
                    intermediate_sum1_block1 += A_block1_cur * B1_block1_cur;
                    intermediate_sum2_block1 += A_block1_cur * B2_block1_cur;
                    intermediate_sum3_block1 += A_block1_cur * B3_block1_cur;
                }
                acc0 += (float)intermediate_sum0_block0 * scale_A_block0 * scale_B0_block0;
                acc0 += (float)intermediate_sum0_block1 * scale_A_block1 * scale_B0_block1;
                acc1 += (float)intermediate_sum1_block0 * scale_A_block0 * scale_B1_block0;
                acc1 += (float)intermediate_sum1_block1 * scale_A_block1 * scale_B1_block1;
                acc2 += (float)intermediate_sum2_block0 * scale_A_block0 * scale_B2_block0;
                acc2 += (float)intermediate_sum2_block1 * scale_A_block1 * scale_B2_block1;
                acc3 += (float)intermediate_sum3_block0 * scale_A_block0 * scale_B3_block0;
                acc3 += (float)intermediate_sum3_block1 * scale_A_block1 * scale_B3_block1;
                ch += block_size * 2;
            }
            C->data_ptr[row * n + col] = acc0;
            C->data_ptr[row * n + col + 1] = acc1;
            C->data_ptr[row * n + col + 2] = acc2;
            C->data_ptr[row * n + col + 3] = acc3;
        }
    }
    return NULL;
}

namespace matmul {
void MatmulOperator::mat_mul_multithreading_loop_unrolling(struct matmul_params *params) {
    const struct matrix *A = &params->A, *C = &params->C;
    const int block_size = params->block_size;
    assert(params->block_size % 32 == 0);
    assert(A->row == C->row);
    assert(params->block_size == 32);

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int n = C->column;
    const int num_thread = 4;
    pthread_t thread_pool[num_thread];
    struct multithreading_loop_unrolling_thread_args threads_args[num_thread];

    assert(n % 4 == 0);
    const int col_blocks = n / 4;
    const int blocks_per_thread = (col_blocks + num_thread - 1) / num_thread;

    for (int t = 0; t < num_thread; t++) {
        const int start_block = t * blocks_per_thread;
        const int end_block = (t + 1) * blocks_per_thread;

        threads_args[t].start = start_block * 4;
        threads_args[t].end = end_block * 4;
        if (threads_args[t].end > n) {
            threads_args[t].end = n;
        }
        threads_args[t].params = params;

        int rc = pthread_create(&thread_pool[t], NULL, multithreading_loop_unrolling_worker_func, &threads_args[t]);
        assert(rc == 0);
    }

    for (int t = 0; t < num_thread; t++) {
        int rc = pthread_join(thread_pool[t], NULL);
        assert(rc == 0);
    }
}
}  // namespace matmul
