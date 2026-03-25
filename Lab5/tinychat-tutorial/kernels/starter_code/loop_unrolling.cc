#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

namespace matmul {
    int sss=10;

void MatmulOperator::mat_mul_loop_unrolling(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32
    float *scale = params->scales, *offset = params->offset;


    //cpu原生最小只支持8bit，也就是对于内存的load，store都需要以8bit为单位，那如果我们想要把量化为4bit的权重存到内存中，该如何操作呢：
    // 4-bit 量化打包 (Pack) 流程详解
    // ============================================================================


    // 1. 【准备阶段】
    // 在 CPU 寄存器中准备好两个 8-bit 变量。
    // 注意：虽然它们是 8-bit 类型，但我们只使用低 4 位来存储有效数据 (0-15)。
    // 例如：val_high = 5 (二进制 0000 0101), val_low = 12 (二进制 0000 1100)
    // int8_t val_high = 5;
    // int8_t val_low  = 12;


    // (可选) 如果原始数据是有符号的 (-8 ~ 7)，需要先加上偏移量 8 转为无符号 (0 ~ 15)
    // uint8_t u_high = val_high + 8; 
    // uint8_t u_low  = val_low + 8;


    // ----------------------------------------------------------------------------


    // 2. 【打包阶段】(位运算)
    // 在 CPU 寄存器内部进行移位和合并操作，生成一个新的 8-bit 变量。
    // 逻辑：
    //   - 将高 4 位数据左移 4 位，腾出低 4 位空间：(0000 0101) << 4  ->  (0101 0000)
    //   - 将低 4 位数据保持不变：                    (0000 1100)        ->  (0000 1100)
    //   - 使用按位或 (|) 将它们“挤”进同一个字节：     (0101 0000) | (0000 1100) -> (0101 1100)
    // 结果：packed = 0x5C (十进制 92)。此时，两个 4-bit 数据已完美融合在一个字节中。
    // uint8_t packed_byte = ((uint8_t)val_high << 4) | (uint8_t)val_low;


    // ----------------------------------------------------------------------------


    // 3. 【访存阶段】(写入内存)
    // 将这个新生成的 8-bit 变量一次性写入内存。
    // 
    // ★ 关键点 ★：
    //   - 内存控制器只看到"1 个字节”被写入。
    //   - 内存硬件根本不知道这个字节里藏着两个数，它只负责忠实地存储这 8 个比特位。
    //   - 效果：原本需要 2 个字节存储的数据，现在只用了 1 个字节，内存带宽节省 50%！
    // memory_address[i] = packed_byte; 


    // ============================================================================
    // 反向过程 (解包/Unpack) - 即推理时的 qj 循环逻辑：
    // 读回 packed_byte (0x5C) -> 右移 4 位取高 4 位 -> 掩码 0x0F 取低 4 位 -> 还原为两个数
    // ============================================================================ 
    






    // 这里做的是 W4A8 量化：weight用4bit量化，activation用8bit量化
    
    // 权重（Weight）： 是B矩阵：模型参数，静态，提前量化好。
    // 存储技巧：2个4-bit数挤在一个 uint8_t (1字节) 里。特殊重排：为了配合硬件指令，权重的存储顺序被打乱了,具体存储顺序 (QM_ARM/QM_x86): (w0, w32), (w1, w33)...。
    // 为什么？
    // 代码在ch的那个循环中，一次处理 2个block (共64个元素)。
    // 第一个block是 t = 0~31，第二个block是 t = 32~63。
    // 为了在一次循环中同时处理这两个block，它把 第1个block的第j个元素 和 第2个block的第j个元素 打包在了一起。
        
    // 激活（Activation）：  是A矩阵：输入 token 的 hidden states，动态，每次推理时量化。量化粒度: 每 block_size (32) 个元素共享一个 scale。A[0...31] 共用 scale[0]，A[32...63] 共用 scale[1]。
    // 因此我们下面可以看到对于A在推理时做动态的量化，而对于B我们已经预先量化好了，直接调用即可


    //我们按照逐block去做量化，一个block是32个元素，量化成8bit后是32个int8元素，等于256bit，也就是AVX2的寄存器宽度
    //然后每32个元素共享一个float类型的scale
    quantize_fp32_to_int8(
        A->data_ptr,        // 输入：原始 FP32 数据（float*）
        A->int8_data_ptr,   // 输出：量化后的 INT8 数据（int8_t* 或 signed char*）
        params->A_scales,   // 输出：每个 block 的 scale（float*）
        A->row * A->column,
        block_size
    );


    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    
    // 注意我们去算的是：C=A*B^T
    // 最朴素的矩阵乘法实现（C = A * B^T）：
    //
    //   for (int i = 0; i < m; i++) {               // 遍历 A 的每一行（共 m 行）
    //       for (int j = 0; j < n; j++) {           // 遍历 B^T 的每一列，也即 B 的每一行，共n个
    //           C[i][j] = 0.0f;
    //           for (int t = 0; t < k; t++) {       // 沿内积维度 t（共 k 个元素）累加
    //               C[i][j] += A[i][t] * B[j][t];   // 注意： B^T[t][j] 就是 B[j][t]
    //           }
    //       }
    //   }
    //
    // 内存布局说明：A，B，C 都是 row-major 存储的二维矩阵
    //   - A.data_ptr[i * k + t] 存储的是 A[i][t]
    //   - B.data_ptr[j * k + t] 存储的是 B[j][t]
    //   - C.data_ptr[i * n + j] 存储的是 C[i][j]




    // 我们在A->int8_data_ptr中存储了量化之后的值，那么在计算的时候怎么去做反量化呢：
    // 由于 A 和 B 都经过分组量化（每 block_size=32 个元素共享一个 scale）：
    //   - A[i][t] ≈ qA[i][t] * sA_block(i,t)     // qA 是 int8，sA 是该 block 的激活 scale
    //   - B[j][t] ≈ qB[j][t] * sB_block(j,t)     // qB 是 int4 解包后的 int8，sB 是权重 scale
    //
    // 因此原始浮点结果可近似为：
    //   C[i][j] = Σ_t A[i][t] * B[j][t]
    //           ≈ Σ_t (qA[i][t] * sA) * (qB[j][t] * sB)
    //           = Σ_t (qA[i][t] * qB[j][t]) * (sA * sB)
    //
    // 关键优化：不显式反量化 qA/qB，而是在整数乘加后一次性乘上 scale：
    //   - 先计算整数点积：intermediate_sum = Σ_t (qA[i][t] * qB[j][t])
    //   - 再融合反量化：C[i][j] += intermediate_sum * sA * sB
    //
    // 注意：由于 scale 是按 block 变化的（每 32 个 t 换一次 scale），
    //       必须将 k 维度按 block 分段处理（即外层 ch 循环），
    //       每段使用对应的 sA 和 sB 进行融合反量化。


    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col += 4) {
            //此处便是所谓的循环展开：一次处理4个col的循环。
            //上面的对于row，col的循环其实就是每次拿 A 的 1 行 和 B 的 4 行，通过内层循环累加，最终算出 C 中的 4 个元素（即 C 矩阵中同一行、相邻的 4 列）
            
            //下面的对于ch的循环其实是为了：处理矩阵乘法的内积维度 (k)。
            // 由于 k 很大 (如 4096+)，无法一次算完，因此将 k 切分为多个小块逐步累加。
            // 步长 = block_size * 2 = 64。即每次处理 64 个元素 (2 个量化块)。
            // 逻辑：
            // 1. 取出 A 当前行的 [ch, ch+64] 片段。
            // 2. 取出 B 的 4 行 (col~col+3) 对应的 [ch, ch+64] 片段。
            // 3. 计算这 64 个元素的局部点积，累加到 acc0~acc3 中。
            // 4. ch += 64，处理下一段，直到遍历完整个 k 维度。


            float acc0 = 0;
            float acc1 = 0;
            float acc2 = 0;
            float acc3 = 0;
            // Compute each block
            for (int ch = 0; ch < k;) {




                // 权重已经预先被重排成交叉格式：(w0, w32), (w1, w33), ..., (w31, w63)。
                // 我们这里一次chunk里面处理64个元素，也就是做量化的时候的2个block


                // pointer of the int8 activation
                // 我们要处理的就是a_int8这个指针接下来的2个block的数据
                const signed char *A_int8 = &A->int8_data_ptr[row * k + ch];


                // pointer of the int4 weights
                // 我们要处理的就是每个Bxx_int4这个指针接下来的2个block的数据
                // 因为我们上面有一个对于col的循环，这里就是把4个循环要用到的都找出来
                // 一个比较hack的方式去寻址：我们要找到B的第col行的第ch列的那个元素，由于我们是使用int4去存每个元素的，也就是地址offset相对于B是(col * k + ch)/2个bytes
                // 而由于我们用的int4_data_ptr是uint8_t*类型（没有int4*这种类型），因此去寻址的时候我们使用(col * k + ch) / 2即可
                uint8_t *B0_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                uint8_t *B1_int4 = &B->int4_data_ptr[((col + 1) * k + ch) / 2];
                uint8_t *B2_int4 = &B->int4_data_ptr[((col + 2) * k + ch) / 2];
                uint8_t *B3_int4 = &B->int4_data_ptr[((col + 3) * k + ch) / 2];


                //上面拿到了原始数据，然后在一轮ch的循环中我们要处理2个block，因此我们需要拿到相应的2个block的scale信息


                // scale of activation
                // 这也是比较hack的一种方式：由于我们32个为一个block去做量化的，因此量化后的元素去除以32（也就是block_size）去找到自己对应的scale
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


                //下面是通过循环处理这2个block的数据
                // int B0_block0_cur = 第0个block的cur_idx_in_block处的数据，存储于B0_int4的某个偏移处
                // int B0_block1_cur = 第1个block的cur_idx_in_block处的数据，本来应该是和B0_block0_cur隔很远的，但是由于我们的设计，他们是紧挨在一起的，然后都是用int4去存，因此可以用int8的方式取出1B的数据然后去解包
                // 逻辑上认为的 order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w62,w63)
                // 重排后的 order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)
                //               |--|
                //               4 bits
                //               |------|
                //               8 bits (byte)
                //            low|----------------------------------------------------------|high
                //               0                         256 bit
                    
                for (int cur_idx_in_block = 0; cur_idx_in_block < block_size; cur_idx_in_block++) {
                    // 每个 byte 中打包了两个 int4：低 4bit -> 第 0 个 block，高 4bit -> 第 1 个 block
                    uint8_t packed_B0 = B0_int4[cur_idx_in_block];
                    uint8_t packed_B1 = B1_int4[cur_idx_in_block];
                    uint8_t packed_B2 = B2_int4[cur_idx_in_block];
                    uint8_t packed_B3 = B3_int4[cur_idx_in_block];


                    //这里主要关注位运算去解包的逻辑，后面的那个-8是因为在做量化的时候我们有对应的平移变换操作
                    signed char B0_block0_cur = (packed_B0 & 0x0F) - 8;
                    signed char B0_block1_cur = (packed_B0 >> 4) - 8;
                    signed char B1_block0_cur = (packed_B1 & 0x0F) - 8;
                    signed char B1_block1_cur = (packed_B1 >> 4) - 8;
                    signed char B2_block0_cur = (packed_B2 & 0x0F) - 8;
                    signed char B2_block1_cur = (packed_B2 >> 4) - 8;
                    signed char B3_block0_cur = (packed_B3 & 0x0F) - 8;
                    signed char B3_block1_cur = (packed_B3 >> 4) - 8;


                    signed char A_block0_cur = A_int8[cur_idx_in_block];
                    signed char A_block1_cur = A_int8[cur_idx_in_block + block_size];


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
                // process two blocks
                ch += block_size * 2;
            }
            C->data_ptr[row * n + col] = acc0;
            C->data_ptr[row * n + col + 1] = acc1;
            C->data_ptr[row * n + col + 2] = acc2;
            C->data_ptr[row * n + col + 3] = acc3;
        }
    }
};



}  // namespace matmul
