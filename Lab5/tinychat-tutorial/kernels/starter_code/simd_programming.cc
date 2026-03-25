#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <xmmintrin.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

#ifdef QM_ARM
#include <arm_neon.h>
#endif
#if defined(QM_x86) || defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif
namespace matmul {
void MatmulOperator::mat_mul_simd_programming(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    // __m256	256-bit 单精度浮点向量
    // __m256d	256-bit 双精度浮点向量
    // __m256i	256-bit 整数向量
    /*
    * ============================================================================
    * 核心概念：__m256 / __m256i / __m256d 的本质与使用规则
    * ============================================================================
    * 
    * 1. 存储层面（容器视角）
    * ----------------------------------------
    * - 本质：__m256、__m256i、__m256d 在内存和寄存器中没有任何区别。
    * - 物理形态：它们都是纯粹的 "256 位二进制盒子" (32 字节)。
    * - 灵活性：你可以往里面塞任何 256 位的比特流。
    *   * 可以是合法的浮点数数组
    *   * 可以是合法的整数数组
    *   * 可以是乱码、图片片段、加密密文，甚至是混合类型的数据
    * - 结论：硬件不关心盒子里装的是什么，它只负责原样存储和搬运这 256 个 bit。
    * 
    * 2. 计算层面（工具视角）
    * ----------------------------------------
    * - 机制：SIMD 指令集是 "有眼睛" 的，它们决定了如何解读盒子里的比特。
    * - 浮点指令 (如 _mm256_add_ps)：
    *   CPU 戴上 "浮点眼镜"，强制将这 256 位切分为 8 段，每段按 IEEE 754 标准解读为 float。
    * - 整数指令 (如 _mm256_add_epi32)：
    *   CPU 戴上 "整数眼镜"，强制将这 256 位切分为 8 段 (或 4 段/16 段)，每段解读为补码整数。
    * - 结论：数据本身没有类型，"类型" 是由你调用的指令决定的。
    * 
    * 3. 如果 "存错了" 或者 "用错了指令" 会发生什么？
    * ----------------------------------------
    * 
    * [情况 A] 存的是整数，却用了浮点指令
    * - 现象：编译通过，运行不崩溃，但结果是完全错误的乱码浮点数。
    * - 原理：整数的二进制模式被强行按浮点规则解读。
    * - 例子：
    *   数据：整数 1 (二进制 ...0001)
    *   错误操作：用 _mm256_add_ps 处理
    *   结果：CPU 将 ...0001 解读为极小的非规约浮点数 (约 1.4e-45)，而非 1.0。
    * 
    * [情况 B] 存的是浮点数，却用了整数指令
    * - 现象：编译通过，运行不崩溃，结果是毫无意义的乱码整数。
    * - 原理：浮点数的二进制模式被强行按整数规则解读。
    * - 例子：
    *   数据：浮点数 1.0 (二进制 0x3F800000，即十进制 1065353216)
    *   错误操作：用 _mm256_add_epi32 对其加 1
    *   结果：得到 1065353217。若再转回浮点数，完全不是预期的 2.0。
    * 
    * [情况 C] 类型不匹配导致编译报错 (最常见)
    * - 现象：编译器直接报错，阻止代码生成。
    * - 原因：C++ 类型系统保护机制。__m256 和 __m256i 是不同的 struct 类型，不能隐式转换。
    * - 解决方法：使用 Intrinsics 提供的 "Cast" 函数。
    *   示例：_mm256_castps_si256(val)  // 把 __m256 强转为 __m256i
    *         _mm256_castsi256_ps(val)  // 把 __m256i 强转为 __m256
    * - 关键点：
    *   * 零开销：这些 Cast 函数不产生任何机器码，不消耗时钟周期。
    *   * 纯语义：它们只是告诉编译器 "别检查了，我知道我在干什么，就把这盒比特当作另一种类型处理"。
    *   * 风险：使用后若指令与数据实际内容不匹配，仍会落入情况 A 或 B 的陷阱。
    * 
    * ============================================================================
    * 总结
    * ============================================================================
    * - 盒子 (变量类型) 是通用的，什么都能装。
    * - 机器 (指令) 是专用的，必须 "门当户对"。
    * - 编程时的核心任务：确保 "盒子里的实际数据格式" 与 "你调用的指令所期望的格式" 严格一致。
    * - 类型转换 (Cast) 是安全的 "换标签" 操作，但换标签后必须配合正确的 "机器" 使用。
    */



    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {

            // order of weights with QM_x86:
            // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w62,w63)
            // QM_ARM order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)
            //               |--|
            //               4 bits
            //               |------|
            //               8 bits (byte)
            //            low|----------------------------------------------------------|high
            //               0                         256 bit
            __m256 acc0 = _mm256_setzero_ps();
            // pointer of the int4 weights
            const __m256i *b_start = (__m256i *)&B->int4_data_ptr[col * k / 2];
            // pointer of the int8 activation
            const __m256i *a_start = (__m256i *)&A->int8_data_ptr[row * k];
            // scale of weight
            float *scale_b_ptr = &params->scales[col * k / 32];
            // scale of activation
            float *scale_a_ptr = &params->A_scales[row * k / 32];

            const int num_block = k / block_size;
            // Compute two blocks in each iteration
            for (int q = 0; q < num_block; q += 2) {
                __m256i a_block0 = a_start[0]; 
                __m256i a_block1 = a_start[1];

                // lowbit mask
                const __m256i low_mask = _mm256_set1_epi8(0xF); //逻辑上可以理解为得到 [15, 15, ..., 15] 这样的数组，总共是256 bit/8 bit=32个元素
  
                //从b_start处这个内存地址读取256bit，存到raw_b这个寄存器中，现在raw_b中是：
                // QM_ARM order: (w0|w32),(w1|w33),(w2|w34),(w3|w35),(w4|w36),... (w31|w63)
                //               |--|
                //               4 bits
                //               |------|
                //               8 bits (byte)
                __m256i raw_b = _mm256_loadu_si256(b_start);
                

                //常见的位运算技巧：
                // aaaabbbb & 00001111 = 0000bbbb  最后的结果就是看我们的源数据的那个b是0还是1
                // aaaabbbb & 11110000 = aaaa0000  最后的结果就是看我们的源数据的那个a是0还是1
                // 因此，我们想要把哪里清零，想要把哪里保留，就是对应的设置对应的位为0，1即可

                // 提取低 4 位，变成(0|w32),(0|w33),...,(0|w63)，那么其实最后数据可以理解为(w32),(w33),...,(w63)
                // 我们用256bit的寄存器去保存它
                __m256i low_vec = _mm256_and_si256(raw_b, low_mask);

                //提取高 4 位，那么就是先右移4bit，把高位的挪到之前低位的那里，然后再用之前的低位提取的那个and操作
                __m256i shifted = _mm256_srli_epi16(raw_b, 4);
                __m256i high_vec = _mm256_and_si256(shifted, low_mask);

                // TODO: apply zero_point to weights and convert the range from (0, 15) to (-8, 7)
                // Hint: using `_mm256_sub_epi8` to the lower-half and upper-half vectors of weights
                // Note: Store the lower half and upper half of weights into `w_0` and `w_128`, respectively
                const __m256i zero_point = _mm256_set1_epi8(8);

                //去思考这个 256bit 的寄存器中的每个小块会怎么变化，便可理解 SIMD
                //因为是i8的指令，也就是说现在是256bit的寄存器总大小，然后按照8bit去分小块，下面的每个[]代表8bit
                //[ w0 ] [ w1 ] [ w2 ] ... [ w31 ]与[ 8 ] [ 8 ] [ 8 ] ... [ 8 ]进行相减
                //变为[ w0-8 ] [ w1-8 ] [ w2-8 ] ... [ w31-8 ]
                __m256i b_block0 = _mm256_sub_epi8(low_vec, zero_point);
                __m256i b_block1 = _mm256_sub_epi8(high_vec, zero_point);

                // 下面我们可以用simd的方式，对于A和B拿到的小block去求乘积
                // 不过有一个核心问题：硬件的限制
                // CPU 提供了一个高效的指令 _mm256_maddubs_epi16 来做字节乘法累加，但它有一个奇怪的硬性规定：
                // 输入 1 (s1)：必须是 无符号字节 (Unsigned Int8, 0~255)。
                // 输入 2 (s2)：必须是 有符号字节 (Signed Int8, -128~127)。
                // 输出：相邻的两个乘积相加，生成 有符号 16 位整数 (Int16)。
                // 困境：
                // 刚才我们解包出来的权重 b_block0 和 b_block1 是 有符号的 (-8 ~ 7)。
                // 激活值 (Activation) 通常也是 有符号的。
                // 👉 两个都是有符号的，直接传给 _mm256_maddubs_epi16 会出错！ 因为第一个参数必须是无符号的。
                // 利用数学变换去处理：
                // 公式： A × B = |B| * (sign(B) * A) 
                //          ^      ^        ^
                //          |      |        |
                //     原始乘积  权重绝对值  (权重符号 × 激活值)


                //计算|B|
                __m256i abs_b_block0 = _mm256_abs_epi8(b_block0);
                __m256i abs_b_block1 = _mm256_abs_epi8(b_block1);
                
                //计算sign(B) * A
                __m256i signed_a_block0 = _mm256_sign_epi8(a_block0, b_block0);
                __m256i signed_a_block1 = _mm256_sign_epi8(a_block1, b_block1);


                //下面这个指令很tricky：
                //官方解读：
                // /// \param __a
                // ///    A 256-bit vector containing one of the source operands.
                // /// \param __b
                // ///    A 256-bit vector containing one of the source operands.
                // /// \returns A 256-bit vector of [16 x i16] containing the result.
                // static __inline__ __m256i __DEFAULT_FN_ATTRS256
                // _mm256_maddubs_epi16(__m256i __a, __m256i __b)
                
                //对于abs_b_block0和signed_a_block0中的每一个slot，也就是8bit的空间中，两个元素进行相乘的话，如果我们还是拿以8bit为slot的总共是256bit的寄存器去存的话，会溢出
                //输入[(x0),(x1),(x2),...]  [(y0),(y1),(y2)...]  每个()是代表一个8bit的slot
                //使用下面的那个madd指令会做如下事情：
                //step1：
                // t0 = x0 * y0   
                // t1 = x1 * y1
                // t2 = x2 * y2
                // t3 = x3 * y3
                // ...
                //这些t0，t1...会使用16bit去存，这样能有效避免溢出
                //
                //step2：
                // out0 = t0 + t1  
                // out1 = t2 + t3   
                //最终输出:[out0, out1, out2, ...]，每个out是占用1个16bit的slot  类型：__m256i (包含 16 个 int16 元素)
                __m256i dot0 = _mm256_maddubs_epi16(abs_b_block0, signed_a_block0);
                __m256i dot1 = _mm256_maddubs_epi16(abs_b_block1, signed_a_block1);

                //下面我们其实是想把dot0中的数和量化的scale相乘之后，加到acc0里面去的；dot1也是如此
                //不过为了高效的利用simd指令，我们在一开始把acc0设为一个向量寄存器，这样子可以做多路并行


                //下面的指令和之前的一样，也是两两相邻的进行相加的那种：
                // /// \param __a
                // ///    A 256-bit vector of [16 x i16] containing one of the source operands.
                // /// \param __b
                // ///    A 256-bit vector of [16 x i16] containing one of the source operands.
                // /// \returns A 256-bit vector of [8 x i32] containing the result.
                // static __inline__ __m256i __DEFAULT_FN_ATTRS256
                // _mm256_madd_epi16(__m256i __a, __m256i __b)
                //得到的那个summed_block0是有8个int32的，summed_block1也是如此
                const __m256i ones = _mm256_set1_epi16(1);
                const __m256i summed_block0 = _mm256_madd_epi16(ones, dot0);
                const __m256i summed_block1 = _mm256_madd_epi16(ones, dot1);

                //把这 8 个 int32 转成 8 个 float
                __m256 intermediate = _mm256_cvtepi32_ps(summed_block0);
                __m256 intermediate2 = _mm256_cvtepi32_ps(summed_block1);

                // 不是把结果加成一个数，而是把结果加到 acc0 的对应位置上,在循环结束之后才对于acc0做reduce                
                __m256 vec_scale0 = _mm256_set1_ps(scale_b_ptr[0] * scale_a_ptr[0]);
                acc0 = _mm256_fmadd_ps(intermediate, vec_scale0, acc0);

                __m256 vec_scale1 = _mm256_set1_ps(scale_b_ptr[1] * scale_a_ptr[1]);
                acc0 = _mm256_fmadd_ps(intermediate2, vec_scale1, acc0);

                scale_b_ptr += 2;
                scale_a_ptr += 2;
                b_start += 1;//b处理完了2个block，每个block是32个int4，因此最后是64个int4，也就是256bit，所以b_start每次移动1个单位
                a_start += 2;//a也是处理完了2个block，每个block是32个int8，因此最后是512bit，所以a_start每次移动2个单位
            }
            // //将acc0这个存了8个float数字的256bit的寄存器reinterpret成一个float类型的指针，这样就可以访问到里面的8个float元素了
            // float *ptr = (float *)&acc0;
            // C->data_ptr[row * n + col] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
            //注意，上面这个存在的问题：
            // 当你写 &acc0 时，编译器并没有获取寄存器 ymm0 (假设 acc0 存在这里) 的地址，因为寄存器根本没有地址。
            // 编译器实际做的操作是：
            // 隐式存储 (Spill)：编译器发现你要取地址，它被迫把寄存器 acc0 里的数据 拷贝 (Store) 到栈内存 (Stack Memory) 的一个临时位置。
            // 取地址：然后它把这个临时内存位置的地址赋给 ptr。
            // 后续读取：当你执行 ptr[0] + ptr[1]... 时，CPU 又从那个临时内存位置把数据 读回 (Load) 到寄存器进行计算
            // 我们希望的是能直接在寄存器里对这 8 个 float 做水平加法 (Horizontal Add)，而不是频繁地在寄存器和内存之间搬运数据。
            
            // --- 开始：高效的寄存器内求和 ---
            
            // 拆分：把 256-bit 拆成两个 128-bit (低半部分和高半部分)
            // _mm256_castps256_ps128: 零开销，只是告诉编译器取低128位
            __m128 low = _mm256_castps256_ps128(acc0); 
            // _mm256_extractf128_ps: 提取高128位 (需要一点指令开销)
            __m128 high = _mm256_extractf128_ps(acc0, 1);
            
            // 第一次相加：[l0, l1, l2, l3] + [h0, h1, h2, h3] = [l0+h0, l1+h1, l2+h2, l3+h3]
            __m128 sum4 = _mm_add_ps(low, high);
            
            // __mm_hadd_ps作用：
            // 调用：temp = _mm_hadd_ps(sum4, sum4)
            // 输入 a = [a0, a1, a2, a3], b = [b0, b1, b2, b3]
            // 前2个slot去放 a 的相邻和：a0+a1, a2+a3
            // 后2个slot去放 b 的相邻和：b0+b1, b2+b3
            // 结果 temp = [a0+a1, a2+a3, b0+b1, b2+b3]
            
            __m128 temp = _mm_hadd_ps(sum4, sum4);// temp = [l0+h0+l1+h1, l2+h2+l3+h3, l0+h0+l1+h1, l2+h2+l3+h3]
            __m128 final_sum = _mm_hadd_ps(temp, temp);// final_sum = [l0+h0+l1+h1+l2+h2+l3+h3, ..., ..., ...]
            
            //在寄存器中操作完毕了最后才写入内存
            C->data_ptr[row * n + col] = _mm_cvtss_f32(final_sum);


        }
    }
};
}  // namespace matmul
