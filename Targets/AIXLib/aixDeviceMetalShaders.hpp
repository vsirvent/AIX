//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once

namespace aix::metal::shaders
{

// Requires Metal Language Version 2_2 and higher.
const char* aixDeviceMetalShaders = R"(

#include <metal_atomic>
#include <metal_stdlib>
using namespace metal;

#define BATCH_PROCESS_SIZE_PER_THREAD       1

struct MatrixSize
{
    size_t rows;
    size_t cols;
};

// -----------------------------------------------------------------
// ATOMIC UTILS
// -----------------------------------------------------------------

#pragma METAL internals : enable
template <typename T>
constexpr constant bool is_metal_atomic = _disjunction<is_same<T, int>,
                                                        is_same<T, uint>,
                                                        is_same<T, ulong>,
                                                        is_same<T, float>>::value;

#pragma METAL internals : disable

template <typename T, typename = void>
struct aix_atomic
{
    atomic<uint> val;
};

template <typename T>
struct aix_atomic<T, enable_if_t<is_metal_atomic<T>>>
{
    atomic<T> val;
};

// -----------------------------------------------------------------
// NATIVE METAL ATOMICS
// -----------------------------------------------------------------

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC T aix_atomic_load_explicit(device aix_atomic<T>* object, uint offset)
{
    return atomic_load_explicit(&(object[offset].val), memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void
aix_atomic_store_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    atomic_store_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_and_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    atomic_fetch_and_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_or_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    atomic_fetch_or_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_min_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    atomic_fetch_min_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_max_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    atomic_fetch_max_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_add_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    atomic_fetch_add_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_mul_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    T expected = aix_atomic_load_explicit(object, offset);
    while (!aix_atomic_compare_exchange_weak_explicit(object, &expected, val * expected, offset)) { }
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC bool aix_atomic_compare_exchange_weak_explicit(device aix_atomic<T>* object,
                                                          thread T* expected,
                                                          T val,
                                                          uint offset)
{
    return atomic_compare_exchange_weak_explicit(&(object[offset].val), expected, val,
                                                 memory_order_relaxed, memory_order_relaxed);
}

// Specialization for float since it does not atomic_fetch_min_explicit
template <>
METAL_FUNC void aix_atomic_fetch_min_explicit<float>(device aix_atomic<float>* object, float val, uint offset)
{
    float expected = aix_atomic_load_explicit(object, offset);
    while (val < expected)
    {
        if (aix_atomic_compare_exchange_weak_explicit(object, &expected, val, offset)) { return; }
    }
}

// Specialization for float since it does not atomic_fetch_max_explicit
template <>
METAL_FUNC void aix_atomic_fetch_max_explicit<float>(device aix_atomic<float>* object, float val, uint offset)
{
    float expected = aix_atomic_load_explicit(object, offset);
    while (val > expected)
    {
        if (aix_atomic_compare_exchange_weak_explicit(object, &expected, val, offset)) { return; }
    }
}

// -----------------------------------------------------------------
// CUSTOM ATOMICS
// -----------------------------------------------------------------

namespace
{

template <typename T>
constexpr constant uint packing_size = sizeof(uint) / sizeof(T);

template <typename T>
union uint_or_packed
{
    T val[packing_size<T>];
    uint bits;
};

template <typename T, typename Op>
struct aix_atomic_update_helper
{
    uint operator()(uint_or_packed<T> init, T update, uint elem_offset)
    {
        Op op;
        init.val[elem_offset] = op(update, init.val[elem_offset]);
        return init.bits;
    }
};

template <typename T, typename Op>
METAL_FUNC void aix_atomic_update_and_store(device aix_atomic<T>* object, T update, uint offset)
{
    uint pack_offset = offset / packing_size<T>;
    uint elem_offset = offset % packing_size<T>;

    aix_atomic_update_helper<T, Op> helper;
    uint_or_packed<T> expected;
    expected.bits = atomic_load_explicit(&(object[pack_offset].val), memory_order_relaxed);

    while (Op::condition(update, expected.val[elem_offset]) &&
           !aix_atomic_compare_exchange_weak_explicit(object, &(expected.bits),helper(expected, update, elem_offset),
                                                      pack_offset))
    { }
}

template <typename T>
struct __None
{
    static bool condition(T a, T b)
    {
        #pragma unused(a)
        #pragma unused(b)
        return true;
    }

    T operator()(T a, T b)
    {
        #pragma unused(b)
        return a;
    }
};

template <typename T>
struct __Add
{
    static bool condition(T a, T b)
    {
        #pragma unused(a)
        #pragma unused(b)
        return true;
    }

    T operator()(T a, T b)
    {
        return a + b;
    }
};

template <typename T>
struct __Mul
{
    static bool condition(T a, T b)
    {
        #pragma unused(a)
        return b != 0;
    }

    T operator()(T a, T b)
    {
        return a * b;
    }
};

template <typename T>
struct __Max
{
    static bool condition(T a, T b)
    {
        return a > b;
    }

    T operator()(T a, T b)
    {
        return max(a, b);
    }
};

template <typename T>
struct __Min
{
    static bool condition(T a, T b)
    {
        return a < b;
    }

    T operator()(T a, T b)
    {
        return min(a, b);
    }
};

} // namespace

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC T aix_atomic_load_explicit(device aix_atomic<T>* object, uint offset)
{
    uint pack_offset = offset / sizeof(T);
    uint elem_offset = offset % sizeof(T);
    uint_or_packed<T> packed_val;
    packed_val.bits = atomic_load_explicit(&(object[pack_offset].val), memory_order_relaxed);
    return packed_val.val[elem_offset];
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_store_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    aix_atomic_update_and_store<T, __None<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_and_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    uint pack_offset = offset / packing_size<T>;
    uint elem_offset = offset % packing_size<T>;
    uint_or_packed<T> identity;
    identity.bits = __UINT32_MAX__;
    identity.val[elem_offset] = val;

    atomic_fetch_and_explicit(&(object[pack_offset].val), identity.bits, memory_order_relaxed);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_or_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    uint pack_offset = offset / packing_size<T>;
    uint elem_offset = offset % packing_size<T>;
    uint_or_packed<T> identity;
    identity.bits = 0;
    identity.val[elem_offset] = val;

    atomic_fetch_or_explicit(&(object[pack_offset].val), identity.bits, memory_order_relaxed);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_min_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    aix_atomic_update_and_store<T, __Min<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_max_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    aix_atomic_update_and_store<T, __Max<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_add_explicit(device aix_atomic<T>* object, T val, uint offset)
{
      aix_atomic_update_and_store<T, __Add<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_mul_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    aix_atomic_update_and_store<T, __Mul<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC bool aix_atomic_compare_exchange_weak_explicit(device aix_atomic<T>* object,
                                                          thread uint* expected,
                                                          uint val,
                                                          uint offset)
{
    return atomic_compare_exchange_weak_explicit(&(object[offset].val), expected, val,
                                                 memory_order_relaxed, memory_order_relaxed);
}


// -----------------------------------------------------------------
// TEMPLATES
// -----------------------------------------------------------------


// Add - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void add_aa(device const T* inA [[buffer(0)]],
                   device const T* inB [[buffer(1)]],
                   device T* result    [[buffer(2)]],
                   uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = inA[index + i] + inB[index + i];
}


// Sub - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void sub_aa(device const T* inA [[buffer(0)]],
                   device const T* inB [[buffer(1)]],
                   device T* result    [[buffer(2)]],
                   uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = inA[index + i] - inB[index + i];
}


// Mul - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void mul_aa(device const T* inA [[buffer(0)]],
                  device const T* inB  [[buffer(1)]],
                  device T* result     [[buffer(2)]],
                  uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = inA[index + i] * inB[index + i];
}


// Div - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void div_aa(device const T* inA [[buffer(0)]],
                   device const T* inB [[buffer(1)]],
                   device T* result    [[buffer(2)]],
                   uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = inA[index + i] / inB[index + i];
}


// Sqrt - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void sqrt_a(device const T* inA [[buffer(0)]],
                   device T* result    [[buffer(1)]],
                   uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = static_cast<T>(sqrt(static_cast<float4>(inA[index + i])));
}


// Sin - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void sin_a(device const T* inA [[buffer(0)]],
                  device T* result    [[buffer(1)]],
                  uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = static_cast<T>(sin(static_cast<float4>(inA[index + i])));
}


// Cos - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void cos_a(device const T* inA [[buffer(0)]],
                  device T* result    [[buffer(1)]],
                  uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = static_cast<T>(cos(static_cast<float4>(inA[index + i])));
}


// Tanh - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void tanh_a(device const T* inA [[buffer(0)]],
                   device T* result    [[buffer(1)]],
                   uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = static_cast<T>(tanh(static_cast<float4>(inA[index + i])));
}


// Log - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void log_a(device const T* inA [[buffer(0)]],
                  device T* result    [[buffer(1)]],
                  uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = static_cast<T>(log(static_cast<float4>(inA[index + i])));
}


// Exp - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void exp_a(device const T* inA [[buffer(0)]],
                  device T* result    [[buffer(1)]],
                  uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = static_cast<T>(exp(static_cast<float4>(inA[index + i])));
}


// Pow - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void pow_aa(device const T* inA  [[buffer(0)]],
                   device const T* expA [[buffer(1)]],
                   device T* result     [[buffer(2)]],
                   uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = static_cast<T>(pow(static_cast<float4>(inA[index + i]), static_cast<float4>(expA[index + i])));
}


// Matrix_Mul
// -----------------------------------------------------------------
template<typename T>
kernel void matrixMul_aa(device const T* inA             [[buffer(0)]],
                         device const T* inB             [[buffer(1)]],
                         device T* result              [[buffer(2)]],
                         constant MatrixSize& matASize [[buffer(3)]],
                         constant MatrixSize& matBSize [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]],
                         uint2 lid [[thread_position_in_threadgroup]])
{
    constexpr int TILE_SIZE = 32;
    size_t M = matASize.rows;
    size_t N = matBSize.cols;
    size_t K = matASize.cols;

    // Allocate shared memory.
    threadgroup T  A[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflict.
    threadgroup T  B[TILE_SIZE][TILE_SIZE + 1];

    // Initialize the sum for each thread's output element.
    T sum = 0;

    // Loop over all tiles.
    for (size_t tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile)
    {
        // Load elements into shared memory.
        size_t aRow = gid.y;
        size_t aCol = tile * TILE_SIZE + lid.x;
        A[lid.y][lid.x] = (aRow < M && aCol < K) ? inA[aRow * K + aCol] : 0;

        size_t bRow = tile * TILE_SIZE + lid.y;
        size_t bCol = gid.x;
        B[lid.x][lid.y] = (bRow < K && bCol < N) ? inB[bRow * N + bCol] : 0;

        // Synchronize to make sure the tile is loaded.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Perform the multiplication for the current tile.
        #pragma clang pragma unroll(full)
        for (size_t k = 0; k < TILE_SIZE; ++k)
            sum += A[lid.y][k] * B[lid.x][k];

        // Synchronize to make sure the computation is done before loading new tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (gid.y < M && gid.x < N)
        result[gid.y * N + gid.x] = sum;
}


// Transpose2D - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void transpose2D_a(device const T* mat          [[buffer(0)]],
                          device T* result             [[buffer(1)]],
                          constant MatrixSize& matSize [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 tid [[thread_position_in_threadgroup]])
{
    uint ofs1 = gid.y * matSize.rows + gid.x;
    uint ofs2 = gid.x * matSize.cols + gid.y;
    result[ofs1] = mat[ofs2];
}


// Transpose - Naive Implementation
// -----------------------------------------------------------------
size_t flattenIndex(thread size_t* indices, size_t indicesSize, device const size_t* strides)
{
    size_t index = 0;
    for (size_t i = 0; i < indicesSize; ++i)
    {
        index += indices[i] * strides[i];
    }
    return index;
}

void unflattenIndex(size_t index, device const size_t* strides, size_t stridesSize, thread size_t* outIndices)
{
    for (size_t i = 0; i < stridesSize; ++i)
    {
        outIndices[i] = index / strides[i];
        index %= strides[i];
    }
}

void swap(thread size_t& a, thread size_t& b)
{
    size_t temp = a;
    a = b;
    b = temp;
}

template<typename T>
kernel void transpose_a(device const T* data            [[buffer(0)]],
                        device T* result                [[buffer(1)]],
                        constant size_t& dim0           [[buffer(2)]],
                        constant size_t& dim1           [[buffer(3)]],
                        device const size_t* strides    [[buffer(4)]],
                        constant size_t& stridesSize    [[buffer(5)]],
                        device const size_t* newStrides [[buffer(6)]],
                        constant size_t& newStridesSize [[buffer(7)]],
                        constant size_t& size           [[buffer(8)]],
                        uint index [[thread_position_in_grid]])
{
    thread size_t oldIndices[16];
    unflattenIndex(index, strides, stridesSize, oldIndices);
    swap(oldIndices[dim0], oldIndices[dim1]);
    size_t newIndex = flattenIndex(oldIndices, stridesSize, newStrides);
    result[newIndex] = data[index];
}


// Copy - Naive Implementation
// -----------------------------------------------------------------
template<typename ST, typename DT>
kernel void copy_aa(device const ST* src [[buffer(0)]],
                    device DT* dst       [[buffer(1)]],
                    uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        dst[index + i] = static_cast<DT>(src[index + i]);
}


// Unary - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void unary_a(device const T* inA [[buffer(0)]],
                    device T* result    [[buffer(1)]],
                    uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = -inA[index + i];
}


// Fill - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void fill_aa(device const T* scalar [[buffer(0)]],
                    device T2* result      [[buffer(1)]],
                    uint index [[thread_position_in_grid]])
{
    T2 scalarVector = static_cast<T2>(scalar[0].xxxx);

    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = scalarVector;
}


// FillMin - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void fillMin_a(device T* result   [[buffer(0)]],
                      uint index [[thread_position_in_grid]])
{
    T minVal = static_cast<T>(numeric_limits<T>::lowest().xxxx);

    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = minVal;
}


// Sum - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void sum_a(device const T* inA [[buffer(0)]],
                  device T* result    [[buffer(1)]],
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]])
{
    const size_t MAX_THREADS = 1024;
    threadgroup T sharedData[MAX_THREADS];
    sharedData[li] = inA[tgi * MAX_THREADS + li];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform parallel reduction in shared memory
    size_t size = threadsPerThreadgroup;
    for (uint stride = size / 2; stride > 0; stride >>= 1)
    {
        if (size % 2 == 1 && li == 0)
            sharedData[0] += sharedData[size-1];
        size >>= 1;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (li < stride)
            sharedData[li] += sharedData[li + stride];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (li == 0)
    {
        result[tgi] = sharedData[0];
    }
}


// Max - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void max_a(device const T* inA [[buffer(0)]],
                  device T* result    [[buffer(1)]],
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]])
{
    const size_t MAX_THREADS = 1024;
    threadgroup T sharedData[MAX_THREADS];
    sharedData[li] = inA[tgi * MAX_THREADS + li];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform parallel reduction in shared memory
    size_t size = threadsPerThreadgroup;
    for (uint stride = size / 2; stride > 0; stride >>= 1)
    {
        if (size % 2 == 1 && li == 0)
            sharedData[0] = sharedData[0] > sharedData[size-1] ? sharedData[0] : sharedData[size-1];
        size >>= 1;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (li < stride)
            sharedData[li] = sharedData[li] > sharedData[li + stride] ? sharedData[li] : sharedData[li + stride];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (li == 0)
    {
        result[tgi] = sharedData[0];
    }
}


// TranslationIndex - Naive Implementation
// -----------------------------------------------------------------
size_t translationIndex(size_t index, device const size_t* shape, device const size_t* newShape,
                        size_t shapeSize, size_t newShapeSize)
{
    size_t originalIndex  = 0;
    size_t targetStride   = 1;
    size_t originalStride = 1;

    for (int64_t i = newShapeSize - 1, j = shapeSize - 1; i >= 0; --i)
    {
        size_t dimIndex = (index / targetStride) % newShape[i];
        if (j >= 0 && shape[j] == newShape[i])
        {
            originalIndex += dimIndex * originalStride;
            originalStride *= shape[--j + 1];
        }
        else if (j >= 0 && shape[j] == 1)
        {
            originalStride *= shape[--j + 1];
        }
        targetStride *= newShape[i];
    }

    return originalIndex;
}


// BroadcastTo - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void broadcastTo_a(device const T* src       [[buffer(0)]],
                          device       T* dst       [[buffer(1)]],
                          device const T2* shape    [[buffer(2)]],
                          device const T2* newShape [[buffer(3)]],
                          constant T2& shapeSize    [[buffer(4)]],
                          constant T2& newShapeSize [[buffer(5)]],
                          uint index [[thread_position_in_grid]])
{
    size_t originalIndex = translationIndex(index, shape, newShape, shapeSize, newShapeSize);
    dst[index] = src[originalIndex];
}


// ReduceTo - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void reduceTo_a(device const T* src       [[buffer(0)]],
                       device       T* dst       [[buffer(1)]],
                       device const T2* shape    [[buffer(2)]],
                       device const T2* newShape [[buffer(3)]],
                       constant T2& shapeSize    [[buffer(4)]],
                       constant T2& newShapeSize [[buffer(5)]],
                       uint index [[thread_position_in_grid]])
{
    size_t originalIndex = translationIndex(index, newShape, shape, newShapeSize, shapeSize);
    atomic_fetch_add_explicit((device atomic<T>*)&(dst[originalIndex]), src[index], memory_order_relaxed);

    // NOTE: Metal Framework supports add and sub operations for only atomic_float, atomic_uint and atomic_uint.
}


// MaxTo - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void maxTo_a(device const T* src       [[buffer(0)]],
                    device       T* dst       [[buffer(1)]],
                    device const T2* shape    [[buffer(2)]],
                    device const T2* newShape [[buffer(3)]],
                    constant T2& shapeSize    [[buffer(4)]],
                    constant T2& newShapeSize [[buffer(5)]],
                    uint index [[thread_position_in_grid]])
{
    size_t originalIndex = translationIndex(index, newShape, shape, newShapeSize, shapeSize);
    aix_atomic_fetch_max_explicit((device aix_atomic<T>*)&(dst[originalIndex]), src[index], memory_order_relaxed);
}


// Slice - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void slice_a(device const T* src       [[buffer(0)]],
                    device       T* dst       [[buffer(1)]],
                    device const T2* shape    [[buffer(2)]],
                    device const T2* newShape [[buffer(3)]],
                    device const T2* strides  [[buffer(4)]],
                    constant T2& shapeSize    [[buffer(5)]],
                    constant T2& newShapeSize [[buffer(6)]],
                    constant T2& stridesSize  [[buffer(7)]],
                    constant T2& dim          [[buffer(8)]],
                    constant T2& start        [[buffer(9)]],
                    constant T2& step         [[buffer(10)]],
                    uint index [[thread_position_in_grid]])
{
    // Translate the flat index into multi-dimensional indices.
    size_t dstIndex = index;
    size_t srcIndex = 0;

    for (int64_t i = static_cast<int64_t>(shapeSize) - 1; i >= 0; --i)
    {
        size_t coordinate = dstIndex % newShape[i];
        dstIndex /= newShape[i];

        if (i == static_cast<int64_t>(dim))   // Handle the slicing dimension.
            srcIndex += (start + coordinate * step) * strides[i];
        else
            srcIndex += coordinate * strides[i];
    }

    dst[index] = src[srcIndex];
}


// SliceSet - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void sliceSet_a(device const T* src       [[buffer(0)]],
                       device       T* dst       [[buffer(1)]],
                       device const T2* shape    [[buffer(2)]],
                       device const T2* newShape [[buffer(3)]],
                       device const T2* strides  [[buffer(4)]],
                       constant T2& shapeSize    [[buffer(5)]],
                       constant T2& newShapeSize [[buffer(6)]],
                       constant T2& stridesSize  [[buffer(7)]],
                       constant T2& dim          [[buffer(8)]],
                       constant T2& start        [[buffer(9)]],
                       constant T2& step         [[buffer(10)]],
                       uint index [[thread_position_in_grid]])
{
    // Translate the flat index into multi-dimensional indices.
    size_t dstIndex = index;
    size_t srcIndex = 0;

    for (int64_t i = static_cast<int64_t>(shapeSize) - 1; i >= 0; --i)
    {
        size_t coordinate = dstIndex % newShape[i];
        dstIndex /= newShape[i];

        if (i == static_cast<int64_t>(dim))   // Handle the slicing dimension.
            srcIndex += (start + coordinate * step) * strides[i];
        else
            srcIndex += coordinate * strides[i];
    }

    dst[srcIndex] = src[index];
}


// Tril - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2, typename T3>
kernel void tril_a(device T* dst             [[buffer(1)]],
                   device const T2* shape    [[buffer(2)]],
                   device const T2* strides  [[buffer(3)]],
                   constant T2& shapeSize    [[buffer(4)]],
                   constant T2& stridesSize  [[buffer(5)]],
                   constant T3& diagonal     [[buffer(6)]],
                   constant T2& size         [[buffer(7)]],
                   uint index [[thread_position_in_grid]])
{
    size_t rows = shape[shapeSize - 2];      // Rows in the last 2-dim tensor.
    size_t cols = shape[shapeSize - 1];      // Columns in the last 2-dim tensor.

    for (size_t i = 0; i < size; ++i)
    {
        // Calculate the row and column indices for the last 2-dim slice.
        size_t row = (i / strides[shapeSize - 2]) % rows;
        size_t col = (i / strides[shapeSize - 1]) % cols;

        // Zero out the elements above the specified diagonal.
        if (static_cast<int64_t>(col) > static_cast<int64_t>(row) + diagonal)
        {
            dst[i] = 0;
        }
    }
}


// nullKernel
// -----------------------------------------------------------------
kernel void nullKernel(uint index [[thread_position_in_grid]])
{
}


// -----------------------------------------------------------------
// TEMPLATE SPECIALIZATIONS
// -----------------------------------------------------------------


// Add
// -----------------------------------------------------------------
template [[ host_name("add_aa_f32") ]]
kernel void add_aa(device const float4*,
                   device const float4*,
                   device float4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("add_aa_f16") ]]
kernel void add_aa(device const half4*,
                   device const half4*,
                   device half4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("add_aa_bf16") ]]
kernel void add_aa(device const bfloat4*,
                   device const bfloat4*,
                   device bfloat4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("add_aa_i64") ]]
kernel void add_aa(device const long4*,
                   device const long4*,
                   device long4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("add_aa_i32") ]]
kernel void add_aa(device const int4*,
                   device const int4*,
                   device int4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("add_aa_i16") ]]
kernel void add_aa(device const short4*,
                   device const short4*,
                   device short4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("add_aa_i8") ]]
kernel void add_aa(device const char4*,
                   device const char4*,
                   device char4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("add_aa_ui8") ]]
kernel void add_aa(device const uchar4*,
                   device const uchar4*,
                   device uchar4*,
                   uint index [[thread_position_in_grid]]);



// Sub
// -----------------------------------------------------------------
template [[ host_name("sub_aa_f32") ]]
kernel void sub_aa(device const float4*,
                   device const float4*,
                   device float4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sub_aa_f16") ]]
kernel void sub_aa(device const half4*,
                   device const half4*,
                   device half4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sub_aa_bf16") ]]
kernel void sub_aa(device const bfloat4*,
                   device const bfloat4*,
                   device bfloat4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sub_aa_i64") ]]
kernel void sub_aa(device const long4*,
                   device const long4*,
                   device long4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sub_aa_i32") ]]
kernel void sub_aa(device const int4*,
                   device const int4*,
                   device int4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sub_aa_i16") ]]
kernel void sub_aa(device const short4*,
                   device const short4*,
                   device short4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sub_aa_i8") ]]
kernel void sub_aa(device const char4*,
                   device const char4*,
                   device char4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sub_aa_ui8") ]]
kernel void sub_aa(device const uchar4*,
                   device const uchar4*,
                   device uchar4*,
                   uint index [[thread_position_in_grid]]);



// Mul
// -----------------------------------------------------------------
template [[ host_name("mul_aa_f32") ]]
kernel void mul_aa(device const float4*,
                   device const float4*,
                   device float4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("mul_aa_f16") ]]
kernel void mul_aa(device const half4*,
                   device const half4*,
                   device half4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("mul_aa_bf16") ]]
kernel void mul_aa(device const bfloat4*,
                   device const bfloat4*,
                   device bfloat4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("mul_aa_i64") ]]
kernel void mul_aa(device const long4*,
                   device const long4*,
                   device long4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("mul_aa_i32") ]]
kernel void mul_aa(device const int4*,
                   device const int4*,
                   device int4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("mul_aa_i16") ]]
kernel void mul_aa(device const short4*,
                   device const short4*,
                   device short4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("mul_aa_i8") ]]
kernel void mul_aa(device const char4*,
                   device const char4*,
                   device char4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("mul_aa_ui8") ]]
kernel void mul_aa(device const uchar4*,
                   device const uchar4*,
                   device uchar4*,
                   uint index [[thread_position_in_grid]]);


// Div
// -----------------------------------------------------------------
template [[ host_name("div_aa_f32") ]]
kernel void div_aa(device const float4*,
                   device const float4*,
                   device float4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("div_aa_f16") ]]
kernel void div_aa(device const half4*,
                   device const half4*,
                   device half4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("div_aa_bf16") ]]
kernel void div_aa(device const bfloat4*,
                   device const bfloat4*,
                   device bfloat4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("div_aa_i64") ]]
kernel void div_aa(device const long4*,
                   device const long4*,
                   device long4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("div_aa_i32") ]]
kernel void div_aa(device const int4*,
                   device const int4*,
                   device int4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("div_aa_i16") ]]
kernel void div_aa(device const short4*,
                   device const short4*,
                   device short4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("div_aa_i8") ]]
kernel void div_aa(device const char4*,
                   device const char4*,
                   device char4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("div_aa_ui8") ]]
kernel void div_aa(device const uchar4*,
                   device const uchar4*,
                   device uchar4*,
                   uint index [[thread_position_in_grid]]);



// Sqrt
// -----------------------------------------------------------------
template [[ host_name("sqrt_a_f32") ]]
kernel void sqrt_a(device const float4*,
                   device float4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sqrt_a_f16") ]]
kernel void sqrt_a(device const half4*,
                   device half4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sqrt_a_bf16") ]]
kernel void sqrt_a(device const bfloat4*,
                   device bfloat4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sqrt_a_i64") ]]
kernel void sqrt_a(device const long4*,
                   device long4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sqrt_a_i32") ]]
kernel void sqrt_a(device const int4*,
                   device int4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sqrt_a_i16") ]]
kernel void sqrt_a(device const short4*,
                   device short4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sqrt_a_i8") ]]
kernel void sqrt_a(device const char4*,
                   device char4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sqrt_a_ui8") ]]
kernel void sqrt_a(device const uchar4*,
                   device uchar4*,
                   uint index [[thread_position_in_grid]]);



// Sin
// -----------------------------------------------------------------
template [[ host_name("sin_a_f32") ]]
kernel void sin_a(device const float4*,
                  device float4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("sin_a_f16") ]]
kernel void sin_a(device const half4*,
                  device half4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("sin_a_bf16") ]]
kernel void sin_a(device const bfloat4*,
                  device bfloat4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("sin_a_i64") ]]
kernel void sin_a(device const long4*,
                  device long4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("sin_a_i32") ]]
kernel void sin_a(device const int4*,
                  device int4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("sin_a_i16") ]]
kernel void sin_a(device const short4*,
                  device short4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("sin_a_i8") ]]
kernel void sin_a(device const char4*,
                  device char4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("sin_a_ui8") ]]
kernel void sin_a(device const uchar4*,
                  device uchar4*,
                  uint index [[thread_position_in_grid]]);


// Cos
// -----------------------------------------------------------------
template [[ host_name("cos_a_f32") ]]
kernel void cos_a(device const float4*,
                  device float4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("cos_a_f16") ]]
kernel void cos_a(device const half4*,
                  device half4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("cos_a_bf16") ]]
kernel void cos_a(device const bfloat4*,
                  device bfloat4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("cos_a_i64") ]]
kernel void cos_a(device const long4*,
                  device long4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("cos_a_i32") ]]
kernel void cos_a(device const int4*,
                  device int4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("cos_a_i16") ]]
kernel void cos_a(device const short4*,
                  device short4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("cos_a_i8") ]]
kernel void cos_a(device const char4*,
                  device char4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("cos_a_ui8") ]]
kernel void cos_a(device const uchar4*,
                  device uchar4*,
                  uint index [[thread_position_in_grid]]);



// Tanh
// -----------------------------------------------------------------
template [[ host_name("tanh_a_f32") ]]
kernel void tanh_a(device const float4*,
                   device float4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tanh_a_f16") ]]
kernel void tanh_a(device const half4*,
                   device half4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tanh_a_bf16") ]]
kernel void tanh_a(device const bfloat4*,
                   device bfloat4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tanh_a_i64") ]]
kernel void tanh_a(device const long4*,
                   device long4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tanh_a_i32") ]]
kernel void tanh_a(device const int4*,
                   device int4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tanh_a_i16") ]]
kernel void tanh_a(device const short4*,
                   device short4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tanh_a_i8") ]]
kernel void tanh_a(device const char4*,
                   device char4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tanh_a_ui8") ]]
kernel void tanh_a(device const uchar4*,
                   device uchar4*,
                   uint index [[thread_position_in_grid]]);



// Log
// -----------------------------------------------------------------
template [[ host_name("log_a_f32") ]]
kernel void log_a(device const float4*,
                  device float4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("log_a_f16") ]]
kernel void log_a(device const half4*,
                  device half4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("log_a_bf16") ]]
kernel void log_a(device const bfloat4*,
                  device bfloat4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("log_a_i64") ]]
kernel void log_a(device const long4*,
                  device long4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("log_a_i32") ]]
kernel void log_a(device const int4*,
                  device int4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("log_a_i16") ]]
kernel void log_a(device const short4*,
                  device short4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("log_a_i8") ]]
kernel void log_a(device const char4*,
                  device char4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("log_a_ui8") ]]
kernel void log_a(device const uchar4*,
                  device uchar4*,
                  uint index [[thread_position_in_grid]]);



// Exp
// -----------------------------------------------------------------
template [[ host_name("exp_a_f32") ]]
kernel void exp_a(device const float4*,
                  device float4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("exp_a_f16") ]]
kernel void exp_a(device const half4*,
                  device half4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("exp_a_bf16") ]]
kernel void exp_a(device const bfloat4*,
                  device bfloat4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("exp_a_i64") ]]
kernel void exp_a(device const long4*,
                  device long4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("exp_a_i32") ]]
kernel void exp_a(device const int4*,
                  device int4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("exp_a_i16") ]]
kernel void exp_a(device const short4*,
                  device short4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("exp_a_i8") ]]
kernel void exp_a(device const char4*,
                  device char4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("exp_a_ui8") ]]
kernel void exp_a(device const uchar4*,
                  device uchar4*,
                  uint index [[thread_position_in_grid]]);


// Pow
// -----------------------------------------------------------------
template [[ host_name("pow_aa_f32") ]]
kernel void pow_aa(device const float4*,
                   device const float4*,
                   device float4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("pow_aa_f16") ]]
kernel void pow_aa(device const half4*,
                   device const half4*,
                   device half4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("pow_aa_bf16") ]]
kernel void pow_aa(device const bfloat4*,
                   device const bfloat4*,
                   device bfloat4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("pow_aa_i64") ]]
kernel void pow_aa(device const long4*,
                   device const long4*,
                   device long4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("pow_aa_i32") ]]
kernel void pow_aa(device const int4*,
                   device const int4*,
                   device int4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("pow_aa_i16") ]]
kernel void pow_aa(device const short4*,
                   device const short4*,
                   device short4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("pow_aa_i8") ]]
kernel void pow_aa(device const char4*,
                   device const char4*,
                   device char4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("pow_aa_ui8") ]]
kernel void pow_aa(device const uchar4*,
                   device const uchar4*,
                   device uchar4*,
                   uint index [[thread_position_in_grid]]);


// Sum
// -----------------------------------------------------------------
template [[ host_name("sum_a_f32") ]]
kernel void sum_a(device const float*,
                  device float*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("sum_a_f16") ]]
kernel void sum_a(device const half*,
                  device half*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("sum_a_bf16") ]]
kernel void sum_a(device const bfloat*,
                  device bfloat*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("sum_a_i64") ]]
kernel void sum_a(device const long*,
                  device long*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("sum_a_i32") ]]
kernel void sum_a(device const int*,
                  device int*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("sum_a_i16") ]]
kernel void sum_a(device const short*,
                  device short*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("sum_a_i8") ]]
kernel void sum_a(device const char*,
                  device char*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("sum_a_ui8") ]]
kernel void sum_a(device const uchar*,
                  device uchar*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);


// Max
// -----------------------------------------------------------------
template [[ host_name("max_a_f32") ]]
kernel void max_a(device const float*,
                  device float*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("max_a_f16") ]]
kernel void max_a(device const half*,
                  device half*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("max_a_bf16") ]]
kernel void max_a(device const bfloat*,
                  device bfloat*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("max_a_i64") ]]
kernel void max_a(device const long*,
                  device long*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("max_a_i32") ]]
kernel void max_a(device const int*,
                  device int*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("max_a_i16") ]]
kernel void max_a(device const short*,
                  device short*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("max_a_i8") ]]
kernel void max_a(device const char*,
                  device char*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("max_a_ui8") ]]
kernel void max_a(device const uchar*,
                  device uchar*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);


// Matrix_Mul
// -----------------------------------------------------------------
template [[ host_name("matrixMul_aa_f32") ]]
kernel void matrixMul_aa(device const float*,
                         device const float*,
                         device float*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]],
                         uint2 lid [[thread_position_in_threadgroup]]);

template [[ host_name("matrixMul_aa_f16") ]]
kernel void matrixMul_aa(device const half*,
                         device const half*,
                         device half*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]],
                         uint2 lid [[thread_position_in_threadgroup]]);

template [[ host_name("matrixMul_aa_bf16") ]]
kernel void matrixMul_aa(device const bfloat*,
                         device const bfloat*,
                         device bfloat*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]],
                         uint2 lid [[thread_position_in_threadgroup]]);

template [[ host_name("matrixMul_aa_i64") ]]
kernel void matrixMul_aa(device const long*,
                         device const long*,
                         device long*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]],
                         uint2 lid [[thread_position_in_threadgroup]]);

template [[ host_name("matrixMul_aa_i32") ]]
kernel void matrixMul_aa(device const int*,
                         device const int*,
                         device int*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]],
                         uint2 lid [[thread_position_in_threadgroup]]);

template [[ host_name("matrixMul_aa_i16") ]]
kernel void matrixMul_aa(device const short*,
                         device const short*,
                         device short*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]],
                         uint2 lid [[thread_position_in_threadgroup]]);

template [[ host_name("matrixMul_aa_i8") ]]
kernel void matrixMul_aa(device const char*,
                         device const char*,
                         device char*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]],
                         uint2 lid [[thread_position_in_threadgroup]]);

template [[ host_name("matrixMul_aa_ui8") ]]
kernel void matrixMul_aa(device const uchar*,
                         device const uchar*,
                         device uchar*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]],
                         uint2 lid [[thread_position_in_threadgroup]]);


// Transpose2D
// -----------------------------------------------------------------
template [[ host_name("transpose2D_a_f32") ]]
kernel void transpose2D_a(device const float*,
                          device float*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);

template [[ host_name("transpose2D_a_f16") ]]
kernel void transpose2D_a(device const half*,
                          device half*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);

template [[ host_name("transpose2D_a_bf16") ]]
kernel void transpose2D_a(device const bfloat*,
                          device bfloat*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);

template [[ host_name("transpose2D_a_i64") ]]
kernel void transpose2D_a(device const long*,
                          device long*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);

template [[ host_name("transpose2D_a_i32") ]]
kernel void transpose2D_a(device const int*,
                          device int*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);

template [[ host_name("transpose2D_a_i16") ]]
kernel void transpose2D_a(device const short*,
                          device short*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);

template [[ host_name("transpose2D_a_i8") ]]
kernel void transpose2D_a(device const char*,
                          device char*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);

template [[ host_name("transpose2D_a_ui8") ]]
kernel void transpose2D_a(device const uchar*,
                          device uchar*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);


// Transpose
// -----------------------------------------------------------------
template [[ host_name("transpose_a_f32") ]]
kernel void transpose_a(device const float* data,
                        device float* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);

template [[ host_name("transpose_a_f16") ]]
kernel void transpose_a(device const half* data,
                        device half* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);

template [[ host_name("transpose_a_bf16") ]]
kernel void transpose_a(device const bfloat* data,
                        device bfloat* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);

template [[ host_name("transpose_a_i64") ]]
kernel void transpose_a(device const long* data,
                        device long* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);

template [[ host_name("transpose_a_i32") ]]
kernel void transpose_a(device const int* data,
                        device int* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);

template [[ host_name("transpose_a_i16") ]]
kernel void transpose_a(device const short* data,
                        device short* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);

template [[ host_name("transpose_a_i8") ]]
kernel void transpose_a(device const char* data,
                        device char* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);

template [[ host_name("transpose_a_ui8") ]]
kernel void transpose_a(device const uchar* data,
                        device uchar* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);


// Copy
// -----------------------------------------------------------------
template [[ host_name("copy_aa_f32_f32") ]]
kernel void copy_aa(device const float4*,
                    device float4*,
                    uint index);

template [[ host_name("copy_aa_f32_f16") ]]
kernel void copy_aa(device const float4*,
                    device half4*,
                    uint index);

template [[ host_name("copy_aa_f32_bf16") ]]
kernel void copy_aa(device const float4*,
                    device bfloat4*,
                    uint index);

template [[ host_name("copy_aa_f32_i64") ]]
kernel void copy_aa(device const float4*,
                    device long4*,
                    uint index);

template [[ host_name("copy_aa_f32_i32") ]]
kernel void copy_aa(device const float4*,
                    device int4*,
                    uint index);

template [[ host_name("copy_aa_f32_i16") ]]
kernel void copy_aa(device const float4*,
                    device short4*,
                    uint index);

template [[ host_name("copy_aa_f32_i8") ]]
kernel void copy_aa(device const float4*,
                    device char4*,
                    uint index);

template [[ host_name("copy_aa_f32_ui8") ]]
kernel void copy_aa(device const float4*,
                    device uchar4*,
                    uint index);

template [[ host_name("copy_aa_f16_f32") ]]
kernel void copy_aa(device const half4*,
                    device float4*,
                    uint index);

template [[ host_name("copy_aa_f16_f16") ]]
kernel void copy_aa(device const half4*,
                    device half4*,
                    uint index);

template [[ host_name("copy_aa_f16_bf16") ]]
kernel void copy_aa(device const half4*,
                    device bfloat4*,
                    uint index);

template [[ host_name("copy_aa_f16_i64") ]]
kernel void copy_aa(device const half4*,
                    device long4*,
                    uint index);

template [[ host_name("copy_aa_f16_i32") ]]
kernel void copy_aa(device const half4*,
                    device int4*,
                    uint index);

template [[ host_name("copy_aa_f16_i16") ]]
kernel void copy_aa(device const half4*,
                    device short4*,
                    uint index);

template [[ host_name("copy_aa_f16_i8") ]]
kernel void copy_aa(device const half4*,
                    device char4*,
                    uint index);

template [[ host_name("copy_aa_f16_ui8") ]]
kernel void copy_aa(device const half4*,
                    device uchar4*,
                    uint index);

template [[ host_name("copy_aa_bf16_f32") ]]
kernel void copy_aa(device const bfloat4*,
                    device float4*,
                    uint index);

template [[ host_name("copy_aa_bf16_f16") ]]
kernel void copy_aa(device const bfloat4*,
                    device half4*,
                    uint index);

template [[ host_name("copy_aa_bf16_bf16") ]]
kernel void copy_aa(device const bfloat4*,
                    device bfloat4*,
                    uint index);

template [[ host_name("copy_aa_bf16_i64") ]]
kernel void copy_aa(device const bfloat4*,
                    device long4*,
                    uint index);

template [[ host_name("copy_aa_bf16_i32") ]]
kernel void copy_aa(device const bfloat4*,
                    device int4*,
                    uint index);

template [[ host_name("copy_aa_bf16_i16") ]]
kernel void copy_aa(device const bfloat4*,
                    device short4*,
                    uint index);

template [[ host_name("copy_aa_bf16_i8") ]]
kernel void copy_aa(device const bfloat4*,
                    device char4*,
                    uint index);

template [[ host_name("copy_aa_bf16_ui8") ]]
kernel void copy_aa(device const bfloat4*,
                    device uchar4*,
                    uint index);

template [[ host_name("copy_aa_i64_f32") ]]
kernel void copy_aa(device const long4*,
                    device float4*,
                    uint index);

template [[ host_name("copy_aa_i64_f16") ]]
kernel void copy_aa(device const long4*,
                    device half4*,
                    uint index);

template [[ host_name("copy_aa_i64_bf16") ]]
kernel void copy_aa(device const long4*,
                    device bfloat4*,
                    uint index);

template [[ host_name("copy_aa_i64_i64") ]]
kernel void copy_aa(device const long4*,
                    device long4*,
                    uint index);

template [[ host_name("copy_aa_i64_i32") ]]
kernel void copy_aa(device const long4*,
                    device int4*,
                    uint index);

template [[ host_name("copy_aa_i64_i16") ]]
kernel void copy_aa(device const long4*,
                    device short4*,
                    uint index);

template [[ host_name("copy_aa_i64_i8") ]]
kernel void copy_aa(device const long4*,
                    device char4*,
                    uint index);

template [[ host_name("copy_aa_i64_ui8") ]]
kernel void copy_aa(device const long4*,
                    device uchar4*,
                    uint index);

template [[ host_name("copy_aa_i32_f32") ]]
kernel void copy_aa(device const int4*,
                    device float4*,
                    uint index);

template [[ host_name("copy_aa_i32_f16") ]]
kernel void copy_aa(device const int4*,
                    device half4*,
                    uint index);

template [[ host_name("copy_aa_i32_bf16") ]]
kernel void copy_aa(device const int4*,
                    device bfloat4*,
                    uint index);

template [[ host_name("copy_aa_i32_i64") ]]
kernel void copy_aa(device const int4*,
                    device long4*,
                    uint index);

template [[ host_name("copy_aa_i32_i32") ]]
kernel void copy_aa(device const int4*,
                    device int4*,
                    uint index);

template [[ host_name("copy_aa_i32_i16") ]]
kernel void copy_aa(device const int4*,
                    device short4*,
                    uint index);

template [[ host_name("copy_aa_i32_i8") ]]
kernel void copy_aa(device const int4*,
                    device char4*,
                    uint index);

template [[ host_name("copy_aa_i32_ui8") ]]
kernel void copy_aa(device const int4*,
                    device uchar4*,
                    uint index);

template [[ host_name("copy_aa_i16_f32") ]]
kernel void copy_aa(device const short4*,
                    device float4*,
                    uint index);

template [[ host_name("copy_aa_i16_f16") ]]
kernel void copy_aa(device const short4*,
                    device half4*,
                    uint index);

template [[ host_name("copy_aa_i16_bf16") ]]
kernel void copy_aa(device const short4*,
                    device bfloat4*,
                    uint index);

template [[ host_name("copy_aa_i16_i64") ]]
kernel void copy_aa(device const short4*,
                    device long4*,
                    uint index);

template [[ host_name("copy_aa_i16_i32") ]]
kernel void copy_aa(device const short4*,
                    device int4*,
                    uint index);

template [[ host_name("copy_aa_i16_i16") ]]
kernel void copy_aa(device const short4*,
                    device short4*,
                    uint index);

template [[ host_name("copy_aa_i16_i8") ]]
kernel void copy_aa(device const short4*,
                    device char4*,
                    uint index);

template [[ host_name("copy_aa_i16_ui8") ]]
kernel void copy_aa(device const short4*,
                    device uchar4*,
                    uint index);

template [[ host_name("copy_aa_i8_f32") ]]
kernel void copy_aa(device const char4*,
                    device float4*,
                    uint index);

template [[ host_name("copy_aa_i8_f16") ]]
kernel void copy_aa(device const char4*,
                    device half4*,
                    uint index);

template [[ host_name("copy_aa_i8_bf16") ]]
kernel void copy_aa(device const char4*,
                    device bfloat4*,
                    uint index);

template [[ host_name("copy_aa_i8_i64") ]]
kernel void copy_aa(device const char4*,
                    device long4*,
                    uint index);

template [[ host_name("copy_aa_i8_i32") ]]
kernel void copy_aa(device const char4*,
                    device int4*,
                    uint index);

template [[ host_name("copy_aa_i8_i16") ]]
kernel void copy_aa(device const char4*,
                    device short4*,
                    uint index);

template [[ host_name("copy_aa_i8_i8") ]]
kernel void copy_aa(device const char4*,
                    device char4*,
                    uint index);

template [[ host_name("copy_aa_i8_ui8") ]]
kernel void copy_aa(device const char4*,
                    device uchar4*,
                    uint index);

template [[ host_name("copy_aa_ui8_f32") ]]
kernel void copy_aa(device const uchar4*,
                    device float4*,
                    uint index);

template [[ host_name("copy_aa_ui8_f16") ]]
kernel void copy_aa(device const uchar4*,
                    device half4*,
                    uint index);

template [[ host_name("copy_aa_ui8_bf16") ]]
kernel void copy_aa(device const uchar4*,
                    device bfloat4*,
                    uint index);

template [[ host_name("copy_aa_ui8_i64") ]]
kernel void copy_aa(device const uchar4*,
                    device long4*,
                    uint index);

template [[ host_name("copy_aa_ui8_i32") ]]
kernel void copy_aa(device const uchar4*,
                    device int4*,
                    uint index);

template [[ host_name("copy_aa_ui8_i16") ]]
kernel void copy_aa(device const uchar4*,
                    device short4*,
                    uint index);

template [[ host_name("copy_aa_ui8_i8") ]]
kernel void copy_aa(device const uchar4*,
                    device char4*,
                    uint index);

template [[ host_name("copy_aa_ui8_ui8") ]]
kernel void copy_aa(device const uchar4*,
                    device uchar4*,
                    uint index);


// Unary
// -----------------------------------------------------------------
template [[ host_name("unary_a_f32") ]]
kernel void unary_a(device const float4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("unary_a_f16") ]]
kernel void unary_a(device const half4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("unary_a_bf16") ]]
kernel void unary_a(device const bfloat4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("unary_a_i64") ]]
kernel void unary_a(device const long4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("unary_a_i32") ]]
kernel void unary_a(device const int4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("unary_a_i16") ]]
kernel void unary_a(device const short4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("unary_a_i8") ]]
kernel void unary_a(device const char4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("unary_a_ui8") ]]
kernel void unary_a(device const uchar4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);


// Fill
// -----------------------------------------------------------------
template [[ host_name("fill_aa_f32_f32") ]]
kernel void fill_aa(device const float4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f32_f16") ]]
kernel void fill_aa(device const float4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f32_bf16") ]]
kernel void fill_aa(device const float4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f32_i64") ]]
kernel void fill_aa(device const float4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f32_i32") ]]
kernel void fill_aa(device const float4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f32_i16") ]]
kernel void fill_aa(device const float4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f32_i8") ]]
kernel void fill_aa(device const float4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f32_ui8") ]]
kernel void fill_aa(device const float4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_f32") ]]
kernel void fill_aa(device const half4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_f16") ]]
kernel void fill_aa(device const half4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_bf16") ]]
kernel void fill_aa(device const half4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_i64") ]]
kernel void fill_aa(device const half4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_i32") ]]
kernel void fill_aa(device const half4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_i16") ]]
kernel void fill_aa(device const half4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_i8") ]]
kernel void fill_aa(device const half4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_ui8") ]]
kernel void fill_aa(device const half4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_f32") ]]
kernel void fill_aa(device const bfloat4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_f16") ]]
kernel void fill_aa(device const bfloat4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_bf16") ]]
kernel void fill_aa(device const bfloat4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_i64") ]]
kernel void fill_aa(device const bfloat4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_i32") ]]
kernel void fill_aa(device const bfloat4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_i16") ]]
kernel void fill_aa(device const bfloat4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_i8") ]]
kernel void fill_aa(device const bfloat4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_ui8") ]]
kernel void fill_aa(device const bfloat4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_f32") ]]
kernel void fill_aa(device const long4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_f16") ]]
kernel void fill_aa(device const long4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_bf16") ]]
kernel void fill_aa(device const long4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_i64") ]]
kernel void fill_aa(device const long4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_i32") ]]
kernel void fill_aa(device const long4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_i16") ]]
kernel void fill_aa(device const long4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_i8") ]]
kernel void fill_aa(device const long4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_ui8") ]]
kernel void fill_aa(device const long4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_f32") ]]
kernel void fill_aa(device const int4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_f16") ]]
kernel void fill_aa(device const int4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_bf16") ]]
kernel void fill_aa(device const int4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_i64") ]]
kernel void fill_aa(device const int4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_i32") ]]
kernel void fill_aa(device const int4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_i16") ]]
kernel void fill_aa(device const int4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_i8") ]]
kernel void fill_aa(device const int4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_ui8") ]]
kernel void fill_aa(device const int4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_f32") ]]
kernel void fill_aa(device const short4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_f16") ]]
kernel void fill_aa(device const short4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_bf16") ]]
kernel void fill_aa(device const short4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_i64") ]]
kernel void fill_aa(device const short4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_i32") ]]
kernel void fill_aa(device const short4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_i16") ]]
kernel void fill_aa(device const short4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_i8") ]]
kernel void fill_aa(device const short4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_ui8") ]]
kernel void fill_aa(device const short4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_f32") ]]
kernel void fill_aa(device const char4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_f16") ]]
kernel void fill_aa(device const char4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_bf16") ]]
kernel void fill_aa(device const char4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_i64") ]]
kernel void fill_aa(device const char4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_i32") ]]
kernel void fill_aa(device const char4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_i16") ]]
kernel void fill_aa(device const char4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_i8") ]]
kernel void fill_aa(device const char4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_ui8") ]]
kernel void fill_aa(device const char4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_f32") ]]
kernel void fill_aa(device const uchar4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_f16") ]]
kernel void fill_aa(device const uchar4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_bf16") ]]
kernel void fill_aa(device const uchar4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_i64") ]]
kernel void fill_aa(device const uchar4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_i32") ]]
kernel void fill_aa(device const uchar4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_i16") ]]
kernel void fill_aa(device const uchar4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_i8") ]]
kernel void fill_aa(device const uchar4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_ui8") ]]
kernel void fill_aa(device const uchar4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);


// FillMin
// -----------------------------------------------------------------
template [[ host_name("fillMin_a_f32") ]]
kernel void fillMin_a(device float4*,
                      uint index [[thread_position_in_grid]]);

template [[ host_name("fillMin_a_f16") ]]
kernel void fillMin_a(device half4*,
                      uint index [[thread_position_in_grid]]);

template [[ host_name("fillMin_a_bf16") ]]
kernel void fillMin_a(device bfloat4*,
                      uint index [[thread_position_in_grid]]);

template [[ host_name("fillMin_a_i64") ]]
kernel void fillMin_a(device long4*,
                      uint index [[thread_position_in_grid]]);

template [[ host_name("fillMin_a_i32") ]]
kernel void fillMin_a(device int4*,
                      uint index [[thread_position_in_grid]]);

template [[ host_name("fillMin_a_i16") ]]
kernel void fillMin_a(device short4*,
                      uint index [[thread_position_in_grid]]);

template [[ host_name("fillMin_a_i8") ]]
kernel void fillMin_a(device char4*,
                      uint index [[thread_position_in_grid]]);

template [[ host_name("fillMin_a_ui8") ]]
kernel void fillMin_a(device uchar4*,
                      uint index [[thread_position_in_grid]]);


// BroadcastTo
// -----------------------------------------------------------------
template [[ host_name("broadcastTo_a_f32") ]]
kernel void broadcastTo_a(device const float* src,
                          device       float* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);

template [[ host_name("broadcastTo_a_f16") ]]
kernel void broadcastTo_a(device const half* src,
                          device       half* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);

template [[ host_name("broadcastTo_a_bf16") ]]
kernel void broadcastTo_a(device const bfloat* src,
                          device       bfloat* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);

template [[ host_name("broadcastTo_a_i64") ]]
kernel void broadcastTo_a(device const int64_t* src,
                          device       int64_t* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);

template [[ host_name("broadcastTo_a_i32") ]]
kernel void broadcastTo_a(device const int32_t* src,
                          device       int32_t* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);

template [[ host_name("broadcastTo_a_i16") ]]
kernel void broadcastTo_a(device const int16_t* src,
                          device       int16_t* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);

template [[ host_name("broadcastTo_a_i8") ]]
kernel void broadcastTo_a(device const int8_t* src,
                          device       int8_t* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);

template [[ host_name("broadcastTo_a_ui8") ]]
kernel void broadcastTo_a(device const uint8_t* src,
                          device       uint8_t* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);


// ReduceTo
// -----------------------------------------------------------------
template [[ host_name("reduceTo_a_f32") ]]
kernel void reduceTo_a(device const float* src,
                       device       float* dst,
                       device const size_t* shape,
                       device const size_t* newShape,
                       constant size_t& shapeSize,
                       constant size_t& newShapeSize,
                       uint index [[thread_position_in_grid]]);

template [[ host_name("reduceTo_a_i32") ]]
kernel void reduceTo_a(device const int* src,
                       device       int* dst,
                       device const size_t* shape,
                       device const size_t* newShape,
                       constant size_t& shapeSize,
                       constant size_t& newShapeSize,
                       uint index [[thread_position_in_grid]]);

// IMPORTANT NOTE: The following specialization is just a dummy kernel. Not for use since atomic<ulong> is not supported.

kernel void reduceTo_a_f16(device const half* src,
                           device       half* dst,
                           device const size_t* shape,
                           device const size_t* newShape,
                           constant size_t& shapeSize,
                           constant size_t& newShapeSize,
                           uint index [[thread_position_in_grid]]) { }

kernel void reduceTo_a_bf16(device const bfloat* src,
                            device       bfloat* dst,
                            device const size_t* shape,
                            device const size_t* newShape,
                            constant size_t& shapeSize,
                            constant size_t& newShapeSize,
                            uint index [[thread_position_in_grid]]) { }

kernel void reduceTo_a_i64(device const int64_t* src,
                           device       int64_t* dst,
                           device const size_t* shape,
                           device const size_t* newShape,
                           constant size_t& shapeSize,
                           constant size_t& newShapeSize,
                           uint index [[thread_position_in_grid]]) { }

kernel void reduceTo_a_i16(device const int16_t* src,
                           device       int16_t* dst,
                           device const size_t* shape,
                           device const size_t* newShape,
                           constant size_t& shapeSize,
                           constant size_t& newShapeSize,
                           uint index [[thread_position_in_grid]]) { }

kernel void reduceTo_a_i8(device const int8_t* src,
                          device       int8_t* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]) { }

kernel void reduceTo_a_ui8(device const uint8_t* src,
                           device       uint8_t* dst,
                           device const size_t* shape,
                           device const size_t* newShape,
                           constant size_t& shapeSize,
                           constant size_t& newShapeSize,
                           uint index [[thread_position_in_grid]]) { }


// MaxTo
// -----------------------------------------------------------------
template [[ host_name("maxTo_a_f32") ]]
kernel void maxTo_a(device const float* src,
                    device       float* dst,
                    device const size_t* shape,
                    device const size_t* newShape,
                    constant size_t& shapeSize,
                    constant size_t& newShapeSize,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("maxTo_a_i32") ]]
kernel void maxTo_a(device const int* src,
                    device       int* dst,
                    device const size_t* shape,
                    device const size_t* newShape,
                    constant size_t& shapeSize,
                    constant size_t& newShapeSize,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("maxTo_a_f16") ]]
kernel void maxTo_a(device const half* src,
                    device       half* dst,
                    device const size_t* shape,
                    device const size_t* newShape,
                    constant size_t& shapeSize,
                    constant size_t& newShapeSize,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("maxTo_a_i16") ]]
kernel void maxTo_a(device const int16_t* src,
                    device       int16_t* dst,
                    device const size_t* shape,
                    device const size_t* newShape,
                    constant size_t& shapeSize,
                    constant size_t& newShapeSize,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("maxTo_a_i8") ]]
kernel void maxTo_a(device const int8_t* src,
                    device       int8_t* dst,
                    device const size_t* shape,
                    device const size_t* newShape,
                    constant size_t& shapeSize,
                    constant size_t& newShapeSize,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("maxTo_a_ui8") ]]
kernel void maxTo_a(device const uint8_t* src,
                    device       uint8_t* dst,
                    device const size_t* shape,
                    device const size_t* newShape,
                    constant size_t& shapeSize,
                    constant size_t& newShapeSize,
                    uint index [[thread_position_in_grid]]);

// IMPORTANT NOTE: The following specialization is just a dummy kernel. Currently the formats are not supported.

kernel void maxTo_a_bf16(device const bfloat* src,
                         device       bfloat* dst,
                         device const size_t* shape,
                         device const size_t* newShape,
                         constant size_t& shapeSize,
                         constant size_t& newShapeSize,
                         uint index [[thread_position_in_grid]]) { }

kernel void maxTo_a_i64(device const int64_t* src,
                        device       int64_t* dst,
                        device const size_t* shape,
                        device const size_t* newShape,
                        constant size_t& shapeSize,
                        constant size_t& newShapeSize,
                        uint index [[thread_position_in_grid]]) { }


// Slice
// -----------------------------------------------------------------
template [[ host_name("slice_a_f32") ]]
kernel void slice_a(device const float* src       [[buffer(0)]],
                    device       float* dst       [[buffer(1)]],
                    device const size_t* shape    [[buffer(2)]],
                    device const size_t* newShape [[buffer(3)]],
                    device const size_t* strides  [[buffer(4)]],
                    constant size_t& shapeSize    [[buffer(5)]],
                    constant size_t& newShapeSize [[buffer(6)]],
                    constant size_t& stridesSize  [[buffer(7)]],
                    constant size_t& dim          [[buffer(8)]],
                    constant size_t& start        [[buffer(9)]],
                    constant size_t& step         [[buffer(10)]],
                    uint index [[thread_position_in_grid]]);

template [[ host_name("slice_a_f16") ]]
kernel void slice_a(device const half* src        [[buffer(0)]],
                    device       half* dst        [[buffer(1)]],
                    device const size_t* shape    [[buffer(2)]],
                    device const size_t* newShape [[buffer(3)]],
                    device const size_t* strides  [[buffer(4)]],
                    constant size_t& shapeSize    [[buffer(5)]],
                    constant size_t& newShapeSize [[buffer(6)]],
                    constant size_t& stridesSize  [[buffer(7)]],
                    constant size_t& dim          [[buffer(8)]],
                    constant size_t& start        [[buffer(9)]],
                    constant size_t& step         [[buffer(10)]],
                    uint index [[thread_position_in_grid]]);

template [[ host_name("slice_a_bf16") ]]
kernel void slice_a(device const bfloat* src        [[buffer(0)]],
                    device       bfloat* dst        [[buffer(1)]],
                    device const size_t* shape    [[buffer(2)]],
                    device const size_t* newShape [[buffer(3)]],
                    device const size_t* strides  [[buffer(4)]],
                    constant size_t& shapeSize    [[buffer(5)]],
                    constant size_t& newShapeSize [[buffer(6)]],
                    constant size_t& stridesSize  [[buffer(7)]],
                    constant size_t& dim          [[buffer(8)]],
                    constant size_t& start        [[buffer(9)]],
                    constant size_t& step         [[buffer(10)]],
                    uint index [[thread_position_in_grid]]);

template [[ host_name("slice_a_i64") ]]
kernel void slice_a(device const long* src        [[buffer(0)]],
                    device       long* dst        [[buffer(1)]],
                    device const size_t* shape    [[buffer(2)]],
                    device const size_t* newShape [[buffer(3)]],
                    device const size_t* strides  [[buffer(4)]],
                    constant size_t& shapeSize    [[buffer(5)]],
                    constant size_t& newShapeSize [[buffer(6)]],
                    constant size_t& stridesSize  [[buffer(7)]],
                    constant size_t& dim          [[buffer(8)]],
                    constant size_t& start        [[buffer(9)]],
                    constant size_t& step         [[buffer(10)]],
                    uint index [[thread_position_in_grid]]);

template [[ host_name("slice_a_i32") ]]
kernel void slice_a(device const int* src         [[buffer(0)]],
                    device       int* dst         [[buffer(1)]],
                    device const size_t* shape    [[buffer(2)]],
                    device const size_t* newShape [[buffer(3)]],
                    device const size_t* strides  [[buffer(4)]],
                    constant size_t& shapeSize    [[buffer(5)]],
                    constant size_t& newShapeSize [[buffer(6)]],
                    constant size_t& stridesSize  [[buffer(7)]],
                    constant size_t& dim          [[buffer(8)]],
                    constant size_t& start        [[buffer(9)]],
                    constant size_t& step         [[buffer(10)]],
                    uint index [[thread_position_in_grid]]);

template [[ host_name("slice_a_i16") ]]
kernel void slice_a(device const short* src       [[buffer(0)]],
                    device       short* dst       [[buffer(1)]],
                    device const size_t* shape    [[buffer(2)]],
                    device const size_t* newShape [[buffer(3)]],
                    device const size_t* strides  [[buffer(4)]],
                    constant size_t& shapeSize    [[buffer(5)]],
                    constant size_t& newShapeSize [[buffer(6)]],
                    constant size_t& stridesSize  [[buffer(7)]],
                    constant size_t& dim          [[buffer(8)]],
                    constant size_t& start        [[buffer(9)]],
                    constant size_t& step         [[buffer(10)]],
                    uint index [[thread_position_in_grid]]);

template [[ host_name("slice_a_i8") ]]
kernel void slice_a(device const char* src        [[buffer(0)]],
                    device       char* dst        [[buffer(1)]],
                    device const size_t* shape    [[buffer(2)]],
                    device const size_t* newShape [[buffer(3)]],
                    device const size_t* strides  [[buffer(4)]],
                    constant size_t& shapeSize    [[buffer(5)]],
                    constant size_t& newShapeSize [[buffer(6)]],
                    constant size_t& stridesSize  [[buffer(7)]],
                    constant size_t& dim          [[buffer(8)]],
                    constant size_t& start        [[buffer(9)]],
                    constant size_t& step         [[buffer(10)]],
                    uint index [[thread_position_in_grid]]);

template [[ host_name("slice_a_ui8") ]]
kernel void slice_a(device const unsigned char* src  [[buffer(0)]],
                    device       unsigned char* dst  [[buffer(1)]],
                    device const size_t* shape       [[buffer(2)]],
                    device const size_t* newShape    [[buffer(3)]],
                    device const size_t* strides     [[buffer(4)]],
                    constant size_t& shapeSize       [[buffer(5)]],
                    constant size_t& newShapeSize    [[buffer(6)]],
                    constant size_t& stridesSize     [[buffer(7)]],
                    constant size_t& dim             [[buffer(8)]],
                    constant size_t& start           [[buffer(9)]],
                    constant size_t& step            [[buffer(10)]],
                    uint index [[thread_position_in_grid]]);


// SliceSet
// -----------------------------------------------------------------
template [[ host_name("sliceSet_a_f32") ]]
kernel void sliceSet_a(device const float* src       [[buffer(0)]],
                       device       float* dst       [[buffer(1)]],
                       device const size_t* shape    [[buffer(2)]],
                       device const size_t* newShape [[buffer(3)]],
                       device const size_t* strides  [[buffer(4)]],
                       constant size_t& shapeSize    [[buffer(5)]],
                       constant size_t& newShapeSize [[buffer(6)]],
                       constant size_t& stridesSize  [[buffer(7)]],
                       constant size_t& dim          [[buffer(8)]],
                       constant size_t& start        [[buffer(9)]],
                       constant size_t& step         [[buffer(10)]],
                       uint index [[thread_position_in_grid]]);

template [[ host_name("sliceSet_a_f16") ]]
kernel void sliceSet_a(device const half* src        [[buffer(0)]],
                       device       half* dst        [[buffer(1)]],
                       device const size_t* shape    [[buffer(2)]],
                       device const size_t* newShape [[buffer(3)]],
                       device const size_t* strides  [[buffer(4)]],
                       constant size_t& shapeSize    [[buffer(5)]],
                       constant size_t& newShapeSize [[buffer(6)]],
                       constant size_t& stridesSize  [[buffer(7)]],
                       constant size_t& dim          [[buffer(8)]],
                       constant size_t& start        [[buffer(9)]],
                       constant size_t& step         [[buffer(10)]],
                       uint index [[thread_position_in_grid]]);

template [[ host_name("sliceSet_a_bf16") ]]
kernel void sliceSet_a(device const bfloat* src        [[buffer(0)]],
                       device       bfloat* dst        [[buffer(1)]],
                       device const size_t* shape    [[buffer(2)]],
                       device const size_t* newShape [[buffer(3)]],
                       device const size_t* strides  [[buffer(4)]],
                       constant size_t& shapeSize    [[buffer(5)]],
                       constant size_t& newShapeSize [[buffer(6)]],
                       constant size_t& stridesSize  [[buffer(7)]],
                       constant size_t& dim          [[buffer(8)]],
                       constant size_t& start        [[buffer(9)]],
                       constant size_t& step         [[buffer(10)]],
                       uint index [[thread_position_in_grid]]);

template [[ host_name("sliceSet_a_i64") ]]
kernel void sliceSet_a(device const long* src        [[buffer(0)]],
                       device       long* dst        [[buffer(1)]],
                       device const size_t* shape    [[buffer(2)]],
                       device const size_t* newShape [[buffer(3)]],
                       device const size_t* strides  [[buffer(4)]],
                       constant size_t& shapeSize    [[buffer(5)]],
                       constant size_t& newShapeSize [[buffer(6)]],
                       constant size_t& stridesSize  [[buffer(7)]],
                       constant size_t& dim          [[buffer(8)]],
                       constant size_t& start        [[buffer(9)]],
                       constant size_t& step         [[buffer(10)]],
                       uint index [[thread_position_in_grid]]);

template [[ host_name("sliceSet_a_i32") ]]
kernel void sliceSet_a(device const int* src         [[buffer(0)]],
                       device       int* dst         [[buffer(1)]],
                       device const size_t* shape    [[buffer(2)]],
                       device const size_t* newShape [[buffer(3)]],
                       device const size_t* strides  [[buffer(4)]],
                       constant size_t& shapeSize    [[buffer(5)]],
                       constant size_t& newShapeSize [[buffer(6)]],
                       constant size_t& stridesSize  [[buffer(7)]],
                       constant size_t& dim          [[buffer(8)]],
                       constant size_t& start        [[buffer(9)]],
                       constant size_t& step         [[buffer(10)]],
                       uint index [[thread_position_in_grid]]);

template [[ host_name("sliceSet_a_i16") ]]
kernel void sliceSet_a(device const short* src       [[buffer(0)]],
                       device       short* dst       [[buffer(1)]],
                       device const size_t* shape    [[buffer(2)]],
                       device const size_t* newShape [[buffer(3)]],
                       device const size_t* strides  [[buffer(4)]],
                       constant size_t& shapeSize    [[buffer(5)]],
                       constant size_t& newShapeSize [[buffer(6)]],
                       constant size_t& stridesSize  [[buffer(7)]],
                       constant size_t& dim          [[buffer(8)]],
                       constant size_t& start        [[buffer(9)]],
                       constant size_t& step         [[buffer(10)]],
                       uint index [[thread_position_in_grid]]);

template [[ host_name("sliceSet_a_i8") ]]
kernel void sliceSet_a(device const char* src        [[buffer(0)]],
                       device       char* dst        [[buffer(1)]],
                       device const size_t* shape    [[buffer(2)]],
                       device const size_t* newShape [[buffer(3)]],
                       device const size_t* strides  [[buffer(4)]],
                       constant size_t& shapeSize    [[buffer(5)]],
                       constant size_t& newShapeSize [[buffer(6)]],
                       constant size_t& stridesSize  [[buffer(7)]],
                       constant size_t& dim          [[buffer(8)]],
                       constant size_t& start        [[buffer(9)]],
                       constant size_t& step         [[buffer(10)]],
                       uint index [[thread_position_in_grid]]);

template [[ host_name("sliceSet_a_ui8") ]]
kernel void sliceSet_a(device const unsigned char* src  [[buffer(0)]],
                       device       unsigned char* dst  [[buffer(1)]],
                       device const size_t* shape       [[buffer(2)]],
                       device const size_t* newShape    [[buffer(3)]],
                       device const size_t* strides     [[buffer(4)]],
                       constant size_t& shapeSize       [[buffer(5)]],
                       constant size_t& newShapeSize    [[buffer(6)]],
                       constant size_t& stridesSize     [[buffer(7)]],
                       constant size_t& dim             [[buffer(8)]],
                       constant size_t& start           [[buffer(9)]],
                       constant size_t& step            [[buffer(10)]],
                       uint index [[thread_position_in_grid]]);


// Tril
// -----------------------------------------------------------------
template [[ host_name("tril_a_f32") ]]
kernel void tril_a(device float* dst             [[buffer(1)]],
                   device const size_t* shape    [[buffer(2)]],
                   device const size_t* strides  [[buffer(3)]],
                   constant size_t& shapeSize    [[buffer(4)]],
                   constant size_t& stridesSize  [[buffer(5)]],
                   constant int64_t& diagonal    [[buffer(6)]],
                   constant size_t& size         [[buffer(7)]],
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tril_a_f16") ]]
kernel void tril_a(device half* dst              [[buffer(1)]],
                   device const size_t* shape    [[buffer(2)]],
                   device const size_t* strides  [[buffer(3)]],
                   constant size_t& shapeSize    [[buffer(4)]],
                   constant size_t& stridesSize  [[buffer(5)]],
                   constant int64_t& diagonal    [[buffer(6)]],
                   constant size_t& size         [[buffer(7)]],
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tril_a_bf16") ]]
kernel void tril_a(device bfloat* dst            [[buffer(1)]],
                   device const size_t* shape    [[buffer(2)]],
                   device const size_t* strides  [[buffer(3)]],
                   constant size_t& shapeSize    [[buffer(4)]],
                   constant size_t& stridesSize  [[buffer(5)]],
                   constant int64_t& diagonal    [[buffer(6)]],
                   constant size_t& size         [[buffer(7)]],
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tril_a_i64") ]]
kernel void tril_a(device long* dst              [[buffer(1)]],
                   device const size_t* shape    [[buffer(2)]],
                   device const size_t* strides  [[buffer(3)]],
                   constant size_t& shapeSize    [[buffer(4)]],
                   constant size_t& stridesSize  [[buffer(5)]],
                   constant int64_t& diagonal    [[buffer(6)]],
                   constant size_t& size         [[buffer(7)]],
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tril_a_i32") ]]
kernel void tril_a(device int* dst               [[buffer(1)]],
                   device const size_t* shape    [[buffer(2)]],
                   device const size_t* strides  [[buffer(3)]],
                   constant size_t& shapeSize    [[buffer(4)]],
                   constant size_t& stridesSize  [[buffer(5)]],
                   constant int64_t& diagonal    [[buffer(6)]],
                   constant size_t& size         [[buffer(7)]],
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tril_a_i16") ]]
kernel void tril_a(device short* dst             [[buffer(1)]],
                   device const size_t* shape    [[buffer(2)]],
                   device const size_t* strides  [[buffer(3)]],
                   constant size_t& shapeSize    [[buffer(4)]],
                   constant size_t& stridesSize  [[buffer(5)]],
                   constant int64_t& diagonal    [[buffer(6)]],
                   constant size_t& size         [[buffer(7)]],
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tril_a_i8") ]]
kernel void tril_a(device char* dst              [[buffer(1)]],
                   device const size_t* shape    [[buffer(2)]],
                   device const size_t* strides  [[buffer(3)]],
                   constant size_t& shapeSize    [[buffer(4)]],
                   constant size_t& stridesSize  [[buffer(5)]],
                   constant int64_t& diagonal    [[buffer(6)]],
                   constant size_t& size         [[buffer(7)]],
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tril_a_ui8") ]]
kernel void tril_a(device unsigned char* dst        [[buffer(1)]],
                   device const size_t* shape       [[buffer(2)]],
                   device const size_t* strides     [[buffer(3)]],
                   constant size_t& shapeSize       [[buffer(4)]],
                   constant size_t& stridesSize     [[buffer(5)]],
                   constant int64_t& diagonal       [[buffer(6)]],
                   constant size_t& size            [[buffer(7)]],
                   uint index [[thread_position_in_grid]]);

)";


}   // namespace