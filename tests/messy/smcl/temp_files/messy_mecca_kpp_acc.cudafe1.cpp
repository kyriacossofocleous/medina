# 1 "messy_mecca_kpp_acc.cu"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
# 1
#pragma GCC diagnostic push
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"
# 1
#pragma GCC diagnostic ignored "-Wunused-function"
# 1
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **); static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
# 1
#pragma GCC diagnostic pop
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"

# 1
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false

# 1
# 61 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
#pragma GCC diagnostic push
# 64
#pragma GCC diagnostic ignored "-Wunused-function"
# 66 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_types.h"
#if 0
# 66
enum cudaRoundMode { 
# 68
cudaRoundNearest, 
# 69
cudaRoundZero, 
# 70
cudaRoundPosInf, 
# 71
cudaRoundMinInf
# 72
}; 
#endif
# 98 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 98
struct char1 { 
# 100
signed char x; 
# 101
}; 
#endif
# 103 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 103
struct uchar1 { 
# 105
unsigned char x; 
# 106
}; 
#endif
# 109 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 109
struct __attribute((aligned(2))) char2 { 
# 111
signed char x, y; 
# 112
}; 
#endif
# 114 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 114
struct __attribute((aligned(2))) uchar2 { 
# 116
unsigned char x, y; 
# 117
}; 
#endif
# 119 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 119
struct char3 { 
# 121
signed char x, y, z; 
# 122
}; 
#endif
# 124 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 124
struct uchar3 { 
# 126
unsigned char x, y, z; 
# 127
}; 
#endif
# 129 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 129
struct __attribute((aligned(4))) char4 { 
# 131
signed char x, y, z, w; 
# 132
}; 
#endif
# 134 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 134
struct __attribute((aligned(4))) uchar4 { 
# 136
unsigned char x, y, z, w; 
# 137
}; 
#endif
# 139 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 139
struct short1 { 
# 141
short x; 
# 142
}; 
#endif
# 144 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 144
struct ushort1 { 
# 146
unsigned short x; 
# 147
}; 
#endif
# 149 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 149
struct __attribute((aligned(4))) short2 { 
# 151
short x, y; 
# 152
}; 
#endif
# 154 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 154
struct __attribute((aligned(4))) ushort2 { 
# 156
unsigned short x, y; 
# 157
}; 
#endif
# 159 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 159
struct short3 { 
# 161
short x, y, z; 
# 162
}; 
#endif
# 164 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 164
struct ushort3 { 
# 166
unsigned short x, y, z; 
# 167
}; 
#endif
# 169 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 169
struct __attribute((aligned(8))) short4 { short x; short y; short z; short w; }; 
#endif
# 170 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 170
struct __attribute((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
# 172 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 172
struct int1 { 
# 174
int x; 
# 175
}; 
#endif
# 177 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 177
struct uint1 { 
# 179
unsigned x; 
# 180
}; 
#endif
# 182 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 182
struct __attribute((aligned(8))) int2 { int x; int y; }; 
#endif
# 183 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 183
struct __attribute((aligned(8))) uint2 { unsigned x; unsigned y; }; 
#endif
# 185 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 185
struct int3 { 
# 187
int x, y, z; 
# 188
}; 
#endif
# 190 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 190
struct uint3 { 
# 192
unsigned x, y, z; 
# 193
}; 
#endif
# 195 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 195
struct __attribute((aligned(16))) int4 { 
# 197
int x, y, z, w; 
# 198
}; 
#endif
# 200 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 200
struct __attribute((aligned(16))) uint4 { 
# 202
unsigned x, y, z, w; 
# 203
}; 
#endif
# 205 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 205
struct long1 { 
# 207
long x; 
# 208
}; 
#endif
# 210 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 210
struct ulong1 { 
# 212
unsigned long x; 
# 213
}; 
#endif
# 220 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 220
struct __attribute((aligned((2) * sizeof(long)))) long2 { 
# 222
long x, y; 
# 223
}; 
#endif
# 225 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 225
struct __attribute((aligned((2) * sizeof(unsigned long)))) ulong2 { 
# 227
unsigned long x, y; 
# 228
}; 
#endif
# 232 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 232
struct long3 { 
# 234
long x, y, z; 
# 235
}; 
#endif
# 237 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 237
struct ulong3 { 
# 239
unsigned long x, y, z; 
# 240
}; 
#endif
# 242 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 242
struct __attribute((aligned(16))) long4 { 
# 244
long x, y, z, w; 
# 245
}; 
#endif
# 247 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 247
struct __attribute((aligned(16))) ulong4 { 
# 249
unsigned long x, y, z, w; 
# 250
}; 
#endif
# 252 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 252
struct float1 { 
# 254
float x; 
# 255
}; 
#endif
# 274 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 274
struct __attribute((aligned(8))) float2 { float x; float y; }; 
#endif
# 279 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 279
struct float3 { 
# 281
float x, y, z; 
# 282
}; 
#endif
# 284 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 284
struct __attribute((aligned(16))) float4 { 
# 286
float x, y, z, w; 
# 287
}; 
#endif
# 289 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 289
struct longlong1 { 
# 291
long long x; 
# 292
}; 
#endif
# 294 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 294
struct ulonglong1 { 
# 296
unsigned long long x; 
# 297
}; 
#endif
# 299 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 299
struct __attribute((aligned(16))) longlong2 { 
# 301
long long x, y; 
# 302
}; 
#endif
# 304 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 304
struct __attribute((aligned(16))) ulonglong2 { 
# 306
unsigned long long x, y; 
# 307
}; 
#endif
# 309 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 309
struct longlong3 { 
# 311
long long x, y, z; 
# 312
}; 
#endif
# 314 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 314
struct ulonglong3 { 
# 316
unsigned long long x, y, z; 
# 317
}; 
#endif
# 319 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 319
struct __attribute((aligned(16))) longlong4 { 
# 321
long long x, y, z, w; 
# 322
}; 
#endif
# 324 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 324
struct __attribute((aligned(16))) ulonglong4 { 
# 326
unsigned long long x, y, z, w; 
# 327
}; 
#endif
# 329 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 329
struct double1 { 
# 331
double x; 
# 332
}; 
#endif
# 334 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 334
struct __attribute((aligned(16))) double2 { 
# 336
double x, y; 
# 337
}; 
#endif
# 339 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 339
struct double3 { 
# 341
double x, y, z; 
# 342
}; 
#endif
# 344 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 344
struct __attribute((aligned(16))) double4 { 
# 346
double x, y, z, w; 
# 347
}; 
#endif
# 361 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char1 
# 361
char1; 
#endif
# 362 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar1 
# 362
uchar1; 
#endif
# 363 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char2 
# 363
char2; 
#endif
# 364 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar2 
# 364
uchar2; 
#endif
# 365 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char3 
# 365
char3; 
#endif
# 366 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar3 
# 366
uchar3; 
#endif
# 367 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char4 
# 367
char4; 
#endif
# 368 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar4 
# 368
uchar4; 
#endif
# 369 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short1 
# 369
short1; 
#endif
# 370 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort1 
# 370
ushort1; 
#endif
# 371 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short2 
# 371
short2; 
#endif
# 372 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort2 
# 372
ushort2; 
#endif
# 373 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short3 
# 373
short3; 
#endif
# 374 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort3 
# 374
ushort3; 
#endif
# 375 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short4 
# 375
short4; 
#endif
# 376 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort4 
# 376
ushort4; 
#endif
# 377 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int1 
# 377
int1; 
#endif
# 378 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint1 
# 378
uint1; 
#endif
# 379 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int2 
# 379
int2; 
#endif
# 380 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint2 
# 380
uint2; 
#endif
# 381 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int3 
# 381
int3; 
#endif
# 382 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint3 
# 382
uint3; 
#endif
# 383 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int4 
# 383
int4; 
#endif
# 384 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint4 
# 384
uint4; 
#endif
# 385 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long1 
# 385
long1; 
#endif
# 386 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong1 
# 386
ulong1; 
#endif
# 387 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long2 
# 387
long2; 
#endif
# 388 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong2 
# 388
ulong2; 
#endif
# 389 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long3 
# 389
long3; 
#endif
# 390 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong3 
# 390
ulong3; 
#endif
# 391 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long4 
# 391
long4; 
#endif
# 392 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong4 
# 392
ulong4; 
#endif
# 393 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float1 
# 393
float1; 
#endif
# 394 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float2 
# 394
float2; 
#endif
# 395 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float3 
# 395
float3; 
#endif
# 396 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float4 
# 396
float4; 
#endif
# 397 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong1 
# 397
longlong1; 
#endif
# 398 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong1 
# 398
ulonglong1; 
#endif
# 399 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong2 
# 399
longlong2; 
#endif
# 400 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong2 
# 400
ulonglong2; 
#endif
# 401 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong3 
# 401
longlong3; 
#endif
# 402 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong3 
# 402
ulonglong3; 
#endif
# 403 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong4 
# 403
longlong4; 
#endif
# 404 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong4 
# 404
ulonglong4; 
#endif
# 405 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double1 
# 405
double1; 
#endif
# 406 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double2 
# 406
double2; 
#endif
# 407 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double3 
# 407
double3; 
#endif
# 408 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double4 
# 408
double4; 
#endif
# 416 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 416
struct dim3 { 
# 418
unsigned x, y, z; 
# 428
}; 
#endif
# 430 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef dim3 
# 430
dim3; 
#endif
# 147 "/usr/lib/gcc/x86_64-redhat-linux/4.8.5/include/stddef.h" 3
typedef long ptrdiff_t; 
# 212 "/usr/lib/gcc/x86_64-redhat-linux/4.8.5/include/stddef.h" 3
typedef unsigned long size_t; 
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
# 189 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 189
enum cudaError { 
# 196
cudaSuccess, 
# 202
cudaErrorInvalidValue, 
# 208
cudaErrorMemoryAllocation, 
# 214
cudaErrorInitializationError, 
# 221
cudaErrorCudartUnloading, 
# 228
cudaErrorProfilerDisabled, 
# 236
cudaErrorProfilerNotInitialized, 
# 243
cudaErrorProfilerAlreadyStarted, 
# 250
cudaErrorProfilerAlreadyStopped, 
# 259 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorInvalidConfiguration, 
# 265
cudaErrorInvalidPitchValue = 12, 
# 271
cudaErrorInvalidSymbol, 
# 279
cudaErrorInvalidHostPointer = 16, 
# 287
cudaErrorInvalidDevicePointer, 
# 293
cudaErrorInvalidTexture, 
# 299
cudaErrorInvalidTextureBinding, 
# 306
cudaErrorInvalidChannelDescriptor, 
# 312
cudaErrorInvalidMemcpyDirection, 
# 322 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorAddressOfConstant, 
# 331 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorTextureFetchFailed, 
# 340 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorTextureNotBound, 
# 349 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorSynchronizationError, 
# 355
cudaErrorInvalidFilterSetting, 
# 361
cudaErrorInvalidNormSetting, 
# 369
cudaErrorMixedDeviceExecution, 
# 377
cudaErrorNotYetImplemented = 31, 
# 386 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorMemoryValueTooLarge, 
# 393
cudaErrorInsufficientDriver = 35, 
# 399
cudaErrorInvalidSurface = 37, 
# 405
cudaErrorDuplicateVariableName = 43, 
# 411
cudaErrorDuplicateTextureName, 
# 417
cudaErrorDuplicateSurfaceName, 
# 427 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorDevicesUnavailable, 
# 440 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorIncompatibleDriverContext = 49, 
# 446
cudaErrorMissingConfiguration = 52, 
# 455 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorPriorLaunchFailure, 
# 462
cudaErrorLaunchMaxDepthExceeded = 65, 
# 470
cudaErrorLaunchFileScopedTex, 
# 478
cudaErrorLaunchFileScopedSurf, 
# 493 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorSyncDepthExceeded, 
# 505 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorLaunchPendingCountExceeded, 
# 511
cudaErrorInvalidDeviceFunction = 98, 
# 517
cudaErrorNoDevice = 100, 
# 523
cudaErrorInvalidDevice, 
# 528
cudaErrorStartupFailure = 127, 
# 533
cudaErrorInvalidKernelImage = 200, 
# 543 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorDeviceUninitilialized, 
# 548
cudaErrorMapBufferObjectFailed = 205, 
# 553
cudaErrorUnmapBufferObjectFailed, 
# 559
cudaErrorArrayIsMapped, 
# 564
cudaErrorAlreadyMapped, 
# 572
cudaErrorNoKernelImageForDevice, 
# 577
cudaErrorAlreadyAcquired, 
# 582
cudaErrorNotMapped, 
# 588
cudaErrorNotMappedAsArray, 
# 594
cudaErrorNotMappedAsPointer, 
# 600
cudaErrorECCUncorrectable, 
# 606
cudaErrorUnsupportedLimit, 
# 612
cudaErrorDeviceAlreadyInUse, 
# 618
cudaErrorPeerAccessUnsupported, 
# 624
cudaErrorInvalidPtx, 
# 629
cudaErrorInvalidGraphicsContext, 
# 635
cudaErrorNvlinkUncorrectable, 
# 642
cudaErrorJitCompilerNotFound, 
# 647
cudaErrorInvalidSource = 300, 
# 652
cudaErrorFileNotFound, 
# 657
cudaErrorSharedObjectSymbolNotFound, 
# 662
cudaErrorSharedObjectInitFailed, 
# 667
cudaErrorOperatingSystem, 
# 674
cudaErrorInvalidResourceHandle = 400, 
# 680
cudaErrorIllegalState, 
# 686
cudaErrorSymbolNotFound = 500, 
# 694
cudaErrorNotReady = 600, 
# 702
cudaErrorIllegalAddress = 700, 
# 711 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorLaunchOutOfResources, 
# 722 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorLaunchTimeout, 
# 728
cudaErrorLaunchIncompatibleTexturing, 
# 735
cudaErrorPeerAccessAlreadyEnabled, 
# 742
cudaErrorPeerAccessNotEnabled, 
# 755 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorSetOnActiveProcess = 708, 
# 762
cudaErrorContextIsDestroyed, 
# 769
cudaErrorAssert, 
# 776
cudaErrorTooManyPeers, 
# 782
cudaErrorHostMemoryAlreadyRegistered, 
# 788
cudaErrorHostMemoryNotRegistered, 
# 797 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorHardwareStackError, 
# 805
cudaErrorIllegalInstruction, 
# 814 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorMisalignedAddress, 
# 825 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorInvalidAddressSpace, 
# 833
cudaErrorInvalidPc, 
# 844 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorLaunchFailure, 
# 853 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorCooperativeLaunchTooLarge, 
# 858
cudaErrorNotPermitted = 800, 
# 864
cudaErrorNotSupported, 
# 873 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorSystemNotReady, 
# 880
cudaErrorSystemDriverMismatch, 
# 889 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorCompatNotSupportedOnDevice, 
# 894
cudaErrorStreamCaptureUnsupported = 900, 
# 900
cudaErrorStreamCaptureInvalidated, 
# 906
cudaErrorStreamCaptureMerge, 
# 911
cudaErrorStreamCaptureUnmatched, 
# 917
cudaErrorStreamCaptureUnjoined, 
# 924
cudaErrorStreamCaptureIsolation, 
# 930
cudaErrorStreamCaptureImplicit, 
# 936
cudaErrorCapturedEvent, 
# 943
cudaErrorStreamCaptureWrongThread, 
# 948
cudaErrorUnknown = 999, 
# 956
cudaErrorApiFailureBase = 10000
# 957
}; 
#endif
# 962 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 962
enum cudaChannelFormatKind { 
# 964
cudaChannelFormatKindSigned, 
# 965
cudaChannelFormatKindUnsigned, 
# 966
cudaChannelFormatKindFloat, 
# 967
cudaChannelFormatKindNone
# 968
}; 
#endif
# 973 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 973
struct cudaChannelFormatDesc { 
# 975
int x; 
# 976
int y; 
# 977
int z; 
# 978
int w; 
# 979
cudaChannelFormatKind f; 
# 980
}; 
#endif
# 985 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
typedef struct cudaArray *cudaArray_t; 
# 990
typedef const cudaArray *cudaArray_const_t; 
# 992
struct cudaArray; 
# 997
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
# 1002
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
# 1004
struct cudaMipmappedArray; 
# 1009
#if 0
# 1009
enum cudaMemoryType { 
# 1011
cudaMemoryTypeUnregistered, 
# 1012
cudaMemoryTypeHost, 
# 1013
cudaMemoryTypeDevice, 
# 1014
cudaMemoryTypeManaged
# 1015
}; 
#endif
# 1020 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1020
enum cudaMemcpyKind { 
# 1022
cudaMemcpyHostToHost, 
# 1023
cudaMemcpyHostToDevice, 
# 1024
cudaMemcpyDeviceToHost, 
# 1025
cudaMemcpyDeviceToDevice, 
# 1026
cudaMemcpyDefault
# 1027
}; 
#endif
# 1034 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1034
struct cudaPitchedPtr { 
# 1036
void *ptr; 
# 1037
size_t pitch; 
# 1038
size_t xsize; 
# 1039
size_t ysize; 
# 1040
}; 
#endif
# 1047 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1047
struct cudaExtent { 
# 1049
size_t width; 
# 1050
size_t height; 
# 1051
size_t depth; 
# 1052
}; 
#endif
# 1059 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1059
struct cudaPos { 
# 1061
size_t x; 
# 1062
size_t y; 
# 1063
size_t z; 
# 1064
}; 
#endif
# 1069 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1069
struct cudaMemcpy3DParms { 
# 1071
cudaArray_t srcArray; 
# 1072
cudaPos srcPos; 
# 1073
cudaPitchedPtr srcPtr; 
# 1075
cudaArray_t dstArray; 
# 1076
cudaPos dstPos; 
# 1077
cudaPitchedPtr dstPtr; 
# 1079
cudaExtent extent; 
# 1080
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1081
}; 
#endif
# 1086 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1086
struct cudaMemcpy3DPeerParms { 
# 1088
cudaArray_t srcArray; 
# 1089
cudaPos srcPos; 
# 1090
cudaPitchedPtr srcPtr; 
# 1091
int srcDevice; 
# 1093
cudaArray_t dstArray; 
# 1094
cudaPos dstPos; 
# 1095
cudaPitchedPtr dstPtr; 
# 1096
int dstDevice; 
# 1098
cudaExtent extent; 
# 1099
}; 
#endif
# 1104 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1104
struct cudaMemsetParams { 
# 1105
void *dst; 
# 1106
size_t pitch; 
# 1107
unsigned value; 
# 1108
unsigned elementSize; 
# 1109
size_t width; 
# 1110
size_t height; 
# 1111
}; 
#endif
# 1123 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
typedef void (*cudaHostFn_t)(void * userData); 
# 1128
#if 0
# 1128
struct cudaHostNodeParams { 
# 1129
cudaHostFn_t fn; 
# 1130
void *userData; 
# 1131
}; 
#endif
# 1136 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1136
enum cudaStreamCaptureStatus { 
# 1137
cudaStreamCaptureStatusNone, 
# 1138
cudaStreamCaptureStatusActive, 
# 1139
cudaStreamCaptureStatusInvalidated
# 1141
}; 
#endif
# 1147 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1147
enum cudaStreamCaptureMode { 
# 1148
cudaStreamCaptureModeGlobal, 
# 1149
cudaStreamCaptureModeThreadLocal, 
# 1150
cudaStreamCaptureModeRelaxed
# 1151
}; 
#endif
# 1156 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
struct cudaGraphicsResource; 
# 1161
#if 0
# 1161
enum cudaGraphicsRegisterFlags { 
# 1163
cudaGraphicsRegisterFlagsNone, 
# 1164
cudaGraphicsRegisterFlagsReadOnly, 
# 1165
cudaGraphicsRegisterFlagsWriteDiscard, 
# 1166
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
# 1167
cudaGraphicsRegisterFlagsTextureGather = 8
# 1168
}; 
#endif
# 1173 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1173
enum cudaGraphicsMapFlags { 
# 1175
cudaGraphicsMapFlagsNone, 
# 1176
cudaGraphicsMapFlagsReadOnly, 
# 1177
cudaGraphicsMapFlagsWriteDiscard
# 1178
}; 
#endif
# 1183 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1183
enum cudaGraphicsCubeFace { 
# 1185
cudaGraphicsCubeFacePositiveX, 
# 1186
cudaGraphicsCubeFaceNegativeX, 
# 1187
cudaGraphicsCubeFacePositiveY, 
# 1188
cudaGraphicsCubeFaceNegativeY, 
# 1189
cudaGraphicsCubeFacePositiveZ, 
# 1190
cudaGraphicsCubeFaceNegativeZ
# 1191
}; 
#endif
# 1196 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1196
enum cudaResourceType { 
# 1198
cudaResourceTypeArray, 
# 1199
cudaResourceTypeMipmappedArray, 
# 1200
cudaResourceTypeLinear, 
# 1201
cudaResourceTypePitch2D
# 1202
}; 
#endif
# 1207 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1207
enum cudaResourceViewFormat { 
# 1209
cudaResViewFormatNone, 
# 1210
cudaResViewFormatUnsignedChar1, 
# 1211
cudaResViewFormatUnsignedChar2, 
# 1212
cudaResViewFormatUnsignedChar4, 
# 1213
cudaResViewFormatSignedChar1, 
# 1214
cudaResViewFormatSignedChar2, 
# 1215
cudaResViewFormatSignedChar4, 
# 1216
cudaResViewFormatUnsignedShort1, 
# 1217
cudaResViewFormatUnsignedShort2, 
# 1218
cudaResViewFormatUnsignedShort4, 
# 1219
cudaResViewFormatSignedShort1, 
# 1220
cudaResViewFormatSignedShort2, 
# 1221
cudaResViewFormatSignedShort4, 
# 1222
cudaResViewFormatUnsignedInt1, 
# 1223
cudaResViewFormatUnsignedInt2, 
# 1224
cudaResViewFormatUnsignedInt4, 
# 1225
cudaResViewFormatSignedInt1, 
# 1226
cudaResViewFormatSignedInt2, 
# 1227
cudaResViewFormatSignedInt4, 
# 1228
cudaResViewFormatHalf1, 
# 1229
cudaResViewFormatHalf2, 
# 1230
cudaResViewFormatHalf4, 
# 1231
cudaResViewFormatFloat1, 
# 1232
cudaResViewFormatFloat2, 
# 1233
cudaResViewFormatFloat4, 
# 1234
cudaResViewFormatUnsignedBlockCompressed1, 
# 1235
cudaResViewFormatUnsignedBlockCompressed2, 
# 1236
cudaResViewFormatUnsignedBlockCompressed3, 
# 1237
cudaResViewFormatUnsignedBlockCompressed4, 
# 1238
cudaResViewFormatSignedBlockCompressed4, 
# 1239
cudaResViewFormatUnsignedBlockCompressed5, 
# 1240
cudaResViewFormatSignedBlockCompressed5, 
# 1241
cudaResViewFormatUnsignedBlockCompressed6H, 
# 1242
cudaResViewFormatSignedBlockCompressed6H, 
# 1243
cudaResViewFormatUnsignedBlockCompressed7
# 1244
}; 
#endif
# 1249 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1249
struct cudaResourceDesc { 
# 1250
cudaResourceType resType; 
# 1252
union { 
# 1253
struct { 
# 1254
cudaArray_t array; 
# 1255
} array; 
# 1256
struct { 
# 1257
cudaMipmappedArray_t mipmap; 
# 1258
} mipmap; 
# 1259
struct { 
# 1260
void *devPtr; 
# 1261
cudaChannelFormatDesc desc; 
# 1262
size_t sizeInBytes; 
# 1263
} linear; 
# 1264
struct { 
# 1265
void *devPtr; 
# 1266
cudaChannelFormatDesc desc; 
# 1267
size_t width; 
# 1268
size_t height; 
# 1269
size_t pitchInBytes; 
# 1270
} pitch2D; 
# 1271
} res; 
# 1272
}; 
#endif
# 1277 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1277
struct cudaResourceViewDesc { 
# 1279
cudaResourceViewFormat format; 
# 1280
size_t width; 
# 1281
size_t height; 
# 1282
size_t depth; 
# 1283
unsigned firstMipmapLevel; 
# 1284
unsigned lastMipmapLevel; 
# 1285
unsigned firstLayer; 
# 1286
unsigned lastLayer; 
# 1287
}; 
#endif
# 1292 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1292
struct cudaPointerAttributes { 
# 1302 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
__attribute((deprecated)) cudaMemoryType memoryType; 
# 1308
cudaMemoryType type; 
# 1319 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
int device; 
# 1325
void *devicePointer; 
# 1334 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
void *hostPointer; 
# 1341
__attribute((deprecated)) int isManaged; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1342
}; 
#endif
# 1347 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1347
struct cudaFuncAttributes { 
# 1354
size_t sharedSizeBytes; 
# 1360
size_t constSizeBytes; 
# 1365
size_t localSizeBytes; 
# 1372
int maxThreadsPerBlock; 
# 1377
int numRegs; 
# 1384
int ptxVersion; 
# 1391
int binaryVersion; 
# 1397
int cacheModeCA; 
# 1404
int maxDynamicSharedSizeBytes; 
# 1413 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
int preferredShmemCarveout; 
# 1414
}; 
#endif
# 1419 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1419
enum cudaFuncAttribute { 
# 1421
cudaFuncAttributeMaxDynamicSharedMemorySize = 8, 
# 1422
cudaFuncAttributePreferredSharedMemoryCarveout, 
# 1423
cudaFuncAttributeMax
# 1424
}; 
#endif
# 1429 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1429
enum cudaFuncCache { 
# 1431
cudaFuncCachePreferNone, 
# 1432
cudaFuncCachePreferShared, 
# 1433
cudaFuncCachePreferL1, 
# 1434
cudaFuncCachePreferEqual
# 1435
}; 
#endif
# 1441 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1441
enum cudaSharedMemConfig { 
# 1443
cudaSharedMemBankSizeDefault, 
# 1444
cudaSharedMemBankSizeFourByte, 
# 1445
cudaSharedMemBankSizeEightByte
# 1446
}; 
#endif
# 1451 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1451
enum cudaSharedCarveout { 
# 1452
cudaSharedmemCarveoutDefault = (-1), 
# 1453
cudaSharedmemCarveoutMaxShared = 100, 
# 1454
cudaSharedmemCarveoutMaxL1 = 0
# 1455
}; 
#endif
# 1460 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1460
enum cudaComputeMode { 
# 1462
cudaComputeModeDefault, 
# 1463
cudaComputeModeExclusive, 
# 1464
cudaComputeModeProhibited, 
# 1465
cudaComputeModeExclusiveProcess
# 1466
}; 
#endif
# 1471 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1471
enum cudaLimit { 
# 1473
cudaLimitStackSize, 
# 1474
cudaLimitPrintfFifoSize, 
# 1475
cudaLimitMallocHeapSize, 
# 1476
cudaLimitDevRuntimeSyncDepth, 
# 1477
cudaLimitDevRuntimePendingLaunchCount, 
# 1478
cudaLimitMaxL2FetchGranularity
# 1479
}; 
#endif
# 1484 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1484
enum cudaMemoryAdvise { 
# 1486
cudaMemAdviseSetReadMostly = 1, 
# 1487
cudaMemAdviseUnsetReadMostly, 
# 1488
cudaMemAdviseSetPreferredLocation, 
# 1489
cudaMemAdviseUnsetPreferredLocation, 
# 1490
cudaMemAdviseSetAccessedBy, 
# 1491
cudaMemAdviseUnsetAccessedBy
# 1492
}; 
#endif
# 1497 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1497
enum cudaMemRangeAttribute { 
# 1499
cudaMemRangeAttributeReadMostly = 1, 
# 1500
cudaMemRangeAttributePreferredLocation, 
# 1501
cudaMemRangeAttributeAccessedBy, 
# 1502
cudaMemRangeAttributeLastPrefetchLocation
# 1503
}; 
#endif
# 1508 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1508
enum cudaOutputMode { 
# 1510
cudaKeyValuePair, 
# 1511
cudaCSV
# 1512
}; 
#endif
# 1517 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1517
enum cudaDeviceAttr { 
# 1519
cudaDevAttrMaxThreadsPerBlock = 1, 
# 1520
cudaDevAttrMaxBlockDimX, 
# 1521
cudaDevAttrMaxBlockDimY, 
# 1522
cudaDevAttrMaxBlockDimZ, 
# 1523
cudaDevAttrMaxGridDimX, 
# 1524
cudaDevAttrMaxGridDimY, 
# 1525
cudaDevAttrMaxGridDimZ, 
# 1526
cudaDevAttrMaxSharedMemoryPerBlock, 
# 1527
cudaDevAttrTotalConstantMemory, 
# 1528
cudaDevAttrWarpSize, 
# 1529
cudaDevAttrMaxPitch, 
# 1530
cudaDevAttrMaxRegistersPerBlock, 
# 1531
cudaDevAttrClockRate, 
# 1532
cudaDevAttrTextureAlignment, 
# 1533
cudaDevAttrGpuOverlap, 
# 1534
cudaDevAttrMultiProcessorCount, 
# 1535
cudaDevAttrKernelExecTimeout, 
# 1536
cudaDevAttrIntegrated, 
# 1537
cudaDevAttrCanMapHostMemory, 
# 1538
cudaDevAttrComputeMode, 
# 1539
cudaDevAttrMaxTexture1DWidth, 
# 1540
cudaDevAttrMaxTexture2DWidth, 
# 1541
cudaDevAttrMaxTexture2DHeight, 
# 1542
cudaDevAttrMaxTexture3DWidth, 
# 1543
cudaDevAttrMaxTexture3DHeight, 
# 1544
cudaDevAttrMaxTexture3DDepth, 
# 1545
cudaDevAttrMaxTexture2DLayeredWidth, 
# 1546
cudaDevAttrMaxTexture2DLayeredHeight, 
# 1547
cudaDevAttrMaxTexture2DLayeredLayers, 
# 1548
cudaDevAttrSurfaceAlignment, 
# 1549
cudaDevAttrConcurrentKernels, 
# 1550
cudaDevAttrEccEnabled, 
# 1551
cudaDevAttrPciBusId, 
# 1552
cudaDevAttrPciDeviceId, 
# 1553
cudaDevAttrTccDriver, 
# 1554
cudaDevAttrMemoryClockRate, 
# 1555
cudaDevAttrGlobalMemoryBusWidth, 
# 1556
cudaDevAttrL2CacheSize, 
# 1557
cudaDevAttrMaxThreadsPerMultiProcessor, 
# 1558
cudaDevAttrAsyncEngineCount, 
# 1559
cudaDevAttrUnifiedAddressing, 
# 1560
cudaDevAttrMaxTexture1DLayeredWidth, 
# 1561
cudaDevAttrMaxTexture1DLayeredLayers, 
# 1562
cudaDevAttrMaxTexture2DGatherWidth = 45, 
# 1563
cudaDevAttrMaxTexture2DGatherHeight, 
# 1564
cudaDevAttrMaxTexture3DWidthAlt, 
# 1565
cudaDevAttrMaxTexture3DHeightAlt, 
# 1566
cudaDevAttrMaxTexture3DDepthAlt, 
# 1567
cudaDevAttrPciDomainId, 
# 1568
cudaDevAttrTexturePitchAlignment, 
# 1569
cudaDevAttrMaxTextureCubemapWidth, 
# 1570
cudaDevAttrMaxTextureCubemapLayeredWidth, 
# 1571
cudaDevAttrMaxTextureCubemapLayeredLayers, 
# 1572
cudaDevAttrMaxSurface1DWidth, 
# 1573
cudaDevAttrMaxSurface2DWidth, 
# 1574
cudaDevAttrMaxSurface2DHeight, 
# 1575
cudaDevAttrMaxSurface3DWidth, 
# 1576
cudaDevAttrMaxSurface3DHeight, 
# 1577
cudaDevAttrMaxSurface3DDepth, 
# 1578
cudaDevAttrMaxSurface1DLayeredWidth, 
# 1579
cudaDevAttrMaxSurface1DLayeredLayers, 
# 1580
cudaDevAttrMaxSurface2DLayeredWidth, 
# 1581
cudaDevAttrMaxSurface2DLayeredHeight, 
# 1582
cudaDevAttrMaxSurface2DLayeredLayers, 
# 1583
cudaDevAttrMaxSurfaceCubemapWidth, 
# 1584
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
# 1585
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
# 1586
cudaDevAttrMaxTexture1DLinearWidth, 
# 1587
cudaDevAttrMaxTexture2DLinearWidth, 
# 1588
cudaDevAttrMaxTexture2DLinearHeight, 
# 1589
cudaDevAttrMaxTexture2DLinearPitch, 
# 1590
cudaDevAttrMaxTexture2DMipmappedWidth, 
# 1591
cudaDevAttrMaxTexture2DMipmappedHeight, 
# 1592
cudaDevAttrComputeCapabilityMajor, 
# 1593
cudaDevAttrComputeCapabilityMinor, 
# 1594
cudaDevAttrMaxTexture1DMipmappedWidth, 
# 1595
cudaDevAttrStreamPrioritiesSupported, 
# 1596
cudaDevAttrGlobalL1CacheSupported, 
# 1597
cudaDevAttrLocalL1CacheSupported, 
# 1598
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
# 1599
cudaDevAttrMaxRegistersPerMultiprocessor, 
# 1600
cudaDevAttrManagedMemory, 
# 1601
cudaDevAttrIsMultiGpuBoard, 
# 1602
cudaDevAttrMultiGpuBoardGroupID, 
# 1603
cudaDevAttrHostNativeAtomicSupported, 
# 1604
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
# 1605
cudaDevAttrPageableMemoryAccess, 
# 1606
cudaDevAttrConcurrentManagedAccess, 
# 1607
cudaDevAttrComputePreemptionSupported, 
# 1608
cudaDevAttrCanUseHostPointerForRegisteredMem, 
# 1609
cudaDevAttrReserved92, 
# 1610
cudaDevAttrReserved93, 
# 1611
cudaDevAttrReserved94, 
# 1612
cudaDevAttrCooperativeLaunch, 
# 1613
cudaDevAttrCooperativeMultiDeviceLaunch, 
# 1614
cudaDevAttrMaxSharedMemoryPerBlockOptin, 
# 1615
cudaDevAttrCanFlushRemoteWrites, 
# 1616
cudaDevAttrHostRegisterSupported, 
# 1617
cudaDevAttrPageableMemoryAccessUsesHostPageTables, 
# 1618
cudaDevAttrDirectManagedMemAccessFromHost
# 1619
}; 
#endif
# 1625 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1625
enum cudaDeviceP2PAttr { 
# 1626
cudaDevP2PAttrPerformanceRank = 1, 
# 1627
cudaDevP2PAttrAccessSupported, 
# 1628
cudaDevP2PAttrNativeAtomicSupported, 
# 1629
cudaDevP2PAttrCudaArrayAccessSupported
# 1630
}; 
#endif
# 1637 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1637
struct CUuuid_st { 
# 1638
char bytes[16]; 
# 1639
}; 
#endif
# 1640 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef CUuuid_st 
# 1640
CUuuid; 
#endif
# 1642 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef CUuuid_st 
# 1642
cudaUUID_t; 
#endif
# 1647 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1647
struct cudaDeviceProp { 
# 1649
char name[256]; 
# 1650
cudaUUID_t uuid; 
# 1651
char luid[8]; 
# 1652
unsigned luidDeviceNodeMask; 
# 1653
size_t totalGlobalMem; 
# 1654
size_t sharedMemPerBlock; 
# 1655
int regsPerBlock; 
# 1656
int warpSize; 
# 1657
size_t memPitch; 
# 1658
int maxThreadsPerBlock; 
# 1659
int maxThreadsDim[3]; 
# 1660
int maxGridSize[3]; 
# 1661
int clockRate; 
# 1662
size_t totalConstMem; 
# 1663
int major; 
# 1664
int minor; 
# 1665
size_t textureAlignment; 
# 1666
size_t texturePitchAlignment; 
# 1667
int deviceOverlap; 
# 1668
int multiProcessorCount; 
# 1669
int kernelExecTimeoutEnabled; 
# 1670
int integrated; 
# 1671
int canMapHostMemory; 
# 1672
int computeMode; 
# 1673
int maxTexture1D; 
# 1674
int maxTexture1DMipmap; 
# 1675
int maxTexture1DLinear; 
# 1676
int maxTexture2D[2]; 
# 1677
int maxTexture2DMipmap[2]; 
# 1678
int maxTexture2DLinear[3]; 
# 1679
int maxTexture2DGather[2]; 
# 1680
int maxTexture3D[3]; 
# 1681
int maxTexture3DAlt[3]; 
# 1682
int maxTextureCubemap; 
# 1683
int maxTexture1DLayered[2]; 
# 1684
int maxTexture2DLayered[3]; 
# 1685
int maxTextureCubemapLayered[2]; 
# 1686
int maxSurface1D; 
# 1687
int maxSurface2D[2]; 
# 1688
int maxSurface3D[3]; 
# 1689
int maxSurface1DLayered[2]; 
# 1690
int maxSurface2DLayered[3]; 
# 1691
int maxSurfaceCubemap; 
# 1692
int maxSurfaceCubemapLayered[2]; 
# 1693
size_t surfaceAlignment; 
# 1694
int concurrentKernels; 
# 1695
int ECCEnabled; 
# 1696
int pciBusID; 
# 1697
int pciDeviceID; 
# 1698
int pciDomainID; 
# 1699
int tccDriver; 
# 1700
int asyncEngineCount; 
# 1701
int unifiedAddressing; 
# 1702
int memoryClockRate; 
# 1703
int memoryBusWidth; 
# 1704
int l2CacheSize; 
# 1705
int maxThreadsPerMultiProcessor; 
# 1706
int streamPrioritiesSupported; 
# 1707
int globalL1CacheSupported; 
# 1708
int localL1CacheSupported; 
# 1709
size_t sharedMemPerMultiprocessor; 
# 1710
int regsPerMultiprocessor; 
# 1711
int managedMemory; 
# 1712
int isMultiGpuBoard; 
# 1713
int multiGpuBoardGroupID; 
# 1714
int hostNativeAtomicSupported; 
# 1715
int singleToDoublePrecisionPerfRatio; 
# 1716
int pageableMemoryAccess; 
# 1717
int concurrentManagedAccess; 
# 1718
int computePreemptionSupported; 
# 1719
int canUseHostPointerForRegisteredMem; 
# 1720
int cooperativeLaunch; 
# 1721
int cooperativeMultiDeviceLaunch; 
# 1722
size_t sharedMemPerBlockOptin; 
# 1723
int pageableMemoryAccessUsesHostPageTables; 
# 1724
int directManagedMemAccessFromHost; 
# 1725
}; 
#endif
# 1818 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 1815
struct cudaIpcEventHandle_st { 
# 1817
char reserved[64]; 
# 1818
} cudaIpcEventHandle_t; 
#endif
# 1826 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 1823
struct cudaIpcMemHandle_st { 
# 1825
char reserved[64]; 
# 1826
} cudaIpcMemHandle_t; 
#endif
# 1831 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1831
enum cudaExternalMemoryHandleType { 
# 1835
cudaExternalMemoryHandleTypeOpaqueFd = 1, 
# 1839
cudaExternalMemoryHandleTypeOpaqueWin32, 
# 1843
cudaExternalMemoryHandleTypeOpaqueWin32Kmt, 
# 1847
cudaExternalMemoryHandleTypeD3D12Heap, 
# 1851
cudaExternalMemoryHandleTypeD3D12Resource
# 1852
}; 
#endif
# 1862 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1862
struct cudaExternalMemoryHandleDesc { 
# 1866
cudaExternalMemoryHandleType type; 
# 1867
union { 
# 1873
int fd; 
# 1885 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
struct { 
# 1889
void *handle; 
# 1894
const void *name; 
# 1895
} win32; 
# 1896
} handle; 
# 1900
unsigned long long size; 
# 1904
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1905
}; 
#endif
# 1910 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1910
struct cudaExternalMemoryBufferDesc { 
# 1914
unsigned long long offset; 
# 1918
unsigned long long size; 
# 1922
unsigned flags; 
# 1923
}; 
#endif
# 1928 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1928
struct cudaExternalMemoryMipmappedArrayDesc { 
# 1933
unsigned long long offset; 
# 1937
cudaChannelFormatDesc formatDesc; 
# 1941
cudaExtent extent; 
# 1946
unsigned flags; 
# 1950
unsigned numLevels; 
# 1951
}; 
#endif
# 1956 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1956
enum cudaExternalSemaphoreHandleType { 
# 1960
cudaExternalSemaphoreHandleTypeOpaqueFd = 1, 
# 1964
cudaExternalSemaphoreHandleTypeOpaqueWin32, 
# 1968
cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt, 
# 1972
cudaExternalSemaphoreHandleTypeD3D12Fence
# 1973
}; 
#endif
# 1978 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1978
struct cudaExternalSemaphoreHandleDesc { 
# 1982
cudaExternalSemaphoreHandleType type; 
# 1983
union { 
# 1988
int fd; 
# 1999 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
struct { 
# 2003
void *handle; 
# 2008
const void *name; 
# 2009
} win32; 
# 2010
} handle; 
# 2014
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 2015
}; 
#endif
# 2020 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2020
struct cudaExternalSemaphoreSignalParams { 
# 2021
union { 
# 2025
struct { 
# 2029
unsigned long long value; 
# 2030
} fence; 
# 2031
} params; 
# 2035
unsigned flags; 
# 2036
}; 
#endif
# 2041 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2041
struct cudaExternalSemaphoreWaitParams { 
# 2042
union { 
# 2046
struct { 
# 2050
unsigned long long value; 
# 2051
} fence; 
# 2052
} params; 
# 2056
unsigned flags; 
# 2057
}; 
#endif
# 2069 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef cudaError 
# 2069
cudaError_t; 
#endif
# 2074 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUstream_st *
# 2074
cudaStream_t; 
#endif
# 2079 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUevent_st *
# 2079
cudaEvent_t; 
#endif
# 2084 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef cudaGraphicsResource *
# 2084
cudaGraphicsResource_t; 
#endif
# 2089 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef cudaOutputMode 
# 2089
cudaOutputMode_t; 
#endif
# 2094 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUexternalMemory_st *
# 2094
cudaExternalMemory_t; 
#endif
# 2099 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUexternalSemaphore_st *
# 2099
cudaExternalSemaphore_t; 
#endif
# 2104 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUgraph_st *
# 2104
cudaGraph_t; 
#endif
# 2109 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUgraphNode_st *
# 2109
cudaGraphNode_t; 
#endif
# 2114 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2114
enum cudaCGScope { 
# 2115
cudaCGScopeInvalid, 
# 2116
cudaCGScopeGrid, 
# 2117
cudaCGScopeMultiGrid
# 2118
}; 
#endif
# 2123 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2123
struct cudaLaunchParams { 
# 2125
void *func; 
# 2126
dim3 gridDim; 
# 2127
dim3 blockDim; 
# 2128
void **args; 
# 2129
size_t sharedMem; 
# 2130
cudaStream_t stream; 
# 2131
}; 
#endif
# 2136 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2136
struct cudaKernelNodeParams { 
# 2137
void *func; 
# 2138
dim3 gridDim; 
# 2139
dim3 blockDim; 
# 2140
unsigned sharedMemBytes; 
# 2141
void **kernelParams; 
# 2142
void **extra; 
# 2143
}; 
#endif
# 2148 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 2148
enum cudaGraphNodeType { 
# 2149
cudaGraphNodeTypeKernel, 
# 2150
cudaGraphNodeTypeMemcpy, 
# 2151
cudaGraphNodeTypeMemset, 
# 2152
cudaGraphNodeTypeHost, 
# 2153
cudaGraphNodeTypeGraph, 
# 2154
cudaGraphNodeTypeEmpty, 
# 2155
cudaGraphNodeTypeCount
# 2156
}; 
#endif
# 2161 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_types.h"
typedef struct CUgraphExec_st *cudaGraphExec_t; 
# 84 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
# 84
enum cudaSurfaceBoundaryMode { 
# 86
cudaBoundaryModeZero, 
# 87
cudaBoundaryModeClamp, 
# 88
cudaBoundaryModeTrap
# 89
}; 
#endif
# 94 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
# 94
enum cudaSurfaceFormatMode { 
# 96
cudaFormatModeForced, 
# 97
cudaFormatModeAuto
# 98
}; 
#endif
# 103 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
# 103
struct surfaceReference { 
# 108
cudaChannelFormatDesc channelDesc; 
# 109
}; 
#endif
# 114 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
typedef unsigned long long 
# 114
cudaSurfaceObject_t; 
#endif
# 84 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 84
enum cudaTextureAddressMode { 
# 86
cudaAddressModeWrap, 
# 87
cudaAddressModeClamp, 
# 88
cudaAddressModeMirror, 
# 89
cudaAddressModeBorder
# 90
}; 
#endif
# 95 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 95
enum cudaTextureFilterMode { 
# 97
cudaFilterModePoint, 
# 98
cudaFilterModeLinear
# 99
}; 
#endif
# 104 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 104
enum cudaTextureReadMode { 
# 106
cudaReadModeElementType, 
# 107
cudaReadModeNormalizedFloat
# 108
}; 
#endif
# 113 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 113
struct textureReference { 
# 118
int normalized; 
# 122
cudaTextureFilterMode filterMode; 
# 126
cudaTextureAddressMode addressMode[3]; 
# 130
cudaChannelFormatDesc channelDesc; 
# 134
int sRGB; 
# 138
unsigned maxAnisotropy; 
# 142
cudaTextureFilterMode mipmapFilterMode; 
# 146
float mipmapLevelBias; 
# 150
float minMipmapLevelClamp; 
# 154
float maxMipmapLevelClamp; 
# 155
int __cudaReserved[15]; 
# 156
}; 
#endif
# 161 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 161
struct cudaTextureDesc { 
# 166
cudaTextureAddressMode addressMode[3]; 
# 170
cudaTextureFilterMode filterMode; 
# 174
cudaTextureReadMode readMode; 
# 178
int sRGB; 
# 182
float borderColor[4]; 
# 186
int normalizedCoords; 
# 190
unsigned maxAnisotropy; 
# 194
cudaTextureFilterMode mipmapFilterMode; 
# 198
float mipmapLevelBias; 
# 202
float minMipmapLevelClamp; 
# 206
float maxMipmapLevelClamp; 
# 207
}; 
#endif
# 212 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
typedef unsigned long long 
# 212
cudaTextureObject_t; 
#endif
# 70 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/library_types.h"
typedef 
# 54
enum cudaDataType_t { 
# 56
CUDA_R_16F = 2, 
# 57
CUDA_C_16F = 6, 
# 58
CUDA_R_32F = 0, 
# 59
CUDA_C_32F = 4, 
# 60
CUDA_R_64F = 1, 
# 61
CUDA_C_64F = 5, 
# 62
CUDA_R_8I = 3, 
# 63
CUDA_C_8I = 7, 
# 64
CUDA_R_8U, 
# 65
CUDA_C_8U, 
# 66
CUDA_R_32I, 
# 67
CUDA_C_32I, 
# 68
CUDA_R_32U, 
# 69
CUDA_C_32U
# 70
} cudaDataType; 
# 78
typedef 
# 73
enum libraryPropertyType_t { 
# 75
MAJOR_VERSION, 
# 76
MINOR_VERSION, 
# 77
PATCH_LEVEL
# 78
} libraryPropertyType; 
# 121 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
extern "C" {
# 123
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 124
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 125
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 126
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 127
extern cudaError_t cudaDeviceSynchronize(); 
# 128
extern cudaError_t cudaGetLastError(); 
# 129
extern cudaError_t cudaPeekAtLastError(); 
# 130
extern const char *cudaGetErrorString(cudaError_t error); 
# 131
extern const char *cudaGetErrorName(cudaError_t error); 
# 132
extern cudaError_t cudaGetDeviceCount(int * count); 
# 133
extern cudaError_t cudaGetDevice(int * device); 
# 134
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 135
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 136
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 137
__attribute__((unused)) extern cudaError_t cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 138
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 139
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream); 
# 140
__attribute__((unused)) extern cudaError_t cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
# 141
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 142
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 143
extern cudaError_t cudaFree(void * devPtr); 
# 144
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 145
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 146
__attribute__((unused)) extern cudaError_t cudaMemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 147
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 148
__attribute__((unused)) extern cudaError_t cudaMemcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 149
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 150
__attribute__((unused)) extern cudaError_t cudaMemcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 151
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 152
__attribute__((unused)) extern cudaError_t cudaMemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 153
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 154
__attribute__((unused)) extern cudaError_t cudaMemset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 155
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 156
__attribute__((unused)) extern cudaError_t cudaMemset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 157
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 178 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern void *cudaGetParameterBuffer(size_t alignment, size_t size); 
# 206 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern void *cudaGetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
# 207
__attribute__((unused)) extern cudaError_t cudaLaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 208
__attribute__((unused)) extern cudaError_t cudaLaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
# 226 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern cudaError_t cudaLaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 227
__attribute__((unused)) extern cudaError_t cudaLaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
# 230
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
# 231
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 233
__attribute__((unused)) extern unsigned long long cudaCGGetIntrinsicHandle(cudaCGScope scope); 
# 234
__attribute__((unused)) extern cudaError_t cudaCGSynchronize(unsigned long long handle, unsigned flags); 
# 235
__attribute__((unused)) extern cudaError_t cudaCGSynchronizeGrid(unsigned long long handle, unsigned flags); 
# 236
__attribute__((unused)) extern cudaError_t cudaCGGetSize(unsigned * numThreads, unsigned * numGrids, unsigned long long handle); 
# 237
__attribute__((unused)) extern cudaError_t cudaCGGetRank(unsigned * threadRank, unsigned * gridRank, unsigned long long handle); 
# 238
}
# 240
template< class T> static inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
# 241
template< class T> static inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
# 242
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
# 243
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 245 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern "C" {
# 280 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceReset(); 
# 301 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSynchronize(); 
# 386 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value); 
# 420 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 453 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 490 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
# 534 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
# 565 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 609 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
# 636 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
# 666 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
# 713 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
# 753 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
# 796 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
# 854 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
# 889 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcCloseMemHandle(void * devPtr); 
# 931 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadExit(); 
# 957 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadSynchronize(); 
# 1006 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value); 
# 1039 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadGetLimit(size_t * pValue, cudaLimit limit); 
# 1075 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 1122 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig); 
# 1181 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetLastError(); 
# 1227 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaPeekAtLastError(); 
# 1243 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern const char *cudaGetErrorName(cudaError_t error); 
# 1259 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern const char *cudaGetErrorString(cudaError_t error); 
# 1288 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceCount(int * count); 
# 1559 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceProperties(cudaDeviceProp * prop, int device); 
# 1748 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 1788 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
# 1809 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
# 1846 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaSetDevice(int device); 
# 1867 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDevice(int * device); 
# 1898 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaSetValidDevices(int * device_arr, int len); 
# 1967 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaSetDeviceFlags(unsigned flags); 
# 2013 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceFlags(unsigned * flags); 
# 2053 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreate(cudaStream_t * pStream); 
# 2085 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 2131 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
# 2158 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
# 2183 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
# 2214 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 2240 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 2248
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
# 2315 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
# 2339 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamSynchronize(cudaStream_t stream); 
# 2364 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamQuery(cudaStream_t stream); 
# 2447 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
# 2483 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode); 
# 2534 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode * mode); 
# 2562 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t * pGraph); 
# 2600 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus); 
# 2628 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus, unsigned long long * pId); 
# 2666 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventCreate(cudaEvent_t * event); 
# 2703 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 2742 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
# 2773 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventQuery(cudaEvent_t event); 
# 2803 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventSynchronize(cudaEvent_t event); 
# 2830 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 2873 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
# 3012 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaImportExternalMemory(cudaExternalMemory_t * extMem_out, const cudaExternalMemoryHandleDesc * memHandleDesc); 
# 3066 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaExternalMemoryGetMappedBuffer(void ** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc * bufferDesc); 
# 3121 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t * mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc * mipmapDesc); 
# 3144 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem); 
# 3238 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t * extSem_out, const cudaExternalSemaphoreHandleDesc * semHandleDesc); 
# 3277 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreSignalParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
# 3320 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreWaitParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
# 3342 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem); 
# 3407 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 3464 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchCooperativeKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 3563 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams * launchParamsList, unsigned numDevices, unsigned flags = 0); 
# 3612 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
# 3667 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
# 3702 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 3741 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetAttribute(const void * func, cudaFuncAttribute attr, int value); 
# 3765 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaSetDoubleForDevice(double * d); 
# 3789 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaSetDoubleForHost(double * d); 
# 3855 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void * userData); 
# 3910 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
# 3954 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
# 4074 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
# 4105 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 4138 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocHost(void ** ptr, size_t size); 
# 4181 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
# 4227 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
# 4256 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFree(void * devPtr); 
# 4279 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFreeHost(void * ptr); 
# 4302 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFreeArray(cudaArray_t array); 
# 4325 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
# 4391 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
# 4475 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostRegister(void * ptr, size_t size, unsigned flags); 
# 4498 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostUnregister(void * ptr); 
# 4543 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
# 4565 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostGetFlags(unsigned * pFlags, void * pHost); 
# 4604 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
# 4743 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
# 4882 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
# 4911 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
# 5016 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms * p); 
# 5047 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
# 5165 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
# 5191 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
# 5213 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemGetInfo(size_t * free, size_t * total); 
# 5239 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
# 5282 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 5317 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
# 5365 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 5414 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 5463 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
# 5510 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 5553 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
# 5596 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
# 5652 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5687 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
# 5749 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5806 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5862 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5913 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5964 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5993 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset(void * devPtr, int value, size_t count); 
# 6027 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
# 6071 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
# 6107 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
# 6148 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
# 6199 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
# 6227 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
# 6254 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol); 
# 6324 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream = 0); 
# 6440 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, int device); 
# 6499 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
# 6538 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
# 6598 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
# 6640 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
# 6683 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 6734 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6784 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6950 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
# 6991 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
# 7033 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
# 7055 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceDisablePeerAccess(int peerDevice); 
# 7118 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
# 7153 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
# 7192 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 7227 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 7259 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
# 7297 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
# 7326 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
# 7397 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaBindTexture(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t size = ((2147483647) * 2U) + 1U); 
# 7456 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaBindTexture2D(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch); 
# 7494 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaBindTextureToArray(const textureReference * texref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
# 7534 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaBindTextureToMipmappedArray(const textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const cudaChannelFormatDesc * desc); 
# 7560 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaUnbindTexture(const textureReference * texref); 
# 7589 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureAlignmentOffset(size_t * offset, const textureReference * texref); 
# 7619 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureReference(const textureReference ** texref, const void * symbol); 
# 7664 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaBindSurfaceToArray(const surfaceReference * surfref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
# 7689 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetSurfaceReference(const surfaceReference ** surfref, const void * symbol); 
# 7724 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
# 7754 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
# 7969 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
# 7988 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject); 
# 8008 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
# 8028 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
# 8049 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
# 8094 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
# 8113 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
# 8132 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
# 8166 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDriverGetVersion(int * driverVersion); 
# 8191 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 8238 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphCreate(cudaGraph_t * pGraph, unsigned flags); 
# 8335 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaKernelNodeParams * pNodeParams); 
# 8368 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams * pNodeParams); 
# 8393 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
# 8437 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemcpy3DParms * pCopyParams); 
# 8460 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms * pNodeParams); 
# 8483 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
# 8525 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemsetParams * pMemsetParams); 
# 8548 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams * pNodeParams); 
# 8571 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
# 8612 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaHostNodeParams * pNodeParams); 
# 8635 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams * pNodeParams); 
# 8658 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
# 8696 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraph_t childGraph); 
# 8720 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t * pGraph); 
# 8757 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies); 
# 8784 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t originalGraph); 
# 8812 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph); 
# 8843 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType * pType); 
# 8874 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t * nodes, size_t * numNodes); 
# 8905 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes); 
# 8939 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges); 
# 8970 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies); 
# 9002 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes); 
# 9033 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
# 9064 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
# 9090 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node); 
# 9126 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, cudaGraphNode_t * pErrorNode, char * pLogBuffer, size_t bufferSize); 
# 9160 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
# 9185 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
# 9206 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec); 
# 9226 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphDestroy(cudaGraph_t graph); 
# 9231
extern cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
# 9476 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
}
# 104 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/channel_descriptor.h"
template< class T> inline cudaChannelFormatDesc cudaCreateChannelDesc() 
# 105
{ 
# 106
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
# 107
} 
# 109
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
# 110
{ 
# 111
int e = (((int)sizeof(unsigned short)) * 8); 
# 113
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 114
} 
# 116
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
# 117
{ 
# 118
int e = (((int)sizeof(unsigned short)) * 8); 
# 120
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 121
} 
# 123
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
# 124
{ 
# 125
int e = (((int)sizeof(unsigned short)) * 8); 
# 127
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 128
} 
# 130
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
# 131
{ 
# 132
int e = (((int)sizeof(unsigned short)) * 8); 
# 134
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 135
} 
# 137
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
# 138
{ 
# 139
int e = (((int)sizeof(char)) * 8); 
# 144
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 146
} 
# 148
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
# 149
{ 
# 150
int e = (((int)sizeof(signed char)) * 8); 
# 152
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 153
} 
# 155
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
# 156
{ 
# 157
int e = (((int)sizeof(unsigned char)) * 8); 
# 159
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 160
} 
# 162
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
# 163
{ 
# 164
int e = (((int)sizeof(signed char)) * 8); 
# 166
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 167
} 
# 169
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
# 170
{ 
# 171
int e = (((int)sizeof(unsigned char)) * 8); 
# 173
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 174
} 
# 176
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
# 177
{ 
# 178
int e = (((int)sizeof(signed char)) * 8); 
# 180
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 181
} 
# 183
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
# 184
{ 
# 185
int e = (((int)sizeof(unsigned char)) * 8); 
# 187
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 188
} 
# 190
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
# 191
{ 
# 192
int e = (((int)sizeof(signed char)) * 8); 
# 194
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 195
} 
# 197
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
# 198
{ 
# 199
int e = (((int)sizeof(unsigned char)) * 8); 
# 201
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 202
} 
# 204
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
# 205
{ 
# 206
int e = (((int)sizeof(short)) * 8); 
# 208
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 209
} 
# 211
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
# 212
{ 
# 213
int e = (((int)sizeof(unsigned short)) * 8); 
# 215
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 216
} 
# 218
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
# 219
{ 
# 220
int e = (((int)sizeof(short)) * 8); 
# 222
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 223
} 
# 225
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
# 226
{ 
# 227
int e = (((int)sizeof(unsigned short)) * 8); 
# 229
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 230
} 
# 232
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
# 233
{ 
# 234
int e = (((int)sizeof(short)) * 8); 
# 236
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 237
} 
# 239
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
# 240
{ 
# 241
int e = (((int)sizeof(unsigned short)) * 8); 
# 243
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 244
} 
# 246
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
# 247
{ 
# 248
int e = (((int)sizeof(short)) * 8); 
# 250
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 251
} 
# 253
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
# 254
{ 
# 255
int e = (((int)sizeof(unsigned short)) * 8); 
# 257
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 258
} 
# 260
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
# 261
{ 
# 262
int e = (((int)sizeof(int)) * 8); 
# 264
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 265
} 
# 267
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
# 268
{ 
# 269
int e = (((int)sizeof(unsigned)) * 8); 
# 271
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 272
} 
# 274
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
# 275
{ 
# 276
int e = (((int)sizeof(int)) * 8); 
# 278
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 279
} 
# 281
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
# 282
{ 
# 283
int e = (((int)sizeof(unsigned)) * 8); 
# 285
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 286
} 
# 288
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
# 289
{ 
# 290
int e = (((int)sizeof(int)) * 8); 
# 292
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 293
} 
# 295
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
# 296
{ 
# 297
int e = (((int)sizeof(unsigned)) * 8); 
# 299
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 300
} 
# 302
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
# 303
{ 
# 304
int e = (((int)sizeof(int)) * 8); 
# 306
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 307
} 
# 309
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
# 310
{ 
# 311
int e = (((int)sizeof(unsigned)) * 8); 
# 313
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 314
} 
# 376 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
# 377
{ 
# 378
int e = (((int)sizeof(float)) * 8); 
# 380
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 381
} 
# 383
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
# 384
{ 
# 385
int e = (((int)sizeof(float)) * 8); 
# 387
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 388
} 
# 390
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
# 391
{ 
# 392
int e = (((int)sizeof(float)) * 8); 
# 394
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 395
} 
# 397
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
# 398
{ 
# 399
int e = (((int)sizeof(float)) * 8); 
# 401
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 402
} 
# 79 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_functions.h"
static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
# 80
{ 
# 81
cudaPitchedPtr s; 
# 83
(s.ptr) = d; 
# 84
(s.pitch) = p; 
# 85
(s.xsize) = xsz; 
# 86
(s.ysize) = ysz; 
# 88
return s; 
# 89
} 
# 106 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_functions.h"
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
# 107
{ 
# 108
cudaPos p; 
# 110
(p.x) = x; 
# 111
(p.y) = y; 
# 112
(p.z) = z; 
# 114
return p; 
# 115
} 
# 132 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/driver_functions.h"
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
# 133
{ 
# 134
cudaExtent e; 
# 136
(e.width) = w; 
# 137
(e.height) = h; 
# 138
(e.depth) = d; 
# 140
return e; 
# 141
} 
# 73 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_functions.h"
static inline char1 make_char1(signed char x); 
# 75
static inline uchar1 make_uchar1(unsigned char x); 
# 77
static inline char2 make_char2(signed char x, signed char y); 
# 79
static inline uchar2 make_uchar2(unsigned char x, unsigned char y); 
# 81
static inline char3 make_char3(signed char x, signed char y, signed char z); 
# 83
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z); 
# 85
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w); 
# 87
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
# 89
static inline short1 make_short1(short x); 
# 91
static inline ushort1 make_ushort1(unsigned short x); 
# 93
static inline short2 make_short2(short x, short y); 
# 95
static inline ushort2 make_ushort2(unsigned short x, unsigned short y); 
# 97
static inline short3 make_short3(short x, short y, short z); 
# 99
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z); 
# 101
static inline short4 make_short4(short x, short y, short z, short w); 
# 103
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w); 
# 105
static inline int1 make_int1(int x); 
# 107
static inline uint1 make_uint1(unsigned x); 
# 109
static inline int2 make_int2(int x, int y); 
# 111
static inline uint2 make_uint2(unsigned x, unsigned y); 
# 113
static inline int3 make_int3(int x, int y, int z); 
# 115
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z); 
# 117
static inline int4 make_int4(int x, int y, int z, int w); 
# 119
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w); 
# 121
static inline long1 make_long1(long x); 
# 123
static inline ulong1 make_ulong1(unsigned long x); 
# 125
static inline long2 make_long2(long x, long y); 
# 127
static inline ulong2 make_ulong2(unsigned long x, unsigned long y); 
# 129
static inline long3 make_long3(long x, long y, long z); 
# 131
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z); 
# 133
static inline long4 make_long4(long x, long y, long z, long w); 
# 135
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
# 137
static inline float1 make_float1(float x); 
# 139
static inline float2 make_float2(float x, float y); 
# 141
static inline float3 make_float3(float x, float y, float z); 
# 143
static inline float4 make_float4(float x, float y, float z, float w); 
# 145
static inline longlong1 make_longlong1(long long x); 
# 147
static inline ulonglong1 make_ulonglong1(unsigned long long x); 
# 149
static inline longlong2 make_longlong2(long long x, long long y); 
# 151
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y); 
# 153
static inline longlong3 make_longlong3(long long x, long long y, long long z); 
# 155
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z); 
# 157
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w); 
# 159
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w); 
# 161
static inline double1 make_double1(double x); 
# 163
static inline double2 make_double2(double x, double y); 
# 165
static inline double3 make_double3(double x, double y, double z); 
# 167
static inline double4 make_double4(double x, double y, double z, double w); 
# 73 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/vector_functions.hpp"
static inline char1 make_char1(signed char x) 
# 74
{ 
# 75
char1 t; (t.x) = x; return t; 
# 76
} 
# 78
static inline uchar1 make_uchar1(unsigned char x) 
# 79
{ 
# 80
uchar1 t; (t.x) = x; return t; 
# 81
} 
# 83
static inline char2 make_char2(signed char x, signed char y) 
# 84
{ 
# 85
char2 t; (t.x) = x; (t.y) = y; return t; 
# 86
} 
# 88
static inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
# 89
{ 
# 90
uchar2 t; (t.x) = x; (t.y) = y; return t; 
# 91
} 
# 93
static inline char3 make_char3(signed char x, signed char y, signed char z) 
# 94
{ 
# 95
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 96
} 
# 98
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
# 99
{ 
# 100
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 101
} 
# 103
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
# 104
{ 
# 105
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 106
} 
# 108
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
# 109
{ 
# 110
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 111
} 
# 113
static inline short1 make_short1(short x) 
# 114
{ 
# 115
short1 t; (t.x) = x; return t; 
# 116
} 
# 118
static inline ushort1 make_ushort1(unsigned short x) 
# 119
{ 
# 120
ushort1 t; (t.x) = x; return t; 
# 121
} 
# 123
static inline short2 make_short2(short x, short y) 
# 124
{ 
# 125
short2 t; (t.x) = x; (t.y) = y; return t; 
# 126
} 
# 128
static inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
# 129
{ 
# 130
ushort2 t; (t.x) = x; (t.y) = y; return t; 
# 131
} 
# 133
static inline short3 make_short3(short x, short y, short z) 
# 134
{ 
# 135
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 136
} 
# 138
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
# 139
{ 
# 140
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 141
} 
# 143
static inline short4 make_short4(short x, short y, short z, short w) 
# 144
{ 
# 145
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 146
} 
# 148
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
# 149
{ 
# 150
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 151
} 
# 153
static inline int1 make_int1(int x) 
# 154
{ 
# 155
int1 t; (t.x) = x; return t; 
# 156
} 
# 158
static inline uint1 make_uint1(unsigned x) 
# 159
{ 
# 160
uint1 t; (t.x) = x; return t; 
# 161
} 
# 163
static inline int2 make_int2(int x, int y) 
# 164
{ 
# 165
int2 t; (t.x) = x; (t.y) = y; return t; 
# 166
} 
# 168
static inline uint2 make_uint2(unsigned x, unsigned y) 
# 169
{ 
# 170
uint2 t; (t.x) = x; (t.y) = y; return t; 
# 171
} 
# 173
static inline int3 make_int3(int x, int y, int z) 
# 174
{ 
# 175
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 176
} 
# 178
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
# 179
{ 
# 180
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 181
} 
# 183
static inline int4 make_int4(int x, int y, int z, int w) 
# 184
{ 
# 185
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 186
} 
# 188
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
# 189
{ 
# 190
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 191
} 
# 193
static inline long1 make_long1(long x) 
# 194
{ 
# 195
long1 t; (t.x) = x; return t; 
# 196
} 
# 198
static inline ulong1 make_ulong1(unsigned long x) 
# 199
{ 
# 200
ulong1 t; (t.x) = x; return t; 
# 201
} 
# 203
static inline long2 make_long2(long x, long y) 
# 204
{ 
# 205
long2 t; (t.x) = x; (t.y) = y; return t; 
# 206
} 
# 208
static inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
# 209
{ 
# 210
ulong2 t; (t.x) = x; (t.y) = y; return t; 
# 211
} 
# 213
static inline long3 make_long3(long x, long y, long z) 
# 214
{ 
# 215
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 216
} 
# 218
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
# 219
{ 
# 220
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 221
} 
# 223
static inline long4 make_long4(long x, long y, long z, long w) 
# 224
{ 
# 225
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 226
} 
# 228
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
# 229
{ 
# 230
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 231
} 
# 233
static inline float1 make_float1(float x) 
# 234
{ 
# 235
float1 t; (t.x) = x; return t; 
# 236
} 
# 238
static inline float2 make_float2(float x, float y) 
# 239
{ 
# 240
float2 t; (t.x) = x; (t.y) = y; return t; 
# 241
} 
# 243
static inline float3 make_float3(float x, float y, float z) 
# 244
{ 
# 245
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 246
} 
# 248
static inline float4 make_float4(float x, float y, float z, float w) 
# 249
{ 
# 250
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 251
} 
# 253
static inline longlong1 make_longlong1(long long x) 
# 254
{ 
# 255
longlong1 t; (t.x) = x; return t; 
# 256
} 
# 258
static inline ulonglong1 make_ulonglong1(unsigned long long x) 
# 259
{ 
# 260
ulonglong1 t; (t.x) = x; return t; 
# 261
} 
# 263
static inline longlong2 make_longlong2(long long x, long long y) 
# 264
{ 
# 265
longlong2 t; (t.x) = x; (t.y) = y; return t; 
# 266
} 
# 268
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y) 
# 269
{ 
# 270
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
# 271
} 
# 273
static inline longlong3 make_longlong3(long long x, long long y, long long z) 
# 274
{ 
# 275
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 276
} 
# 278
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z) 
# 279
{ 
# 280
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 281
} 
# 283
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w) 
# 284
{ 
# 285
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 286
} 
# 288
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w) 
# 289
{ 
# 290
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 291
} 
# 293
static inline double1 make_double1(double x) 
# 294
{ 
# 295
double1 t; (t.x) = x; return t; 
# 296
} 
# 298
static inline double2 make_double2(double x, double y) 
# 299
{ 
# 300
double2 t; (t.x) = x; (t.y) = y; return t; 
# 301
} 
# 303
static inline double3 make_double3(double x, double y, double z) 
# 304
{ 
# 305
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 306
} 
# 308
static inline double4 make_double4(double x, double y, double z, double w) 
# 309
{ 
# 310
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 311
} 
# 27 "/usr/include/string.h" 3
extern "C" {
# 42 "/usr/include/string.h" 3
extern void *memcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 43
 __attribute((__nonnull__(1, 2))); 
# 46
extern void *memmove(void * __dest, const void * __src, size_t __n) throw()
# 47
 __attribute((__nonnull__(1, 2))); 
# 54
extern void *memccpy(void *__restrict__ __dest, const void *__restrict__ __src, int __c, size_t __n) throw()
# 56
 __attribute((__nonnull__(1, 2))); 
# 62
extern void *memset(void * __s, int __c, size_t __n) throw() __attribute((__nonnull__(1))); 
# 65
extern int memcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 66
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 70
extern "C++" {
# 72
extern __attribute((gnu_inline)) inline void *memchr(void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 73
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 74
extern __attribute((gnu_inline)) inline const void *memchr(const void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 75
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 78
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline void *
# 79
memchr(void *__s, int __c, size_t __n) throw() 
# 80
{ 
# 81
return __builtin_memchr(__s, __c, __n); 
# 82
} 
# 84
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline const void *
# 85
memchr(const void *__s, int __c, size_t __n) throw() 
# 86
{ 
# 87
return __builtin_memchr(__s, __c, __n); 
# 88
} 
# 90
}
# 101
extern "C++" void *rawmemchr(void * __s, int __c) throw() __asm__("rawmemchr")
# 102
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 103
extern "C++" const void *rawmemchr(const void * __s, int __c) throw() __asm__("rawmemchr")
# 104
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 112
extern "C++" void *memrchr(void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 113
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 114
extern "C++" const void *memrchr(const void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 115
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 125
extern char *strcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 126
 __attribute((__nonnull__(1, 2))); 
# 128
extern char *strncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 130
 __attribute((__nonnull__(1, 2))); 
# 133
extern char *strcat(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 134
 __attribute((__nonnull__(1, 2))); 
# 136
extern char *strncat(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 137
 __attribute((__nonnull__(1, 2))); 
# 140
extern int strcmp(const char * __s1, const char * __s2) throw()
# 141
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 143
extern int strncmp(const char * __s1, const char * __s2, size_t __n) throw()
# 144
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 147
extern int strcoll(const char * __s1, const char * __s2) throw()
# 148
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 150
extern size_t strxfrm(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 152
 __attribute((__nonnull__(2))); 
# 39 "/usr/include/xlocale.h" 3
typedef 
# 27
struct __locale_struct { 
# 30
struct __locale_data *__locales[13]; 
# 33
const unsigned short *__ctype_b; 
# 34
const int *__ctype_tolower; 
# 35
const int *__ctype_toupper; 
# 38
const char *__names[13]; 
# 39
} *__locale_t; 
# 42
typedef __locale_t locale_t; 
# 162 "/usr/include/string.h" 3
extern int strcoll_l(const char * __s1, const char * __s2, __locale_t __l) throw()
# 163
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 165
extern size_t strxfrm_l(char * __dest, const char * __src, size_t __n, __locale_t __l) throw()
# 166
 __attribute((__nonnull__(2, 4))); 
# 172
extern char *strdup(const char * __s) throw()
# 173
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 180
extern char *strndup(const char * __string, size_t __n) throw()
# 181
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 210 "/usr/include/string.h" 3
extern "C++" {
# 212
extern __attribute((gnu_inline)) inline char *strchr(char * __s, int __c) throw() __asm__("strchr")
# 213
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 214
extern __attribute((gnu_inline)) inline const char *strchr(const char * __s, int __c) throw() __asm__("strchr")
# 215
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 218
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline char *
# 219
strchr(char *__s, int __c) throw() 
# 220
{ 
# 221
return __builtin_strchr(__s, __c); 
# 222
} 
# 224
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline const char *
# 225
strchr(const char *__s, int __c) throw() 
# 226
{ 
# 227
return __builtin_strchr(__s, __c); 
# 228
} 
# 230
}
# 237
extern "C++" {
# 239
extern __attribute((gnu_inline)) inline char *strrchr(char * __s, int __c) throw() __asm__("strrchr")
# 240
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 241
extern __attribute((gnu_inline)) inline const char *strrchr(const char * __s, int __c) throw() __asm__("strrchr")
# 242
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 245
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline char *
# 246
strrchr(char *__s, int __c) throw() 
# 247
{ 
# 248
return __builtin_strrchr(__s, __c); 
# 249
} 
# 251
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline const char *
# 252
strrchr(const char *__s, int __c) throw() 
# 253
{ 
# 254
return __builtin_strrchr(__s, __c); 
# 255
} 
# 257
}
# 268
extern "C++" char *strchrnul(char * __s, int __c) throw() __asm__("strchrnul")
# 269
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 270
extern "C++" const char *strchrnul(const char * __s, int __c) throw() __asm__("strchrnul")
# 271
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 281
extern size_t strcspn(const char * __s, const char * __reject) throw()
# 282
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 285
extern size_t strspn(const char * __s, const char * __accept) throw()
# 286
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 289
extern "C++" {
# 291
extern __attribute((gnu_inline)) inline char *strpbrk(char * __s, const char * __accept) throw() __asm__("strpbrk")
# 292
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 293
extern __attribute((gnu_inline)) inline const char *strpbrk(const char * __s, const char * __accept) throw() __asm__("strpbrk")
# 294
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 297
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline char *
# 298
strpbrk(char *__s, const char *__accept) throw() 
# 299
{ 
# 300
return __builtin_strpbrk(__s, __accept); 
# 301
} 
# 303
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline const char *
# 304
strpbrk(const char *__s, const char *__accept) throw() 
# 305
{ 
# 306
return __builtin_strpbrk(__s, __accept); 
# 307
} 
# 309
}
# 316
extern "C++" {
# 318
extern __attribute((gnu_inline)) inline char *strstr(char * __haystack, const char * __needle) throw() __asm__("strstr")
# 319
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 320
extern __attribute((gnu_inline)) inline const char *strstr(const char * __haystack, const char * __needle) throw() __asm__("strstr")
# 321
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 324
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline char *
# 325
strstr(char *__haystack, const char *__needle) throw() 
# 326
{ 
# 327
return __builtin_strstr(__haystack, __needle); 
# 328
} 
# 330
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline const char *
# 331
strstr(const char *__haystack, const char *__needle) throw() 
# 332
{ 
# 333
return __builtin_strstr(__haystack, __needle); 
# 334
} 
# 336
}
# 344
extern char *strtok(char *__restrict__ __s, const char *__restrict__ __delim) throw()
# 345
 __attribute((__nonnull__(2))); 
# 350
extern char *__strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 353
 __attribute((__nonnull__(2, 3))); 
# 355
extern char *strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 357
 __attribute((__nonnull__(2, 3))); 
# 363
extern "C++" char *strcasestr(char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 364
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 365
extern "C++" const char *strcasestr(const char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 367
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 378 "/usr/include/string.h" 3
extern void *memmem(const void * __haystack, size_t __haystacklen, const void * __needle, size_t __needlelen) throw()
# 380
 __attribute((__pure__)) __attribute((__nonnull__(1, 3))); 
# 384
extern void *__mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 386
 __attribute((__nonnull__(1, 2))); 
# 387
extern void *mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 389
 __attribute((__nonnull__(1, 2))); 
# 395
extern size_t strlen(const char * __s) throw()
# 396
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 402
extern size_t strnlen(const char * __string, size_t __maxlen) throw()
# 403
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 409
extern char *strerror(int __errnum) throw(); 
# 434 "/usr/include/string.h" 3
extern char *strerror_r(int __errnum, char * __buf, size_t __buflen) throw()
# 435
 __attribute((__nonnull__(2))); 
# 441
extern char *strerror_l(int __errnum, __locale_t __l) throw(); 
# 447
extern void __bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 451
extern void bcopy(const void * __src, void * __dest, size_t __n) throw()
# 452
 __attribute((__nonnull__(1, 2))); 
# 455
extern void bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 458
extern int bcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 459
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 463
extern "C++" {
# 465
extern __attribute((gnu_inline)) inline char *index(char * __s, int __c) throw() __asm__("index")
# 466
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 467
extern __attribute((gnu_inline)) inline const char *index(const char * __s, int __c) throw() __asm__("index")
# 468
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 471
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline char *
# 472
index(char *__s, int __c) throw() 
# 473
{ 
# 474
return __builtin_index(__s, __c); 
# 475
} 
# 477
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline const char *
# 478
index(const char *__s, int __c) throw() 
# 479
{ 
# 480
return __builtin_index(__s, __c); 
# 481
} 
# 483
}
# 491
extern "C++" {
# 493
extern __attribute((gnu_inline)) inline char *rindex(char * __s, int __c) throw() __asm__("rindex")
# 494
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 495
extern __attribute((gnu_inline)) inline const char *rindex(const char * __s, int __c) throw() __asm__("rindex")
# 496
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 499
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline char *
# 500
rindex(char *__s, int __c) throw() 
# 501
{ 
# 502
return __builtin_rindex(__s, __c); 
# 503
} 
# 505
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline const char *
# 506
rindex(const char *__s, int __c) throw() 
# 507
{ 
# 508
return __builtin_rindex(__s, __c); 
# 509
} 
# 511
}
# 519
extern int ffs(int __i) throw() __attribute((const)); 
# 524
extern int ffsl(long __l) throw() __attribute((const)); 
# 526
__extension__ extern int ffsll(long long __ll) throw()
# 527
 __attribute((const)); 
# 532
extern int strcasecmp(const char * __s1, const char * __s2) throw()
# 533
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 536
extern int strncasecmp(const char * __s1, const char * __s2, size_t __n) throw()
# 537
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 543
extern int strcasecmp_l(const char * __s1, const char * __s2, __locale_t __loc) throw()
# 545
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 547
extern int strncasecmp_l(const char * __s1, const char * __s2, size_t __n, __locale_t __loc) throw()
# 549
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 4))); 
# 555
extern char *strsep(char **__restrict__ __stringp, const char *__restrict__ __delim) throw()
# 557
 __attribute((__nonnull__(1, 2))); 
# 562
extern char *strsignal(int __sig) throw(); 
# 565
extern char *__stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 566
 __attribute((__nonnull__(1, 2))); 
# 567
extern char *stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 568
 __attribute((__nonnull__(1, 2))); 
# 572
extern char *__stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 574
 __attribute((__nonnull__(1, 2))); 
# 575
extern char *stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 577
 __attribute((__nonnull__(1, 2))); 
# 582
extern int strverscmp(const char * __s1, const char * __s2) throw()
# 583
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 586
extern char *strfry(char * __string) throw() __attribute((__nonnull__(1))); 
# 589
extern void *memfrob(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 597
extern "C++" char *basename(char * __filename) throw() __asm__("basename")
# 598
 __attribute((__nonnull__(1))); 
# 599
extern "C++" const char *basename(const char * __filename) throw() __asm__("basename")
# 600
 __attribute((__nonnull__(1))); 
# 642 "/usr/include/string.h" 3
}
# 29 "/usr/include/time.h" 3
extern "C" {
# 30 "/usr/include/bits/types.h" 3
typedef unsigned char __u_char; 
# 31
typedef unsigned short __u_short; 
# 32
typedef unsigned __u_int; 
# 33
typedef unsigned long __u_long; 
# 36
typedef signed char __int8_t; 
# 37
typedef unsigned char __uint8_t; 
# 38
typedef signed short __int16_t; 
# 39
typedef unsigned short __uint16_t; 
# 40
typedef signed int __int32_t; 
# 41
typedef unsigned __uint32_t; 
# 43
typedef signed long __int64_t; 
# 44
typedef unsigned long __uint64_t; 
# 52
typedef long __quad_t; 
# 53
typedef unsigned long __u_quad_t; 
# 133 "/usr/include/bits/types.h" 3
typedef unsigned long __dev_t; 
# 134
typedef unsigned __uid_t; 
# 135
typedef unsigned __gid_t; 
# 136
typedef unsigned long __ino_t; 
# 137
typedef unsigned long __ino64_t; 
# 138
typedef unsigned __mode_t; 
# 139
typedef unsigned long __nlink_t; 
# 140
typedef long __off_t; 
# 141
typedef long __off64_t; 
# 142
typedef int __pid_t; 
# 143
typedef struct { int __val[2]; } __fsid_t; 
# 144
typedef long __clock_t; 
# 145
typedef unsigned long __rlim_t; 
# 146
typedef unsigned long __rlim64_t; 
# 147
typedef unsigned __id_t; 
# 148
typedef long __time_t; 
# 149
typedef unsigned __useconds_t; 
# 150
typedef long __suseconds_t; 
# 152
typedef int __daddr_t; 
# 153
typedef int __key_t; 
# 156
typedef int __clockid_t; 
# 159
typedef void *__timer_t; 
# 162
typedef long __blksize_t; 
# 167
typedef long __blkcnt_t; 
# 168
typedef long __blkcnt64_t; 
# 171
typedef unsigned long __fsblkcnt_t; 
# 172
typedef unsigned long __fsblkcnt64_t; 
# 175
typedef unsigned long __fsfilcnt_t; 
# 176
typedef unsigned long __fsfilcnt64_t; 
# 179
typedef long __fsword_t; 
# 181
typedef long __ssize_t; 
# 184
typedef long __syscall_slong_t; 
# 186
typedef unsigned long __syscall_ulong_t; 
# 190
typedef __off64_t __loff_t; 
# 191
typedef __quad_t *__qaddr_t; 
# 192
typedef char *__caddr_t; 
# 195
typedef long __intptr_t; 
# 198
typedef unsigned __socklen_t; 
# 30 "/usr/include/bits/time.h" 3
struct timeval { 
# 32
__time_t tv_sec; 
# 33
__suseconds_t tv_usec; 
# 34
}; 
# 25 "/usr/include/bits/timex.h" 3
struct timex { 
# 27
unsigned modes; 
# 28
__syscall_slong_t offset; 
# 29
__syscall_slong_t freq; 
# 30
__syscall_slong_t maxerror; 
# 31
__syscall_slong_t esterror; 
# 32
int status; 
# 33
__syscall_slong_t constant; 
# 34
__syscall_slong_t precision; 
# 35
__syscall_slong_t tolerance; 
# 36
timeval time; 
# 37
__syscall_slong_t tick; 
# 38
__syscall_slong_t ppsfreq; 
# 39
__syscall_slong_t jitter; 
# 40
int shift; 
# 41
__syscall_slong_t stabil; 
# 42
__syscall_slong_t jitcnt; 
# 43
__syscall_slong_t calcnt; 
# 44
__syscall_slong_t errcnt; 
# 45
__syscall_slong_t stbcnt; 
# 47
int tai; 
# 50
int:32; int:32; int:32; int:32; 
# 51
int:32; int:32; int:32; int:32; 
# 52
int:32; int:32; int:32; 
# 53
}; 
# 90 "/usr/include/bits/time.h" 3
extern "C" {
# 93
extern int clock_adjtime(__clockid_t __clock_id, timex * __utx) throw(); 
# 95
}
# 59 "/usr/include/time.h" 3
typedef __clock_t clock_t; 
# 75 "/usr/include/time.h" 3
typedef __time_t time_t; 
# 91 "/usr/include/time.h" 3
typedef __clockid_t clockid_t; 
# 103 "/usr/include/time.h" 3
typedef __timer_t timer_t; 
# 120 "/usr/include/time.h" 3
struct timespec { 
# 122
__time_t tv_sec; 
# 123
__syscall_slong_t tv_nsec; 
# 124
}; 
# 133
struct tm { 
# 135
int tm_sec; 
# 136
int tm_min; 
# 137
int tm_hour; 
# 138
int tm_mday; 
# 139
int tm_mon; 
# 140
int tm_year; 
# 141
int tm_wday; 
# 142
int tm_yday; 
# 143
int tm_isdst; 
# 146
long tm_gmtoff; 
# 147
const char *tm_zone; 
# 152
}; 
# 161
struct itimerspec { 
# 163
timespec it_interval; 
# 164
timespec it_value; 
# 165
}; 
# 168
struct sigevent; 
# 174
typedef __pid_t pid_t; 
# 189 "/usr/include/time.h" 3
extern clock_t clock() throw(); 
# 192
extern time_t time(time_t * __timer) throw(); 
# 195
extern double difftime(time_t __time1, time_t __time0) throw()
# 196
 __attribute((const)); 
# 199
extern time_t mktime(tm * __tp) throw(); 
# 205
extern size_t strftime(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp) throw(); 
# 213
extern char *strptime(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp) throw(); 
# 223
extern size_t strftime_l(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp, __locale_t __loc) throw(); 
# 230
extern char *strptime_l(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp, __locale_t __loc) throw(); 
# 239
extern tm *gmtime(const time_t * __timer) throw(); 
# 243
extern tm *localtime(const time_t * __timer) throw(); 
# 249
extern tm *gmtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 254
extern tm *localtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 261
extern char *asctime(const tm * __tp) throw(); 
# 264
extern char *ctime(const time_t * __timer) throw(); 
# 272
extern char *asctime_r(const tm *__restrict__ __tp, char *__restrict__ __buf) throw(); 
# 276
extern char *ctime_r(const time_t *__restrict__ __timer, char *__restrict__ __buf) throw(); 
# 282
extern char *__tzname[2]; 
# 283
extern int __daylight; 
# 284
extern long __timezone; 
# 289
extern char *tzname[2]; 
# 293
extern void tzset() throw(); 
# 297
extern int daylight; 
# 298
extern long timezone; 
# 304
extern int stime(const time_t * __when) throw(); 
# 319 "/usr/include/time.h" 3
extern time_t timegm(tm * __tp) throw(); 
# 322
extern time_t timelocal(tm * __tp) throw(); 
# 325
extern int dysize(int __year) throw() __attribute((const)); 
# 334 "/usr/include/time.h" 3
extern int nanosleep(const timespec * __requested_time, timespec * __remaining); 
# 339
extern int clock_getres(clockid_t __clock_id, timespec * __res) throw(); 
# 342
extern int clock_gettime(clockid_t __clock_id, timespec * __tp) throw(); 
# 345
extern int clock_settime(clockid_t __clock_id, const timespec * __tp) throw(); 
# 353
extern int clock_nanosleep(clockid_t __clock_id, int __flags, const timespec * __req, timespec * __rem); 
# 358
extern int clock_getcpuclockid(pid_t __pid, clockid_t * __clock_id) throw(); 
# 363
extern int timer_create(clockid_t __clock_id, sigevent *__restrict__ __evp, timer_t *__restrict__ __timerid) throw(); 
# 368
extern int timer_delete(timer_t __timerid) throw(); 
# 371
extern int timer_settime(timer_t __timerid, int __flags, const itimerspec *__restrict__ __value, itimerspec *__restrict__ __ovalue) throw(); 
# 376
extern int timer_gettime(timer_t __timerid, itimerspec * __value) throw(); 
# 380
extern int timer_getoverrun(timer_t __timerid) throw(); 
# 386
extern int timespec_get(timespec * __ts, int __base) throw()
# 387
 __attribute((__nonnull__(1))); 
# 403 "/usr/include/time.h" 3
extern int getdate_err; 
# 412 "/usr/include/time.h" 3
extern tm *getdate(const char * __string); 
# 426 "/usr/include/time.h" 3
extern int getdate_r(const char *__restrict__ __string, tm *__restrict__ __resbufp); 
# 430
}
# 80 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/common_functions.h"
extern "C" {
# 83
extern clock_t clock() throw(); 
# 88
extern void *memset(void *, int, size_t) throw(); 
# 89
extern void *memcpy(void *, const void *, size_t) throw(); 
# 91
}
# 108 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern "C" {
# 192 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int abs(int) throw(); 
# 193
extern long labs(long) throw(); 
# 194
extern long long llabs(long long) throw(); 
# 244 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fabs(double x) throw(); 
# 285 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fabsf(float x) throw(); 
# 289
extern inline int min(int, int); 
# 291
extern inline unsigned umin(unsigned, unsigned); 
# 292
extern inline long long llmin(long long, long long); 
# 293
extern inline unsigned long long ullmin(unsigned long long, unsigned long long); 
# 314 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fminf(float x, float y) throw(); 
# 334 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fmin(double x, double y) throw(); 
# 341
extern inline int max(int, int); 
# 343
extern inline unsigned umax(unsigned, unsigned); 
# 344
extern inline long long llmax(long long, long long); 
# 345
extern inline unsigned long long ullmax(unsigned long long, unsigned long long); 
# 366 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fmaxf(float x, float y) throw(); 
# 386 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fmax(double, double) throw(); 
# 430 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double sin(double x) throw(); 
# 463 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cos(double x) throw(); 
# 482 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern void sincos(double x, double * sptr, double * cptr) throw(); 
# 498 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern void sincosf(float x, float * sptr, float * cptr) throw(); 
# 543 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double tan(double x) throw(); 
# 612 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double sqrt(double x) throw(); 
# 684 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rsqrt(double x); 
# 754 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rsqrtf(float x); 
# 810 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double log2(double x) throw(); 
# 835 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double exp2(double x) throw(); 
# 860 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float exp2f(float x) throw(); 
# 887 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double exp10(double x) throw(); 
# 910 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float exp10f(float x) throw(); 
# 956 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double expm1(double x) throw(); 
# 1001 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float expm1f(float x) throw(); 
# 1056 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float log2f(float x) throw(); 
# 1110 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double log10(double x) throw(); 
# 1181 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double log(double x) throw(); 
# 1275 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double log1p(double x) throw(); 
# 1372 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float log1pf(float x) throw(); 
# 1447 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double floor(double x) throw(); 
# 1486 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double exp(double x) throw(); 
# 1517 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cosh(double x) throw(); 
# 1547 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double sinh(double x) throw(); 
# 1577 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double tanh(double x) throw(); 
# 1612 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double acosh(double x) throw(); 
# 1650 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float acoshf(float x) throw(); 
# 1666 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double asinh(double x) throw(); 
# 1682 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float asinhf(float x) throw(); 
# 1736 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double atanh(double x) throw(); 
# 1790 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float atanhf(float x) throw(); 
# 1849 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double ldexp(double x, int exp) throw(); 
# 1905 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float ldexpf(float x, int exp) throw(); 
# 1957 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double logb(double x) throw(); 
# 2012 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float logbf(float x) throw(); 
# 2042 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int ilogb(double x) throw(); 
# 2072 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int ilogbf(float x) throw(); 
# 2148 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double scalbn(double x, int n) throw(); 
# 2224 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float scalbnf(float x, int n) throw(); 
# 2300 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double scalbln(double x, long n) throw(); 
# 2376 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float scalblnf(float x, long n) throw(); 
# 2454 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double frexp(double x, int * nptr) throw(); 
# 2529 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float frexpf(float x, int * nptr) throw(); 
# 2543 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double round(double x) throw(); 
# 2560 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float roundf(float x) throw(); 
# 2578 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long lround(double x) throw(); 
# 2596 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long lroundf(float x) throw(); 
# 2614 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long long llround(double x) throw(); 
# 2632 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long long llroundf(float x) throw(); 
# 2684 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rintf(float x) throw(); 
# 2701 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long lrint(double x) throw(); 
# 2718 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long lrintf(float x) throw(); 
# 2735 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long long llrint(double x) throw(); 
# 2752 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long long llrintf(float x) throw(); 
# 2805 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double nearbyint(double x) throw(); 
# 2858 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float nearbyintf(float x) throw(); 
# 2920 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double ceil(double x) throw(); 
# 2932 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double trunc(double x) throw(); 
# 2947 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float truncf(float x) throw(); 
# 2973 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fdim(double x, double y) throw(); 
# 2999 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fdimf(float x, float y) throw(); 
# 3035 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double atan2(double y, double x) throw(); 
# 3066 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double atan(double x) throw(); 
# 3089 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double acos(double x) throw(); 
# 3121 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double asin(double x) throw(); 
# 3167 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double hypot(double x, double y) throw(); 
# 3219 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rhypot(double x, double y) throw(); 
# 3265 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float hypotf(float x, float y) throw(); 
# 3317 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rhypotf(float x, float y) throw(); 
# 3361 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double norm3d(double a, double b, double c) throw(); 
# 3412 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rnorm3d(double a, double b, double c) throw(); 
# 3461 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double norm4d(double a, double b, double c, double d) throw(); 
# 3517 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rnorm4d(double a, double b, double c, double d) throw(); 
# 3562 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double norm(int dim, const double * t) throw(); 
# 3613 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rnorm(int dim, const double * t) throw(); 
# 3665 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rnormf(int dim, const float * a) throw(); 
# 3709 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float normf(int dim, const float * a) throw(); 
# 3754 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float norm3df(float a, float b, float c) throw(); 
# 3805 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rnorm3df(float a, float b, float c) throw(); 
# 3854 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float norm4df(float a, float b, float c, float d) throw(); 
# 3910 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rnorm4df(float a, float b, float c, float d) throw(); 
# 3997 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cbrt(double x) throw(); 
# 4083 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float cbrtf(float x) throw(); 
# 4138 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rcbrt(double x); 
# 4188 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rcbrtf(float x); 
# 4248 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double sinpi(double x); 
# 4308 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float sinpif(float x); 
# 4360 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cospi(double x); 
# 4412 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float cospif(float x); 
# 4442 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern void sincospi(double x, double * sptr, double * cptr); 
# 4472 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern void sincospif(float x, float * sptr, float * cptr); 
# 4784 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double pow(double x, double y) throw(); 
# 4840 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double modf(double x, double * iptr) throw(); 
# 4899 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fmod(double x, double y) throw(); 
# 4985 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double remainder(double x, double y) throw(); 
# 5075 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float remainderf(float x, float y) throw(); 
# 5129 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double remquo(double x, double y, int * quo) throw(); 
# 5183 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float remquof(float x, float y, int * quo) throw(); 
# 5224 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double j0(double x) throw(); 
# 5266 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float j0f(float x) throw(); 
# 5327 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double j1(double x) throw(); 
# 5388 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float j1f(float x) throw(); 
# 5431 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double jn(int n, double x) throw(); 
# 5474 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float jnf(int n, float x) throw(); 
# 5526 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double y0(double x) throw(); 
# 5578 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float y0f(float x) throw(); 
# 5630 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double y1(double x) throw(); 
# 5682 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float y1f(float x) throw(); 
# 5735 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double yn(int n, double x) throw(); 
# 5788 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float ynf(int n, float x) throw(); 
# 5815 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cyl_bessel_i0(double x) throw(); 
# 5841 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float cyl_bessel_i0f(float x) throw(); 
# 5868 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cyl_bessel_i1(double x) throw(); 
# 5894 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float cyl_bessel_i1f(float x) throw(); 
# 5977 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erf(double x) throw(); 
# 6059 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erff(float x) throw(); 
# 6123 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erfinv(double y); 
# 6180 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erfinvf(float y); 
# 6219 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erfc(double x) throw(); 
# 6257 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erfcf(float x) throw(); 
# 6385 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double lgamma(double x) throw(); 
# 6448 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erfcinv(double y); 
# 6504 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erfcinvf(float y); 
# 6562 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double normcdfinv(double y); 
# 6620 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float normcdfinvf(float y); 
# 6663 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double normcdf(double y); 
# 6706 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float normcdff(float y); 
# 6781 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erfcx(double x); 
# 6856 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erfcxf(float x); 
# 6990 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float lgammaf(float x) throw(); 
# 7099 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double tgamma(double x) throw(); 
# 7208 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float tgammaf(float x) throw(); 
# 7221 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double copysign(double x, double y) throw(); 
# 7234 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float copysignf(float x, float y) throw(); 
# 7271 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double nextafter(double x, double y) throw(); 
# 7308 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float nextafterf(float x, float y) throw(); 
# 7324 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double nan(const char * tagp) throw(); 
# 7340 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float nanf(const char * tagp) throw(); 
# 7347
extern int __isinff(float) throw(); 
# 7348
extern int __isnanf(float) throw(); 
# 7358 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __finite(double) throw(); 
# 7359
extern int __finitef(float) throw(); 
# 7360
extern __attribute((gnu_inline)) inline int __signbit(double) throw(); 
# 7361
extern int __isnan(double) throw(); 
# 7362
extern int __isinf(double) throw(); 
# 7365
extern __attribute((gnu_inline)) inline int __signbitf(float) throw(); 
# 7524 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fma(double x, double y, double z) throw(); 
# 7682 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fmaf(float x, float y, float z) throw(); 
# 7693 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern __attribute((gnu_inline)) inline int __signbitl(long double) throw(); 
# 7699
extern int __finitel(long double) throw(); 
# 7700
extern int __isinfl(long double) throw(); 
# 7701
extern int __isnanl(long double) throw(); 
# 7751 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float acosf(float x) throw(); 
# 7791 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float asinf(float x) throw(); 
# 7831 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float atanf(float x) throw(); 
# 7864 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float atan2f(float y, float x) throw(); 
# 7888 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float cosf(float x) throw(); 
# 7930 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float sinf(float x) throw(); 
# 7972 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float tanf(float x) throw(); 
# 7996 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float coshf(float x) throw(); 
# 8037 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float sinhf(float x) throw(); 
# 8067 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float tanhf(float x) throw(); 
# 8118 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float logf(float x) throw(); 
# 8168 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float expf(float x) throw(); 
# 8219 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float log10f(float x) throw(); 
# 8274 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float modff(float x, float * iptr) throw(); 
# 8582 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float powf(float x, float y) throw(); 
# 8651 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float sqrtf(float x) throw(); 
# 8710 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float ceilf(float x) throw(); 
# 8782 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float floorf(float x) throw(); 
# 8841 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fmodf(float x, float y) throw(); 
# 8856 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
}
# 29 "/usr/include/math.h" 3
extern "C" {
# 28 "/usr/include/bits/mathdef.h" 3
typedef float float_t; 
# 29
typedef double double_t; 
# 54 "/usr/include/bits/mathcalls.h" 3
extern double acos(double __x) throw(); extern double __acos(double __x) throw(); 
# 56
extern double asin(double __x) throw(); extern double __asin(double __x) throw(); 
# 58
extern double atan(double __x) throw(); extern double __atan(double __x) throw(); 
# 60
extern double atan2(double __y, double __x) throw(); extern double __atan2(double __y, double __x) throw(); 
# 63
extern double cos(double __x) throw(); extern double __cos(double __x) throw(); 
# 65
extern double sin(double __x) throw(); extern double __sin(double __x) throw(); 
# 67
extern double tan(double __x) throw(); extern double __tan(double __x) throw(); 
# 72
extern double cosh(double __x) throw(); extern double __cosh(double __x) throw(); 
# 74
extern double sinh(double __x) throw(); extern double __sinh(double __x) throw(); 
# 76
extern double tanh(double __x) throw(); extern double __tanh(double __x) throw(); 
# 81
extern void sincos(double __x, double * __sinx, double * __cosx) throw(); extern void __sincos(double __x, double * __sinx, double * __cosx) throw(); 
# 88
extern double acosh(double __x) throw(); extern double __acosh(double __x) throw(); 
# 90
extern double asinh(double __x) throw(); extern double __asinh(double __x) throw(); 
# 92
extern double atanh(double __x) throw(); extern double __atanh(double __x) throw(); 
# 100
extern double exp(double __x) throw(); extern double __exp(double __x) throw(); 
# 103
extern double frexp(double __x, int * __exponent) throw(); extern double __frexp(double __x, int * __exponent) throw(); 
# 106
extern double ldexp(double __x, int __exponent) throw(); extern double __ldexp(double __x, int __exponent) throw(); 
# 109
extern double log(double __x) throw(); extern double __log(double __x) throw(); 
# 112
extern double log10(double __x) throw(); extern double __log10(double __x) throw(); 
# 115
extern double modf(double __x, double * __iptr) throw(); extern double __modf(double __x, double * __iptr) throw()
# 116
 __attribute((__nonnull__(2))); 
# 121
extern double exp10(double __x) throw(); extern double __exp10(double __x) throw(); 
# 123
extern double pow10(double __x) throw(); extern double __pow10(double __x) throw(); 
# 129
extern double expm1(double __x) throw(); extern double __expm1(double __x) throw(); 
# 132
extern double log1p(double __x) throw(); extern double __log1p(double __x) throw(); 
# 135
extern double logb(double __x) throw(); extern double __logb(double __x) throw(); 
# 142
extern double exp2(double __x) throw(); extern double __exp2(double __x) throw(); 
# 145
extern double log2(double __x) throw(); extern double __log2(double __x) throw(); 
# 154
extern double pow(double __x, double __y) throw(); extern double __pow(double __x, double __y) throw(); 
# 157
extern double sqrt(double __x) throw(); extern double __sqrt(double __x) throw(); 
# 163
extern double hypot(double __x, double __y) throw(); extern double __hypot(double __x, double __y) throw(); 
# 170
extern double cbrt(double __x) throw(); extern double __cbrt(double __x) throw(); 
# 179
extern double ceil(double __x) throw() __attribute((const)); extern double __ceil(double __x) throw() __attribute((const)); 
# 182
extern double fabs(double __x) throw() __attribute((const)); extern double __fabs(double __x) throw() __attribute((const)); 
# 185
extern double floor(double __x) throw() __attribute((const)); extern double __floor(double __x) throw() __attribute((const)); 
# 188
extern double fmod(double __x, double __y) throw(); extern double __fmod(double __x, double __y) throw(); 
# 193
extern int __isinf(double __value) throw() __attribute((const)); 
# 196
extern int __finite(double __value) throw() __attribute((const)); 
# 202
extern inline int isinf(double __value) throw() __attribute((const)); 
# 205
extern int finite(double __value) throw() __attribute((const)); 
# 208
extern double drem(double __x, double __y) throw(); extern double __drem(double __x, double __y) throw(); 
# 212
extern double significand(double __x) throw(); extern double __significand(double __x) throw(); 
# 218
extern double copysign(double __x, double __y) throw() __attribute((const)); extern double __copysign(double __x, double __y) throw() __attribute((const)); 
# 225
extern double nan(const char * __tagb) throw() __attribute((const)); extern double __nan(const char * __tagb) throw() __attribute((const)); 
# 231
extern int __isnan(double __value) throw() __attribute((const)); 
# 235
extern inline int isnan(double __value) throw() __attribute((const)); 
# 238
extern double j0(double) throw(); extern double __j0(double) throw(); 
# 239
extern double j1(double) throw(); extern double __j1(double) throw(); 
# 240
extern double jn(int, double) throw(); extern double __jn(int, double) throw(); 
# 241
extern double y0(double) throw(); extern double __y0(double) throw(); 
# 242
extern double y1(double) throw(); extern double __y1(double) throw(); 
# 243
extern double yn(int, double) throw(); extern double __yn(int, double) throw(); 
# 250
extern double erf(double) throw(); extern double __erf(double) throw(); 
# 251
extern double erfc(double) throw(); extern double __erfc(double) throw(); 
# 252
extern double lgamma(double) throw(); extern double __lgamma(double) throw(); 
# 259
extern double tgamma(double) throw(); extern double __tgamma(double) throw(); 
# 265
extern double gamma(double) throw(); extern double __gamma(double) throw(); 
# 272
extern double lgamma_r(double, int * __signgamp) throw(); extern double __lgamma_r(double, int * __signgamp) throw(); 
# 280
extern double rint(double __x) throw(); extern double __rint(double __x) throw(); 
# 283
extern double nextafter(double __x, double __y) throw() __attribute((const)); extern double __nextafter(double __x, double __y) throw() __attribute((const)); 
# 285
extern double nexttoward(double __x, long double __y) throw() __attribute((const)); extern double __nexttoward(double __x, long double __y) throw() __attribute((const)); 
# 289
extern double remainder(double __x, double __y) throw(); extern double __remainder(double __x, double __y) throw(); 
# 293
extern double scalbn(double __x, int __n) throw(); extern double __scalbn(double __x, int __n) throw(); 
# 297
extern int ilogb(double __x) throw(); extern int __ilogb(double __x) throw(); 
# 302
extern double scalbln(double __x, long __n) throw(); extern double __scalbln(double __x, long __n) throw(); 
# 306
extern double nearbyint(double __x) throw(); extern double __nearbyint(double __x) throw(); 
# 310
extern double round(double __x) throw() __attribute((const)); extern double __round(double __x) throw() __attribute((const)); 
# 314
extern double trunc(double __x) throw() __attribute((const)); extern double __trunc(double __x) throw() __attribute((const)); 
# 319
extern double remquo(double __x, double __y, int * __quo) throw(); extern double __remquo(double __x, double __y, int * __quo) throw(); 
# 326
extern long lrint(double __x) throw(); extern long __lrint(double __x) throw(); 
# 327
extern long long llrint(double __x) throw(); extern long long __llrint(double __x) throw(); 
# 331
extern long lround(double __x) throw(); extern long __lround(double __x) throw(); 
# 332
extern long long llround(double __x) throw(); extern long long __llround(double __x) throw(); 
# 336
extern double fdim(double __x, double __y) throw(); extern double __fdim(double __x, double __y) throw(); 
# 339
extern double fmax(double __x, double __y) throw() __attribute((const)); extern double __fmax(double __x, double __y) throw() __attribute((const)); 
# 342
extern double fmin(double __x, double __y) throw() __attribute((const)); extern double __fmin(double __x, double __y) throw() __attribute((const)); 
# 346
extern int __fpclassify(double __value) throw()
# 347
 __attribute((const)); 
# 350
extern __attribute((gnu_inline)) inline int __signbit(double __value) throw()
# 351
 __attribute((const)); 
# 355
extern double fma(double __x, double __y, double __z) throw(); extern double __fma(double __x, double __y, double __z) throw(); 
# 364
extern double scalb(double __x, double __n) throw(); extern double __scalb(double __x, double __n) throw(); 
# 54 "/usr/include/bits/mathcalls.h" 3
extern float acosf(float __x) throw(); extern float __acosf(float __x) throw(); 
# 56
extern float asinf(float __x) throw(); extern float __asinf(float __x) throw(); 
# 58
extern float atanf(float __x) throw(); extern float __atanf(float __x) throw(); 
# 60
extern float atan2f(float __y, float __x) throw(); extern float __atan2f(float __y, float __x) throw(); 
# 63
extern float cosf(float __x) throw(); 
# 65
extern float sinf(float __x) throw(); 
# 67
extern float tanf(float __x) throw(); 
# 72
extern float coshf(float __x) throw(); extern float __coshf(float __x) throw(); 
# 74
extern float sinhf(float __x) throw(); extern float __sinhf(float __x) throw(); 
# 76
extern float tanhf(float __x) throw(); extern float __tanhf(float __x) throw(); 
# 81
extern void sincosf(float __x, float * __sinx, float * __cosx) throw(); 
# 88
extern float acoshf(float __x) throw(); extern float __acoshf(float __x) throw(); 
# 90
extern float asinhf(float __x) throw(); extern float __asinhf(float __x) throw(); 
# 92
extern float atanhf(float __x) throw(); extern float __atanhf(float __x) throw(); 
# 100
extern float expf(float __x) throw(); 
# 103
extern float frexpf(float __x, int * __exponent) throw(); extern float __frexpf(float __x, int * __exponent) throw(); 
# 106
extern float ldexpf(float __x, int __exponent) throw(); extern float __ldexpf(float __x, int __exponent) throw(); 
# 109
extern float logf(float __x) throw(); 
# 112
extern float log10f(float __x) throw(); 
# 115
extern float modff(float __x, float * __iptr) throw(); extern float __modff(float __x, float * __iptr) throw()
# 116
 __attribute((__nonnull__(2))); 
# 121
extern float exp10f(float __x) throw(); 
# 123
extern float pow10f(float __x) throw(); extern float __pow10f(float __x) throw(); 
# 129
extern float expm1f(float __x) throw(); extern float __expm1f(float __x) throw(); 
# 132
extern float log1pf(float __x) throw(); extern float __log1pf(float __x) throw(); 
# 135
extern float logbf(float __x) throw(); extern float __logbf(float __x) throw(); 
# 142
extern float exp2f(float __x) throw(); extern float __exp2f(float __x) throw(); 
# 145
extern float log2f(float __x) throw(); 
# 154
extern float powf(float __x, float __y) throw(); 
# 157
extern float sqrtf(float __x) throw(); extern float __sqrtf(float __x) throw(); 
# 163
extern float hypotf(float __x, float __y) throw(); extern float __hypotf(float __x, float __y) throw(); 
# 170
extern float cbrtf(float __x) throw(); extern float __cbrtf(float __x) throw(); 
# 179
extern float ceilf(float __x) throw() __attribute((const)); extern float __ceilf(float __x) throw() __attribute((const)); 
# 182
extern float fabsf(float __x) throw() __attribute((const)); extern float __fabsf(float __x) throw() __attribute((const)); 
# 185
extern float floorf(float __x) throw() __attribute((const)); extern float __floorf(float __x) throw() __attribute((const)); 
# 188
extern float fmodf(float __x, float __y) throw(); extern float __fmodf(float __x, float __y) throw(); 
# 193
extern int __isinff(float __value) throw() __attribute((const)); 
# 196
extern int __finitef(float __value) throw() __attribute((const)); 
# 202
extern int isinff(float __value) throw() __attribute((const)); 
# 205
extern int finitef(float __value) throw() __attribute((const)); 
# 208
extern float dremf(float __x, float __y) throw(); extern float __dremf(float __x, float __y) throw(); 
# 212
extern float significandf(float __x) throw(); extern float __significandf(float __x) throw(); 
# 218
extern float copysignf(float __x, float __y) throw() __attribute((const)); extern float __copysignf(float __x, float __y) throw() __attribute((const)); 
# 225
extern float nanf(const char * __tagb) throw() __attribute((const)); extern float __nanf(const char * __tagb) throw() __attribute((const)); 
# 231
extern int __isnanf(float __value) throw() __attribute((const)); 
# 235
extern int isnanf(float __value) throw() __attribute((const)); 
# 238
extern float j0f(float) throw(); extern float __j0f(float) throw(); 
# 239
extern float j1f(float) throw(); extern float __j1f(float) throw(); 
# 240
extern float jnf(int, float) throw(); extern float __jnf(int, float) throw(); 
# 241
extern float y0f(float) throw(); extern float __y0f(float) throw(); 
# 242
extern float y1f(float) throw(); extern float __y1f(float) throw(); 
# 243
extern float ynf(int, float) throw(); extern float __ynf(int, float) throw(); 
# 250
extern float erff(float) throw(); extern float __erff(float) throw(); 
# 251
extern float erfcf(float) throw(); extern float __erfcf(float) throw(); 
# 252
extern float lgammaf(float) throw(); extern float __lgammaf(float) throw(); 
# 259
extern float tgammaf(float) throw(); extern float __tgammaf(float) throw(); 
# 265
extern float gammaf(float) throw(); extern float __gammaf(float) throw(); 
# 272
extern float lgammaf_r(float, int * __signgamp) throw(); extern float __lgammaf_r(float, int * __signgamp) throw(); 
# 280
extern float rintf(float __x) throw(); extern float __rintf(float __x) throw(); 
# 283
extern float nextafterf(float __x, float __y) throw() __attribute((const)); extern float __nextafterf(float __x, float __y) throw() __attribute((const)); 
# 285
extern float nexttowardf(float __x, long double __y) throw() __attribute((const)); extern float __nexttowardf(float __x, long double __y) throw() __attribute((const)); 
# 289
extern float remainderf(float __x, float __y) throw(); extern float __remainderf(float __x, float __y) throw(); 
# 293
extern float scalbnf(float __x, int __n) throw(); extern float __scalbnf(float __x, int __n) throw(); 
# 297
extern int ilogbf(float __x) throw(); extern int __ilogbf(float __x) throw(); 
# 302
extern float scalblnf(float __x, long __n) throw(); extern float __scalblnf(float __x, long __n) throw(); 
# 306
extern float nearbyintf(float __x) throw(); extern float __nearbyintf(float __x) throw(); 
# 310
extern float roundf(float __x) throw() __attribute((const)); extern float __roundf(float __x) throw() __attribute((const)); 
# 314
extern float truncf(float __x) throw() __attribute((const)); extern float __truncf(float __x) throw() __attribute((const)); 
# 319
extern float remquof(float __x, float __y, int * __quo) throw(); extern float __remquof(float __x, float __y, int * __quo) throw(); 
# 326
extern long lrintf(float __x) throw(); extern long __lrintf(float __x) throw(); 
# 327
extern long long llrintf(float __x) throw(); extern long long __llrintf(float __x) throw(); 
# 331
extern long lroundf(float __x) throw(); extern long __lroundf(float __x) throw(); 
# 332
extern long long llroundf(float __x) throw(); extern long long __llroundf(float __x) throw(); 
# 336
extern float fdimf(float __x, float __y) throw(); extern float __fdimf(float __x, float __y) throw(); 
# 339
extern float fmaxf(float __x, float __y) throw() __attribute((const)); extern float __fmaxf(float __x, float __y) throw() __attribute((const)); 
# 342
extern float fminf(float __x, float __y) throw() __attribute((const)); extern float __fminf(float __x, float __y) throw() __attribute((const)); 
# 346
extern int __fpclassifyf(float __value) throw()
# 347
 __attribute((const)); 
# 350
extern __attribute((gnu_inline)) inline int __signbitf(float __value) throw()
# 351
 __attribute((const)); 
# 355
extern float fmaf(float __x, float __y, float __z) throw(); extern float __fmaf(float __x, float __y, float __z) throw(); 
# 364
extern float scalbf(float __x, float __n) throw(); extern float __scalbf(float __x, float __n) throw(); 
# 54 "/usr/include/bits/mathcalls.h" 3
extern long double acosl(long double __x) throw(); extern long double __acosl(long double __x) throw(); 
# 56
extern long double asinl(long double __x) throw(); extern long double __asinl(long double __x) throw(); 
# 58
extern long double atanl(long double __x) throw(); extern long double __atanl(long double __x) throw(); 
# 60
extern long double atan2l(long double __y, long double __x) throw(); extern long double __atan2l(long double __y, long double __x) throw(); 
# 63
extern long double cosl(long double __x) throw(); extern long double __cosl(long double __x) throw(); 
# 65
extern long double sinl(long double __x) throw(); extern long double __sinl(long double __x) throw(); 
# 67
extern long double tanl(long double __x) throw(); extern long double __tanl(long double __x) throw(); 
# 72
extern long double coshl(long double __x) throw(); extern long double __coshl(long double __x) throw(); 
# 74
extern long double sinhl(long double __x) throw(); extern long double __sinhl(long double __x) throw(); 
# 76
extern long double tanhl(long double __x) throw(); extern long double __tanhl(long double __x) throw(); 
# 81
extern void sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); extern void __sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); 
# 88
extern long double acoshl(long double __x) throw(); extern long double __acoshl(long double __x) throw(); 
# 90
extern long double asinhl(long double __x) throw(); extern long double __asinhl(long double __x) throw(); 
# 92
extern long double atanhl(long double __x) throw(); extern long double __atanhl(long double __x) throw(); 
# 100
extern long double expl(long double __x) throw(); extern long double __expl(long double __x) throw(); 
# 103
extern long double frexpl(long double __x, int * __exponent) throw(); extern long double __frexpl(long double __x, int * __exponent) throw(); 
# 106
extern long double ldexpl(long double __x, int __exponent) throw(); extern long double __ldexpl(long double __x, int __exponent) throw(); 
# 109
extern long double logl(long double __x) throw(); extern long double __logl(long double __x) throw(); 
# 112
extern long double log10l(long double __x) throw(); extern long double __log10l(long double __x) throw(); 
# 115
extern long double modfl(long double __x, long double * __iptr) throw(); extern long double __modfl(long double __x, long double * __iptr) throw()
# 116
 __attribute((__nonnull__(2))); 
# 121
extern long double exp10l(long double __x) throw(); extern long double __exp10l(long double __x) throw(); 
# 123
extern long double pow10l(long double __x) throw(); extern long double __pow10l(long double __x) throw(); 
# 129
extern long double expm1l(long double __x) throw(); extern long double __expm1l(long double __x) throw(); 
# 132
extern long double log1pl(long double __x) throw(); extern long double __log1pl(long double __x) throw(); 
# 135
extern long double logbl(long double __x) throw(); extern long double __logbl(long double __x) throw(); 
# 142
extern long double exp2l(long double __x) throw(); extern long double __exp2l(long double __x) throw(); 
# 145
extern long double log2l(long double __x) throw(); extern long double __log2l(long double __x) throw(); 
# 154
extern long double powl(long double __x, long double __y) throw(); extern long double __powl(long double __x, long double __y) throw(); 
# 157
extern long double sqrtl(long double __x) throw(); extern long double __sqrtl(long double __x) throw(); 
# 163
extern long double hypotl(long double __x, long double __y) throw(); extern long double __hypotl(long double __x, long double __y) throw(); 
# 170
extern long double cbrtl(long double __x) throw(); extern long double __cbrtl(long double __x) throw(); 
# 179
extern long double ceill(long double __x) throw() __attribute((const)); extern long double __ceill(long double __x) throw() __attribute((const)); 
# 182
extern long double fabsl(long double __x) throw() __attribute((const)); extern long double __fabsl(long double __x) throw() __attribute((const)); 
# 185
extern long double floorl(long double __x) throw() __attribute((const)); extern long double __floorl(long double __x) throw() __attribute((const)); 
# 188
extern long double fmodl(long double __x, long double __y) throw(); extern long double __fmodl(long double __x, long double __y) throw(); 
# 193
extern int __isinfl(long double __value) throw() __attribute((const)); 
# 196
extern int __finitel(long double __value) throw() __attribute((const)); 
# 202
extern int isinfl(long double __value) throw() __attribute((const)); 
# 205
extern int finitel(long double __value) throw() __attribute((const)); 
# 208
extern long double dreml(long double __x, long double __y) throw(); extern long double __dreml(long double __x, long double __y) throw(); 
# 212
extern long double significandl(long double __x) throw(); extern long double __significandl(long double __x) throw(); 
# 218
extern long double copysignl(long double __x, long double __y) throw() __attribute((const)); extern long double __copysignl(long double __x, long double __y) throw() __attribute((const)); 
# 225
extern long double nanl(const char * __tagb) throw() __attribute((const)); extern long double __nanl(const char * __tagb) throw() __attribute((const)); 
# 231
extern int __isnanl(long double __value) throw() __attribute((const)); 
# 235
extern int isnanl(long double __value) throw() __attribute((const)); 
# 238
extern long double j0l(long double) throw(); extern long double __j0l(long double) throw(); 
# 239
extern long double j1l(long double) throw(); extern long double __j1l(long double) throw(); 
# 240
extern long double jnl(int, long double) throw(); extern long double __jnl(int, long double) throw(); 
# 241
extern long double y0l(long double) throw(); extern long double __y0l(long double) throw(); 
# 242
extern long double y1l(long double) throw(); extern long double __y1l(long double) throw(); 
# 243
extern long double ynl(int, long double) throw(); extern long double __ynl(int, long double) throw(); 
# 250
extern long double erfl(long double) throw(); extern long double __erfl(long double) throw(); 
# 251
extern long double erfcl(long double) throw(); extern long double __erfcl(long double) throw(); 
# 252
extern long double lgammal(long double) throw(); extern long double __lgammal(long double) throw(); 
# 259
extern long double tgammal(long double) throw(); extern long double __tgammal(long double) throw(); 
# 265
extern long double gammal(long double) throw(); extern long double __gammal(long double) throw(); 
# 272
extern long double lgammal_r(long double, int * __signgamp) throw(); extern long double __lgammal_r(long double, int * __signgamp) throw(); 
# 280
extern long double rintl(long double __x) throw(); extern long double __rintl(long double __x) throw(); 
# 283
extern long double nextafterl(long double __x, long double __y) throw() __attribute((const)); extern long double __nextafterl(long double __x, long double __y) throw() __attribute((const)); 
# 285
extern long double nexttowardl(long double __x, long double __y) throw() __attribute((const)); extern long double __nexttowardl(long double __x, long double __y) throw() __attribute((const)); 
# 289
extern long double remainderl(long double __x, long double __y) throw(); extern long double __remainderl(long double __x, long double __y) throw(); 
# 293
extern long double scalbnl(long double __x, int __n) throw(); extern long double __scalbnl(long double __x, int __n) throw(); 
# 297
extern int ilogbl(long double __x) throw(); extern int __ilogbl(long double __x) throw(); 
# 302
extern long double scalblnl(long double __x, long __n) throw(); extern long double __scalblnl(long double __x, long __n) throw(); 
# 306
extern long double nearbyintl(long double __x) throw(); extern long double __nearbyintl(long double __x) throw(); 
# 310
extern long double roundl(long double __x) throw() __attribute((const)); extern long double __roundl(long double __x) throw() __attribute((const)); 
# 314
extern long double truncl(long double __x) throw() __attribute((const)); extern long double __truncl(long double __x) throw() __attribute((const)); 
# 319
extern long double remquol(long double __x, long double __y, int * __quo) throw(); extern long double __remquol(long double __x, long double __y, int * __quo) throw(); 
# 326
extern long lrintl(long double __x) throw(); extern long __lrintl(long double __x) throw(); 
# 327
extern long long llrintl(long double __x) throw(); extern long long __llrintl(long double __x) throw(); 
# 331
extern long lroundl(long double __x) throw(); extern long __lroundl(long double __x) throw(); 
# 332
extern long long llroundl(long double __x) throw(); extern long long __llroundl(long double __x) throw(); 
# 336
extern long double fdiml(long double __x, long double __y) throw(); extern long double __fdiml(long double __x, long double __y) throw(); 
# 339
extern long double fmaxl(long double __x, long double __y) throw() __attribute((const)); extern long double __fmaxl(long double __x, long double __y) throw() __attribute((const)); 
# 342
extern long double fminl(long double __x, long double __y) throw() __attribute((const)); extern long double __fminl(long double __x, long double __y) throw() __attribute((const)); 
# 346
extern int __fpclassifyl(long double __value) throw()
# 347
 __attribute((const)); 
# 350
extern __attribute((gnu_inline)) inline int __signbitl(long double __value) throw()
# 351
 __attribute((const)); 
# 355
extern long double fmal(long double __x, long double __y, long double __z) throw(); extern long double __fmal(long double __x, long double __y, long double __z) throw(); 
# 364
extern long double scalbl(long double __x, long double __n) throw(); extern long double __scalbl(long double __x, long double __n) throw(); 
# 149 "/usr/include/math.h" 3
extern int signgam; 
# 191 "/usr/include/math.h" 3
enum { 
# 192
FP_NAN, 
# 195
FP_INFINITE, 
# 198
FP_ZERO, 
# 201
FP_SUBNORMAL, 
# 204
FP_NORMAL
# 207
}; 
# 295 "/usr/include/math.h" 3
typedef 
# 289
enum { 
# 290
_IEEE_ = (-1), 
# 291
_SVID_ = 0, 
# 292
_XOPEN_, 
# 293
_POSIX_, 
# 294
_ISOC_
# 295
} _LIB_VERSION_TYPE; 
# 300
extern _LIB_VERSION_TYPE _LIB_VERSION; 
# 311 "/usr/include/math.h" 3
struct __exception { 
# 316
int type; 
# 317
char *name; 
# 318
double arg1; 
# 319
double arg2; 
# 320
double retval; 
# 321
}; 
# 324
extern int matherr(__exception * __exc) throw(); 
# 126 "/usr/include/bits/mathinline.h" 3
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline int
# 127
 __attribute((__leaf__)) __signbitf(float __x) throw() 
# 128
{ 
# 130
int __m; 
# 131
__asm__("pmovmskb %1, %0" : "=r" (__m) : "x" (__x)); 
# 132
return (__m & 8) != 0; 
# 137
} 
# 138
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline int
# 139
 __attribute((__leaf__)) __signbit(double __x) throw() 
# 140
{ 
# 142
int __m; 
# 143
__asm__("pmovmskb %1, %0" : "=r" (__m) : "x" (__x)); 
# 144
return (__m & 128) != 0; 
# 149
} 
# 150
__attribute((__always_inline__)) __attribute((__gnu_inline__)) extern inline int
# 151
 __attribute((__leaf__)) __signbitl(long double __x) throw() 
# 152
{ 
# 153
__extension__ union { long double __l; int __i[3]; } __u = {__l: __x}; 
# 154
return (((__u.__i)[2]) & 32768) != 0; 
# 155
} 
# 475 "/usr/include/math.h" 3
}
# 34 "/usr/include/stdlib.h" 3
extern "C" {
# 45 "/usr/include/bits/byteswap.h" 3
static inline unsigned __bswap_32(unsigned __bsx) 
# 46
{ 
# 47
return __builtin_bswap32(__bsx); 
# 48
} 
# 109 "/usr/include/bits/byteswap.h" 3
static inline __uint64_t __bswap_64(__uint64_t __bsx) 
# 110
{ 
# 111
return __builtin_bswap64(__bsx); 
# 112
} 
# 66 "/usr/include/bits/waitstatus.h" 3
union wait { 
# 68
int w_status; 
# 70
struct { 
# 72
unsigned __w_termsig:7; 
# 73
unsigned __w_coredump:1; 
# 74
unsigned __w_retcode:8; 
# 75
unsigned:16; 
# 83
} __wait_terminated; 
# 85
struct { 
# 87
unsigned __w_stopval:8; 
# 88
unsigned __w_stopsig:8; 
# 89
unsigned:16; 
# 96
} __wait_stopped; 
# 97
}; 
# 101 "/usr/include/stdlib.h" 3
typedef 
# 98
struct { 
# 99
int quot; 
# 100
int rem; 
# 101
} div_t; 
# 109
typedef 
# 106
struct { 
# 107
long quot; 
# 108
long rem; 
# 109
} ldiv_t; 
# 121
__extension__ typedef 
# 118
struct { 
# 119
long long quot; 
# 120
long long rem; 
# 121
} lldiv_t; 
# 139 "/usr/include/stdlib.h" 3
extern size_t __ctype_get_mb_cur_max() throw(); 
# 144
extern __attribute((gnu_inline)) inline double atof(const char * __nptr) throw()
# 145
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 147
extern __attribute((gnu_inline)) inline int atoi(const char * __nptr) throw()
# 148
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 150
extern __attribute((gnu_inline)) inline long atol(const char * __nptr) throw()
# 151
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 157
__extension__ extern __attribute((gnu_inline)) inline long long atoll(const char * __nptr) throw()
# 158
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 164
extern double strtod(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 166
 __attribute((__nonnull__(1))); 
# 172
extern float strtof(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 173
 __attribute((__nonnull__(1))); 
# 175
extern long double strtold(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 177
 __attribute((__nonnull__(1))); 
# 183
extern long strtol(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 185
 __attribute((__nonnull__(1))); 
# 187
extern unsigned long strtoul(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 189
 __attribute((__nonnull__(1))); 
# 195
__extension__ extern long long strtoq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 197
 __attribute((__nonnull__(1))); 
# 200
__extension__ extern unsigned long long strtouq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 202
 __attribute((__nonnull__(1))); 
# 209
__extension__ extern long long strtoll(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 211
 __attribute((__nonnull__(1))); 
# 214
__extension__ extern unsigned long long strtoull(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 216
 __attribute((__nonnull__(1))); 
# 239 "/usr/include/stdlib.h" 3
extern long strtol_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 241
 __attribute((__nonnull__(1, 4))); 
# 243
extern unsigned long strtoul_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 246
 __attribute((__nonnull__(1, 4))); 
# 249
__extension__ extern long long strtoll_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 252
 __attribute((__nonnull__(1, 4))); 
# 255
__extension__ extern unsigned long long strtoull_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 258
 __attribute((__nonnull__(1, 4))); 
# 260
extern double strtod_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 262
 __attribute((__nonnull__(1, 3))); 
# 264
extern float strtof_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 266
 __attribute((__nonnull__(1, 3))); 
# 268
extern long double strtold_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 271
 __attribute((__nonnull__(1, 3))); 
# 277
__attribute((__gnu_inline__)) extern inline int
# 278
 __attribute((__leaf__)) atoi(const char *__nptr) throw() 
# 279
{ 
# 280
return (int)strtol(__nptr, (char **)__null, 10); 
# 281
} 
# 282
__attribute((__gnu_inline__)) extern inline long
# 283
 __attribute((__leaf__)) atol(const char *__nptr) throw() 
# 284
{ 
# 285
return strtol(__nptr, (char **)__null, 10); 
# 286
} 
# 292
__extension__ 
# 291
__attribute((__gnu_inline__)) extern inline long long
# 292
 __attribute((__leaf__)) atoll(const char *__nptr) throw() 
# 293
{ 
# 294
return strtoll(__nptr, (char **)__null, 10); 
# 295
} 
# 305 "/usr/include/stdlib.h" 3
extern char *l64a(long __n) throw(); 
# 308
extern long a64l(const char * __s) throw()
# 309
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 27 "/usr/include/sys/types.h" 3
extern "C" {
# 33
typedef __u_char u_char; 
# 34
typedef __u_short u_short; 
# 35
typedef __u_int u_int; 
# 36
typedef __u_long u_long; 
# 37
typedef __quad_t quad_t; 
# 38
typedef __u_quad_t u_quad_t; 
# 39
typedef __fsid_t fsid_t; 
# 44
typedef __loff_t loff_t; 
# 48
typedef __ino_t ino_t; 
# 55
typedef __ino64_t ino64_t; 
# 60
typedef __dev_t dev_t; 
# 65
typedef __gid_t gid_t; 
# 70
typedef __mode_t mode_t; 
# 75
typedef __nlink_t nlink_t; 
# 80
typedef __uid_t uid_t; 
# 86
typedef __off_t off_t; 
# 93
typedef __off64_t off64_t; 
# 104 "/usr/include/sys/types.h" 3
typedef __id_t id_t; 
# 109
typedef __ssize_t ssize_t; 
# 115
typedef __daddr_t daddr_t; 
# 116
typedef __caddr_t caddr_t; 
# 122
typedef __key_t key_t; 
# 136 "/usr/include/sys/types.h" 3
typedef __useconds_t useconds_t; 
# 140
typedef __suseconds_t suseconds_t; 
# 150 "/usr/include/sys/types.h" 3
typedef unsigned long ulong; 
# 151
typedef unsigned short ushort; 
# 152
typedef unsigned uint; 
# 194 "/usr/include/sys/types.h" 3
typedef signed char int8_t __attribute((__mode__(__QI__))); 
# 195
typedef short int16_t __attribute((__mode__(__HI__))); 
# 196
typedef int int32_t __attribute((__mode__(__SI__))); 
# 197
typedef long int64_t __attribute((__mode__(__DI__))); 
# 200
typedef unsigned char u_int8_t __attribute((__mode__(__QI__))); 
# 201
typedef unsigned short u_int16_t __attribute((__mode__(__HI__))); 
# 202
typedef unsigned u_int32_t __attribute((__mode__(__SI__))); 
# 203
typedef unsigned long u_int64_t __attribute((__mode__(__DI__))); 
# 205
typedef long register_t __attribute((__mode__(__word__))); 
# 23 "/usr/include/bits/sigset.h" 3
typedef int __sig_atomic_t; 
# 31
typedef 
# 29
struct { 
# 30
unsigned long __val[(1024) / ((8) * sizeof(unsigned long))]; 
# 31
} __sigset_t; 
# 37 "/usr/include/sys/select.h" 3
typedef __sigset_t sigset_t; 
# 54 "/usr/include/sys/select.h" 3
typedef long __fd_mask; 
# 75 "/usr/include/sys/select.h" 3
typedef 
# 65
struct { 
# 69
__fd_mask fds_bits[1024 / (8 * ((int)sizeof(__fd_mask)))]; 
# 75
} fd_set; 
# 82
typedef __fd_mask fd_mask; 
# 96 "/usr/include/sys/select.h" 3
extern "C" {
# 106 "/usr/include/sys/select.h" 3
extern int select(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, timeval *__restrict__ __timeout); 
# 118 "/usr/include/sys/select.h" 3
extern int pselect(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, const timespec *__restrict__ __timeout, const __sigset_t *__restrict__ __sigmask); 
# 131 "/usr/include/sys/select.h" 3
}
# 29 "/usr/include/sys/sysmacros.h" 3
extern "C" {
# 32
__extension__ extern __attribute((gnu_inline)) inline unsigned gnu_dev_major(unsigned long long __dev) throw()
# 33
 __attribute((const)); 
# 35
__extension__ extern __attribute((gnu_inline)) inline unsigned gnu_dev_minor(unsigned long long __dev) throw()
# 36
 __attribute((const)); 
# 38
__extension__ extern __attribute((gnu_inline)) inline unsigned long long gnu_dev_makedev(unsigned __major, unsigned __minor) throw()
# 40
 __attribute((const)); 
# 44
__extension__ 
# 43
__attribute((__gnu_inline__)) __attribute((const)) extern inline unsigned
# 44
 __attribute((__leaf__)) gnu_dev_major(unsigned long long __dev) throw() 
# 45
{ 
# 46
return ((__dev >> 8) & (4095)) | (((unsigned)(__dev >> 32)) & (~4095)); 
# 47
} 
# 50
__extension__ 
# 49
__attribute((__gnu_inline__)) __attribute((const)) extern inline unsigned
# 50
 __attribute((__leaf__)) gnu_dev_minor(unsigned long long __dev) throw() 
# 51
{ 
# 52
return (__dev & (255)) | (((unsigned)(__dev >> 12)) & (~255)); 
# 53
} 
# 56
__extension__ 
# 55
__attribute((__gnu_inline__)) __attribute((const)) extern inline unsigned long long
# 56
 __attribute((__leaf__)) gnu_dev_makedev(unsigned __major, unsigned __minor) throw() 
# 57
{ 
# 58
return (((__minor & (255)) | ((__major & (4095)) << 8)) | (((unsigned long long)(__minor & (~255))) << 12)) | (((unsigned long long)(__major & (~4095))) << 32); 
# 61
} 
# 63
}
# 228 "/usr/include/sys/types.h" 3
typedef __blksize_t blksize_t; 
# 235
typedef __blkcnt_t blkcnt_t; 
# 239
typedef __fsblkcnt_t fsblkcnt_t; 
# 243
typedef __fsfilcnt_t fsfilcnt_t; 
# 262 "/usr/include/sys/types.h" 3
typedef __blkcnt64_t blkcnt64_t; 
# 263
typedef __fsblkcnt64_t fsblkcnt64_t; 
# 264
typedef __fsfilcnt64_t fsfilcnt64_t; 
# 60 "/usr/include/bits/pthreadtypes.h" 3
typedef unsigned long pthread_t; 
# 63
union pthread_attr_t { 
# 65
char __size[56]; 
# 66
long __align; 
# 67
}; 
# 69
typedef pthread_attr_t pthread_attr_t; 
# 79
typedef 
# 75
struct __pthread_internal_list { 
# 77
__pthread_internal_list *__prev; 
# 78
__pthread_internal_list *__next; 
# 79
} __pthread_list_t; 
# 128 "/usr/include/bits/pthreadtypes.h" 3
typedef 
# 91 "/usr/include/bits/pthreadtypes.h" 3
union { 
# 92
struct __pthread_mutex_s { 
# 94
int __lock; 
# 95
unsigned __count; 
# 96
int __owner; 
# 98
unsigned __nusers; 
# 102
int __kind; 
# 104
short __spins; 
# 105
short __elision; 
# 106
__pthread_list_t __list; 
# 125 "/usr/include/bits/pthreadtypes.h" 3
} __data; 
# 126
char __size[40]; 
# 127
long __align; 
# 128
} pthread_mutex_t; 
# 134
typedef 
# 131
union { 
# 132
char __size[4]; 
# 133
int __align; 
# 134
} pthread_mutexattr_t; 
# 154
typedef 
# 140
union { 
# 142
struct { 
# 143
int __lock; 
# 144
unsigned __futex; 
# 145
__extension__ unsigned long long __total_seq; 
# 146
__extension__ unsigned long long __wakeup_seq; 
# 147
__extension__ unsigned long long __woken_seq; 
# 148
void *__mutex; 
# 149
unsigned __nwaiters; 
# 150
unsigned __broadcast_seq; 
# 151
} __data; 
# 152
char __size[48]; 
# 153
__extension__ long long __align; 
# 154
} pthread_cond_t; 
# 160
typedef 
# 157
union { 
# 158
char __size[4]; 
# 159
int __align; 
# 160
} pthread_condattr_t; 
# 164
typedef unsigned pthread_key_t; 
# 168
typedef int pthread_once_t; 
# 214 "/usr/include/bits/pthreadtypes.h" 3
typedef 
# 175 "/usr/include/bits/pthreadtypes.h" 3
union { 
# 178
struct { 
# 179
int __lock; 
# 180
unsigned __nr_readers; 
# 181
unsigned __readers_wakeup; 
# 182
unsigned __writer_wakeup; 
# 183
unsigned __nr_readers_queued; 
# 184
unsigned __nr_writers_queued; 
# 185
int __writer; 
# 186
int __shared; 
# 187
unsigned long __pad1; 
# 188
unsigned long __pad2; 
# 191
unsigned __flags; 
# 193
} __data; 
# 212 "/usr/include/bits/pthreadtypes.h" 3
char __size[56]; 
# 213
long __align; 
# 214
} pthread_rwlock_t; 
# 220
typedef 
# 217
union { 
# 218
char __size[8]; 
# 219
long __align; 
# 220
} pthread_rwlockattr_t; 
# 226
typedef volatile int pthread_spinlock_t; 
# 235
typedef 
# 232
union { 
# 233
char __size[32]; 
# 234
long __align; 
# 235
} pthread_barrier_t; 
# 241
typedef 
# 238
union { 
# 239
char __size[4]; 
# 240
int __align; 
# 241
} pthread_barrierattr_t; 
# 273 "/usr/include/sys/types.h" 3
}
# 321 "/usr/include/stdlib.h" 3
extern long random() throw(); 
# 324
extern void srandom(unsigned __seed) throw(); 
# 330
extern char *initstate(unsigned __seed, char * __statebuf, size_t __statelen) throw()
# 331
 __attribute((__nonnull__(2))); 
# 335
extern char *setstate(char * __statebuf) throw() __attribute((__nonnull__(1))); 
# 343
struct random_data { 
# 345
int32_t *fptr; 
# 346
int32_t *rptr; 
# 347
int32_t *state; 
# 348
int rand_type; 
# 349
int rand_deg; 
# 350
int rand_sep; 
# 351
int32_t *end_ptr; 
# 352
}; 
# 354
extern int random_r(random_data *__restrict__ __buf, int32_t *__restrict__ __result) throw()
# 355
 __attribute((__nonnull__(1, 2))); 
# 357
extern int srandom_r(unsigned __seed, random_data * __buf) throw()
# 358
 __attribute((__nonnull__(2))); 
# 360
extern int initstate_r(unsigned __seed, char *__restrict__ __statebuf, size_t __statelen, random_data *__restrict__ __buf) throw()
# 363
 __attribute((__nonnull__(2, 4))); 
# 365
extern int setstate_r(char *__restrict__ __statebuf, random_data *__restrict__ __buf) throw()
# 367
 __attribute((__nonnull__(1, 2))); 
# 374
extern int rand() throw(); 
# 376
extern void srand(unsigned __seed) throw(); 
# 381
extern int rand_r(unsigned * __seed) throw(); 
# 389
extern double drand48() throw(); 
# 390
extern double erand48(unsigned short  __xsubi[3]) throw() __attribute((__nonnull__(1))); 
# 393
extern long lrand48() throw(); 
# 394
extern long nrand48(unsigned short  __xsubi[3]) throw()
# 395
 __attribute((__nonnull__(1))); 
# 398
extern long mrand48() throw(); 
# 399
extern long jrand48(unsigned short  __xsubi[3]) throw()
# 400
 __attribute((__nonnull__(1))); 
# 403
extern void srand48(long __seedval) throw(); 
# 404
extern unsigned short *seed48(unsigned short  __seed16v[3]) throw()
# 405
 __attribute((__nonnull__(1))); 
# 406
extern void lcong48(unsigned short  __param[7]) throw() __attribute((__nonnull__(1))); 
# 412
struct drand48_data { 
# 414
unsigned short __x[3]; 
# 415
unsigned short __old_x[3]; 
# 416
unsigned short __c; 
# 417
unsigned short __init; 
# 418
unsigned long long __a; 
# 419
}; 
# 422
extern int drand48_r(drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 423
 __attribute((__nonnull__(1, 2))); 
# 424
extern int erand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 426
 __attribute((__nonnull__(1, 2))); 
# 429
extern int lrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 431
 __attribute((__nonnull__(1, 2))); 
# 432
extern int nrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 435
 __attribute((__nonnull__(1, 2))); 
# 438
extern int mrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 440
 __attribute((__nonnull__(1, 2))); 
# 441
extern int jrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 444
 __attribute((__nonnull__(1, 2))); 
# 447
extern int srand48_r(long __seedval, drand48_data * __buffer) throw()
# 448
 __attribute((__nonnull__(2))); 
# 450
extern int seed48_r(unsigned short  __seed16v[3], drand48_data * __buffer) throw()
# 451
 __attribute((__nonnull__(1, 2))); 
# 453
extern int lcong48_r(unsigned short  __param[7], drand48_data * __buffer) throw()
# 455
 __attribute((__nonnull__(1, 2))); 
# 465
extern void *malloc(size_t __size) throw() __attribute((__malloc__)); 
# 467
extern void *calloc(size_t __nmemb, size_t __size) throw()
# 468
 __attribute((__malloc__)); 
# 479
extern void *realloc(void * __ptr, size_t __size) throw()
# 480
 __attribute((__warn_unused_result__)); 
# 482
extern void free(void * __ptr) throw(); 
# 487
extern void cfree(void * __ptr) throw(); 
# 26 "/usr/include/alloca.h" 3
extern "C" {
# 32
extern void *alloca(size_t __size) throw(); 
# 38
}
# 497 "/usr/include/stdlib.h" 3
extern void *valloc(size_t __size) throw() __attribute((__malloc__)); 
# 502
extern int posix_memalign(void ** __memptr, size_t __alignment, size_t __size) throw()
# 503
 __attribute((__nonnull__(1))); 
# 508
extern void *aligned_alloc(size_t __alignment, size_t __size) throw()
# 509
 __attribute((__malloc__, __alloc_size__(2))); 
# 514
extern void abort() throw() __attribute((__noreturn__)); 
# 518
extern int atexit(void (* __func)(void)) throw() __attribute((__nonnull__(1))); 
# 523
extern "C++" int at_quick_exit(void (* __func)(void)) throw() __asm__("at_quick_exit")
# 524
 __attribute((__nonnull__(1))); 
# 534
extern int on_exit(void (* __func)(int __status, void * __arg), void * __arg) throw()
# 535
 __attribute((__nonnull__(1))); 
# 542
extern void exit(int __status) throw() __attribute((__noreturn__)); 
# 548
extern void quick_exit(int __status) throw() __attribute((__noreturn__)); 
# 556
extern void _Exit(int __status) throw() __attribute((__noreturn__)); 
# 563
extern char *getenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 569
extern char *secure_getenv(const char * __name) throw()
# 570
 __attribute((__nonnull__(1))); 
# 577
extern int putenv(char * __string) throw() __attribute((__nonnull__(1))); 
# 583
extern int setenv(const char * __name, const char * __value, int __replace) throw()
# 584
 __attribute((__nonnull__(2))); 
# 587
extern int unsetenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 594
extern int clearenv() throw(); 
# 605 "/usr/include/stdlib.h" 3
extern char *mktemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 619 "/usr/include/stdlib.h" 3
extern int mkstemp(char * __template) __attribute((__nonnull__(1))); 
# 629 "/usr/include/stdlib.h" 3
extern int mkstemp64(char * __template) __attribute((__nonnull__(1))); 
# 641 "/usr/include/stdlib.h" 3
extern int mkstemps(char * __template, int __suffixlen) __attribute((__nonnull__(1))); 
# 651 "/usr/include/stdlib.h" 3
extern int mkstemps64(char * __template, int __suffixlen)
# 652
 __attribute((__nonnull__(1))); 
# 662 "/usr/include/stdlib.h" 3
extern char *mkdtemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 673 "/usr/include/stdlib.h" 3
extern int mkostemp(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 683 "/usr/include/stdlib.h" 3
extern int mkostemp64(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 693 "/usr/include/stdlib.h" 3
extern int mkostemps(char * __template, int __suffixlen, int __flags)
# 694
 __attribute((__nonnull__(1))); 
# 705 "/usr/include/stdlib.h" 3
extern int mkostemps64(char * __template, int __suffixlen, int __flags)
# 706
 __attribute((__nonnull__(1))); 
# 716
extern int system(const char * __command); 
# 723
extern char *canonicalize_file_name(const char * __name) throw()
# 724
 __attribute((__nonnull__(1))); 
# 733 "/usr/include/stdlib.h" 3
extern char *realpath(const char *__restrict__ __name, char *__restrict__ __resolved) throw(); 
# 741
typedef int (*__compar_fn_t)(const void *, const void *); 
# 744
typedef __compar_fn_t comparison_fn_t; 
# 748
typedef int (*__compar_d_fn_t)(const void *, const void *, void *); 
# 754
extern void *bsearch(const void * __key, const void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 756
 __attribute((__nonnull__(1, 2, 5))); 
# 760
extern void qsort(void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 761
 __attribute((__nonnull__(1, 4))); 
# 763
extern void qsort_r(void * __base, size_t __nmemb, size_t __size, __compar_d_fn_t __compar, void * __arg)
# 765
 __attribute((__nonnull__(1, 4))); 
# 770
extern int abs(int __x) throw() __attribute((const)); 
# 771
extern long labs(long __x) throw() __attribute((const)); 
# 775
__extension__ extern long long llabs(long long __x) throw()
# 776
 __attribute((const)); 
# 784
extern div_t div(int __numer, int __denom) throw()
# 785
 __attribute((const)); 
# 786
extern ldiv_t ldiv(long __numer, long __denom) throw()
# 787
 __attribute((const)); 
# 792
__extension__ extern lldiv_t lldiv(long long __numer, long long __denom) throw()
# 794
 __attribute((const)); 
# 807 "/usr/include/stdlib.h" 3
extern char *ecvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 808
 __attribute((__nonnull__(3, 4))); 
# 813
extern char *fcvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 814
 __attribute((__nonnull__(3, 4))); 
# 819
extern char *gcvt(double __value, int __ndigit, char * __buf) throw()
# 820
 __attribute((__nonnull__(3))); 
# 825
extern char *qecvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 827
 __attribute((__nonnull__(3, 4))); 
# 828
extern char *qfcvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 830
 __attribute((__nonnull__(3, 4))); 
# 831
extern char *qgcvt(long double __value, int __ndigit, char * __buf) throw()
# 832
 __attribute((__nonnull__(3))); 
# 837
extern int ecvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 839
 __attribute((__nonnull__(3, 4, 5))); 
# 840
extern int fcvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 842
 __attribute((__nonnull__(3, 4, 5))); 
# 844
extern int qecvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 847
 __attribute((__nonnull__(3, 4, 5))); 
# 848
extern int qfcvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 851
 __attribute((__nonnull__(3, 4, 5))); 
# 859
extern int mblen(const char * __s, size_t __n) throw(); 
# 862
extern int mbtowc(wchar_t *__restrict__ __pwc, const char *__restrict__ __s, size_t __n) throw(); 
# 866
extern int wctomb(char * __s, wchar_t __wchar) throw(); 
# 870
extern size_t mbstowcs(wchar_t *__restrict__ __pwcs, const char *__restrict__ __s, size_t __n) throw(); 
# 873
extern size_t wcstombs(char *__restrict__ __s, const wchar_t *__restrict__ __pwcs, size_t __n) throw(); 
# 884
extern int rpmatch(const char * __response) throw() __attribute((__nonnull__(1))); 
# 895 "/usr/include/stdlib.h" 3
extern int getsubopt(char **__restrict__ __optionp, char *const *__restrict__ __tokens, char **__restrict__ __valuep) throw()
# 898
 __attribute((__nonnull__(1, 2, 3))); 
# 904
extern void setkey(const char * __key) throw() __attribute((__nonnull__(1))); 
# 912
extern int posix_openpt(int __oflag); 
# 920
extern int grantpt(int __fd) throw(); 
# 924
extern int unlockpt(int __fd) throw(); 
# 929
extern char *ptsname(int __fd) throw(); 
# 936
extern int ptsname_r(int __fd, char * __buf, size_t __buflen) throw()
# 937
 __attribute((__nonnull__(2))); 
# 940
extern int getpt(); 
# 947
extern int getloadavg(double  __loadavg[], int __nelem) throw()
# 948
 __attribute((__nonnull__(1))); 
# 25 "/usr/include/bits/stdlib-float.h" 3
__attribute((__gnu_inline__)) extern inline double
# 26
 __attribute((__leaf__)) atof(const char *__nptr) throw() 
# 27
{ 
# 28
return strtod(__nptr, (char **)__null); 
# 29
} 
# 964 "/usr/include/stdlib.h" 3
}
# 1855 "/usr/include/c++/4.8.2/x86_64-redhat-linux/bits/c++config.h" 3
namespace std { 
# 1857
typedef unsigned long size_t; 
# 1858
typedef long ptrdiff_t; 
# 1863
}
# 68 "/usr/include/c++/4.8.2/bits/cpp_type_traits.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 72
template< class _Iterator, class _Container> class __normal_iterator; 
# 76
}
# 78
namespace std __attribute((__visibility__("default"))) { 
# 82
struct __true_type { }; 
# 83
struct __false_type { }; 
# 85
template< bool > 
# 86
struct __truth_type { 
# 87
typedef __false_type __type; }; 
# 90
template<> struct __truth_type< true>  { 
# 91
typedef __true_type __type; }; 
# 95
template< class _Sp, class _Tp> 
# 96
struct __traitor { 
# 98
enum { __value = ((bool)_Sp::__value) || ((bool)_Tp::__value)}; 
# 99
typedef typename __truth_type< __value> ::__type __type; 
# 100
}; 
# 103
template< class , class > 
# 104
struct __are_same { 
# 106
enum { __value}; 
# 107
typedef __false_type __type; 
# 108
}; 
# 110
template< class _Tp> 
# 111
struct __are_same< _Tp, _Tp>  { 
# 113
enum { __value = 1}; 
# 114
typedef __true_type __type; 
# 115
}; 
# 118
template< class _Tp> 
# 119
struct __is_void { 
# 121
enum { __value}; 
# 122
typedef __false_type __type; 
# 123
}; 
# 126
template<> struct __is_void< void>  { 
# 128
enum { __value = 1}; 
# 129
typedef __true_type __type; 
# 130
}; 
# 135
template< class _Tp> 
# 136
struct __is_integer { 
# 138
enum { __value}; 
# 139
typedef __false_type __type; 
# 140
}; 
# 146
template<> struct __is_integer< bool>  { 
# 148
enum { __value = 1}; 
# 149
typedef __true_type __type; 
# 150
}; 
# 153
template<> struct __is_integer< char>  { 
# 155
enum { __value = 1}; 
# 156
typedef __true_type __type; 
# 157
}; 
# 160
template<> struct __is_integer< signed char>  { 
# 162
enum { __value = 1}; 
# 163
typedef __true_type __type; 
# 164
}; 
# 167
template<> struct __is_integer< unsigned char>  { 
# 169
enum { __value = 1}; 
# 170
typedef __true_type __type; 
# 171
}; 
# 175
template<> struct __is_integer< wchar_t>  { 
# 177
enum { __value = 1}; 
# 178
typedef __true_type __type; 
# 179
}; 
# 199 "/usr/include/c++/4.8.2/bits/cpp_type_traits.h" 3
template<> struct __is_integer< short>  { 
# 201
enum { __value = 1}; 
# 202
typedef __true_type __type; 
# 203
}; 
# 206
template<> struct __is_integer< unsigned short>  { 
# 208
enum { __value = 1}; 
# 209
typedef __true_type __type; 
# 210
}; 
# 213
template<> struct __is_integer< int>  { 
# 215
enum { __value = 1}; 
# 216
typedef __true_type __type; 
# 217
}; 
# 220
template<> struct __is_integer< unsigned>  { 
# 222
enum { __value = 1}; 
# 223
typedef __true_type __type; 
# 224
}; 
# 227
template<> struct __is_integer< long>  { 
# 229
enum { __value = 1}; 
# 230
typedef __true_type __type; 
# 231
}; 
# 234
template<> struct __is_integer< unsigned long>  { 
# 236
enum { __value = 1}; 
# 237
typedef __true_type __type; 
# 238
}; 
# 241
template<> struct __is_integer< long long>  { 
# 243
enum { __value = 1}; 
# 244
typedef __true_type __type; 
# 245
}; 
# 248
template<> struct __is_integer< unsigned long long>  { 
# 250
enum { __value = 1}; 
# 251
typedef __true_type __type; 
# 252
}; 
# 257
template< class _Tp> 
# 258
struct __is_floating { 
# 260
enum { __value}; 
# 261
typedef __false_type __type; 
# 262
}; 
# 266
template<> struct __is_floating< float>  { 
# 268
enum { __value = 1}; 
# 269
typedef __true_type __type; 
# 270
}; 
# 273
template<> struct __is_floating< double>  { 
# 275
enum { __value = 1}; 
# 276
typedef __true_type __type; 
# 277
}; 
# 280
template<> struct __is_floating< long double>  { 
# 282
enum { __value = 1}; 
# 283
typedef __true_type __type; 
# 284
}; 
# 289
template< class _Tp> 
# 290
struct __is_pointer { 
# 292
enum { __value}; 
# 293
typedef __false_type __type; 
# 294
}; 
# 296
template< class _Tp> 
# 297
struct __is_pointer< _Tp *>  { 
# 299
enum { __value = 1}; 
# 300
typedef __true_type __type; 
# 301
}; 
# 306
template< class _Tp> 
# 307
struct __is_normal_iterator { 
# 309
enum { __value}; 
# 310
typedef __false_type __type; 
# 311
}; 
# 313
template< class _Iterator, class _Container> 
# 314
struct __is_normal_iterator< __gnu_cxx::__normal_iterator< _Iterator, _Container> >  { 
# 317
enum { __value = 1}; 
# 318
typedef __true_type __type; 
# 319
}; 
# 324
template< class _Tp> 
# 325
struct __is_arithmetic : public __traitor< __is_integer< _Tp> , __is_floating< _Tp> >  { 
# 327
}; 
# 332
template< class _Tp> 
# 333
struct __is_fundamental : public __traitor< __is_void< _Tp> , __is_arithmetic< _Tp> >  { 
# 335
}; 
# 340
template< class _Tp> 
# 341
struct __is_scalar : public __traitor< __is_arithmetic< _Tp> , __is_pointer< _Tp> >  { 
# 343
}; 
# 348
template< class _Tp> 
# 349
struct __is_char { 
# 351
enum { __value}; 
# 352
typedef __false_type __type; 
# 353
}; 
# 356
template<> struct __is_char< char>  { 
# 358
enum { __value = 1}; 
# 359
typedef __true_type __type; 
# 360
}; 
# 364
template<> struct __is_char< wchar_t>  { 
# 366
enum { __value = 1}; 
# 367
typedef __true_type __type; 
# 368
}; 
# 371
template< class _Tp> 
# 372
struct __is_byte { 
# 374
enum { __value}; 
# 375
typedef __false_type __type; 
# 376
}; 
# 379
template<> struct __is_byte< char>  { 
# 381
enum { __value = 1}; 
# 382
typedef __true_type __type; 
# 383
}; 
# 386
template<> struct __is_byte< signed char>  { 
# 388
enum { __value = 1}; 
# 389
typedef __true_type __type; 
# 390
}; 
# 393
template<> struct __is_byte< unsigned char>  { 
# 395
enum { __value = 1}; 
# 396
typedef __true_type __type; 
# 397
}; 
# 402
template< class _Tp> 
# 403
struct __is_move_iterator { 
# 405
enum { __value}; 
# 406
typedef __false_type __type; 
# 407
}; 
# 422 "/usr/include/c++/4.8.2/bits/cpp_type_traits.h" 3
}
# 37 "/usr/include/c++/4.8.2/ext/type_traits.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 42
template< bool , class > 
# 43
struct __enable_if { 
# 44
}; 
# 46
template< class _Tp> 
# 47
struct __enable_if< true, _Tp>  { 
# 48
typedef _Tp __type; }; 
# 52
template< bool _Cond, class _Iftrue, class _Iffalse> 
# 53
struct __conditional_type { 
# 54
typedef _Iftrue __type; }; 
# 56
template< class _Iftrue, class _Iffalse> 
# 57
struct __conditional_type< false, _Iftrue, _Iffalse>  { 
# 58
typedef _Iffalse __type; }; 
# 62
template< class _Tp> 
# 63
struct __add_unsigned { 
# 66
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 69
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 70
}; 
# 73
template<> struct __add_unsigned< char>  { 
# 74
typedef unsigned char __type; }; 
# 77
template<> struct __add_unsigned< signed char>  { 
# 78
typedef unsigned char __type; }; 
# 81
template<> struct __add_unsigned< short>  { 
# 82
typedef unsigned short __type; }; 
# 85
template<> struct __add_unsigned< int>  { 
# 86
typedef unsigned __type; }; 
# 89
template<> struct __add_unsigned< long>  { 
# 90
typedef unsigned long __type; }; 
# 93
template<> struct __add_unsigned< long long>  { 
# 94
typedef unsigned long long __type; }; 
# 98
template<> struct __add_unsigned< bool> ; 
# 101
template<> struct __add_unsigned< wchar_t> ; 
# 105
template< class _Tp> 
# 106
struct __remove_unsigned { 
# 109
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 112
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 113
}; 
# 116
template<> struct __remove_unsigned< char>  { 
# 117
typedef signed char __type; }; 
# 120
template<> struct __remove_unsigned< unsigned char>  { 
# 121
typedef signed char __type; }; 
# 124
template<> struct __remove_unsigned< unsigned short>  { 
# 125
typedef short __type; }; 
# 128
template<> struct __remove_unsigned< unsigned>  { 
# 129
typedef int __type; }; 
# 132
template<> struct __remove_unsigned< unsigned long>  { 
# 133
typedef long __type; }; 
# 136
template<> struct __remove_unsigned< unsigned long long>  { 
# 137
typedef long long __type; }; 
# 141
template<> struct __remove_unsigned< bool> ; 
# 144
template<> struct __remove_unsigned< wchar_t> ; 
# 148
template< class _Type> inline bool 
# 150
__is_null_pointer(_Type *__ptr) 
# 151
{ return __ptr == 0; } 
# 153
template< class _Type> inline bool 
# 155
__is_null_pointer(_Type) 
# 156
{ return false; } 
# 160
template< class _Tp, bool  = std::__is_integer< _Tp> ::__value> 
# 161
struct __promote { 
# 162
typedef double __type; }; 
# 167
template< class _Tp> 
# 168
struct __promote< _Tp, false>  { 
# 169
}; 
# 172
template<> struct __promote< long double>  { 
# 173
typedef long double __type; }; 
# 176
template<> struct __promote< double>  { 
# 177
typedef double __type; }; 
# 180
template<> struct __promote< float>  { 
# 181
typedef float __type; }; 
# 183
template< class _Tp, class _Up, class 
# 184
_Tp2 = typename __promote< _Tp> ::__type, class 
# 185
_Up2 = typename __promote< _Up> ::__type> 
# 186
struct __promote_2 { 
# 188
typedef __typeof__(_Tp2() + _Up2()) __type; 
# 189
}; 
# 191
template< class _Tp, class _Up, class _Vp, class 
# 192
_Tp2 = typename __promote< _Tp> ::__type, class 
# 193
_Up2 = typename __promote< _Up> ::__type, class 
# 194
_Vp2 = typename __promote< _Vp> ::__type> 
# 195
struct __promote_3 { 
# 197
typedef __typeof__((_Tp2() + _Up2()) + _Vp2()) __type; 
# 198
}; 
# 200
template< class _Tp, class _Up, class _Vp, class _Wp, class 
# 201
_Tp2 = typename __promote< _Tp> ::__type, class 
# 202
_Up2 = typename __promote< _Up> ::__type, class 
# 203
_Vp2 = typename __promote< _Vp> ::__type, class 
# 204
_Wp2 = typename __promote< _Wp> ::__type> 
# 205
struct __promote_4 { 
# 207
typedef __typeof__(((_Tp2() + _Up2()) + _Vp2()) + _Wp2()) __type; 
# 208
}; 
# 211
}
# 75 "/usr/include/c++/4.8.2/cmath" 3
namespace std __attribute((__visibility__("default"))) { 
# 81
inline double abs(double __x) 
# 82
{ return __builtin_fabs(__x); } 
# 87
inline float abs(float __x) 
# 88
{ return __builtin_fabsf(__x); } 
# 91
inline long double abs(long double __x) 
# 92
{ return __builtin_fabsl(__x); } 
# 95
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 99
abs(_Tp __x) 
# 100
{ return __builtin_fabs(__x); } 
# 102
using ::acos;
# 106
inline float acos(float __x) 
# 107
{ return __builtin_acosf(__x); } 
# 110
inline long double acos(long double __x) 
# 111
{ return __builtin_acosl(__x); } 
# 114
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 118
acos(_Tp __x) 
# 119
{ return __builtin_acos(__x); } 
# 121
using ::asin;
# 125
inline float asin(float __x) 
# 126
{ return __builtin_asinf(__x); } 
# 129
inline long double asin(long double __x) 
# 130
{ return __builtin_asinl(__x); } 
# 133
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 137
asin(_Tp __x) 
# 138
{ return __builtin_asin(__x); } 
# 140
using ::atan;
# 144
inline float atan(float __x) 
# 145
{ return __builtin_atanf(__x); } 
# 148
inline long double atan(long double __x) 
# 149
{ return __builtin_atanl(__x); } 
# 152
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 156
atan(_Tp __x) 
# 157
{ return __builtin_atan(__x); } 
# 159
using ::atan2;
# 163
inline float atan2(float __y, float __x) 
# 164
{ return __builtin_atan2f(__y, __x); } 
# 167
inline long double atan2(long double __y, long double __x) 
# 168
{ return __builtin_atan2l(__y, __x); } 
# 171
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 174
atan2(_Tp __y, _Up __x) 
# 175
{ 
# 176
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 177
return atan2((__type)__y, (__type)__x); 
# 178
} 
# 180
using ::ceil;
# 184
inline float ceil(float __x) 
# 185
{ return __builtin_ceilf(__x); } 
# 188
inline long double ceil(long double __x) 
# 189
{ return __builtin_ceill(__x); } 
# 192
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 196
ceil(_Tp __x) 
# 197
{ return __builtin_ceil(__x); } 
# 199
using ::cos;
# 203
inline float cos(float __x) 
# 204
{ return __builtin_cosf(__x); } 
# 207
inline long double cos(long double __x) 
# 208
{ return __builtin_cosl(__x); } 
# 211
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 215
cos(_Tp __x) 
# 216
{ return __builtin_cos(__x); } 
# 218
using ::cosh;
# 222
inline float cosh(float __x) 
# 223
{ return __builtin_coshf(__x); } 
# 226
inline long double cosh(long double __x) 
# 227
{ return __builtin_coshl(__x); } 
# 230
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 234
cosh(_Tp __x) 
# 235
{ return __builtin_cosh(__x); } 
# 237
using ::exp;
# 241
inline float exp(float __x) 
# 242
{ return __builtin_expf(__x); } 
# 245
inline long double exp(long double __x) 
# 246
{ return __builtin_expl(__x); } 
# 249
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 253
exp(_Tp __x) 
# 254
{ return __builtin_exp(__x); } 
# 256
using ::fabs;
# 260
inline float fabs(float __x) 
# 261
{ return __builtin_fabsf(__x); } 
# 264
inline long double fabs(long double __x) 
# 265
{ return __builtin_fabsl(__x); } 
# 268
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 272
fabs(_Tp __x) 
# 273
{ return __builtin_fabs(__x); } 
# 275
using ::floor;
# 279
inline float floor(float __x) 
# 280
{ return __builtin_floorf(__x); } 
# 283
inline long double floor(long double __x) 
# 284
{ return __builtin_floorl(__x); } 
# 287
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 291
floor(_Tp __x) 
# 292
{ return __builtin_floor(__x); } 
# 294
using ::fmod;
# 298
inline float fmod(float __x, float __y) 
# 299
{ return __builtin_fmodf(__x, __y); } 
# 302
inline long double fmod(long double __x, long double __y) 
# 303
{ return __builtin_fmodl(__x, __y); } 
# 306
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 309
fmod(_Tp __x, _Up __y) 
# 310
{ 
# 311
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 312
return fmod((__type)__x, (__type)__y); 
# 313
} 
# 315
using ::frexp;
# 319
inline float frexp(float __x, int *__exp) 
# 320
{ return __builtin_frexpf(__x, __exp); } 
# 323
inline long double frexp(long double __x, int *__exp) 
# 324
{ return __builtin_frexpl(__x, __exp); } 
# 327
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 331
frexp(_Tp __x, int *__exp) 
# 332
{ return __builtin_frexp(__x, __exp); } 
# 334
using ::ldexp;
# 338
inline float ldexp(float __x, int __exp) 
# 339
{ return __builtin_ldexpf(__x, __exp); } 
# 342
inline long double ldexp(long double __x, int __exp) 
# 343
{ return __builtin_ldexpl(__x, __exp); } 
# 346
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 350
ldexp(_Tp __x, int __exp) 
# 351
{ return __builtin_ldexp(__x, __exp); } 
# 353
using ::log;
# 357
inline float log(float __x) 
# 358
{ return __builtin_logf(__x); } 
# 361
inline long double log(long double __x) 
# 362
{ return __builtin_logl(__x); } 
# 365
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 369
log(_Tp __x) 
# 370
{ return __builtin_log(__x); } 
# 372
using ::log10;
# 376
inline float log10(float __x) 
# 377
{ return __builtin_log10f(__x); } 
# 380
inline long double log10(long double __x) 
# 381
{ return __builtin_log10l(__x); } 
# 384
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 388
log10(_Tp __x) 
# 389
{ return __builtin_log10(__x); } 
# 391
using ::modf;
# 395
inline float modf(float __x, float *__iptr) 
# 396
{ return __builtin_modff(__x, __iptr); } 
# 399
inline long double modf(long double __x, long double *__iptr) 
# 400
{ return __builtin_modfl(__x, __iptr); } 
# 403
using ::pow;
# 407
inline float pow(float __x, float __y) 
# 408
{ return __builtin_powf(__x, __y); } 
# 411
inline long double pow(long double __x, long double __y) 
# 412
{ return __builtin_powl(__x, __y); } 
# 418
inline double pow(double __x, int __i) 
# 419
{ return __builtin_powi(__x, __i); } 
# 422
inline float pow(float __x, int __n) 
# 423
{ return __builtin_powif(__x, __n); } 
# 426
inline long double pow(long double __x, int __n) 
# 427
{ return __builtin_powil(__x, __n); } 
# 431
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 434
pow(_Tp __x, _Up __y) 
# 435
{ 
# 436
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 437
return pow((__type)__x, (__type)__y); 
# 438
} 
# 440
using ::sin;
# 444
inline float sin(float __x) 
# 445
{ return __builtin_sinf(__x); } 
# 448
inline long double sin(long double __x) 
# 449
{ return __builtin_sinl(__x); } 
# 452
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 456
sin(_Tp __x) 
# 457
{ return __builtin_sin(__x); } 
# 459
using ::sinh;
# 463
inline float sinh(float __x) 
# 464
{ return __builtin_sinhf(__x); } 
# 467
inline long double sinh(long double __x) 
# 468
{ return __builtin_sinhl(__x); } 
# 471
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 475
sinh(_Tp __x) 
# 476
{ return __builtin_sinh(__x); } 
# 478
using ::sqrt;
# 482
inline float sqrt(float __x) 
# 483
{ return __builtin_sqrtf(__x); } 
# 486
inline long double sqrt(long double __x) 
# 487
{ return __builtin_sqrtl(__x); } 
# 490
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 494
sqrt(_Tp __x) 
# 495
{ return __builtin_sqrt(__x); } 
# 497
using ::tan;
# 501
inline float tan(float __x) 
# 502
{ return __builtin_tanf(__x); } 
# 505
inline long double tan(long double __x) 
# 506
{ return __builtin_tanl(__x); } 
# 509
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 513
tan(_Tp __x) 
# 514
{ return __builtin_tan(__x); } 
# 516
using ::tanh;
# 520
inline float tanh(float __x) 
# 521
{ return __builtin_tanhf(__x); } 
# 524
inline long double tanh(long double __x) 
# 525
{ return __builtin_tanhl(__x); } 
# 528
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 532
tanh(_Tp __x) 
# 533
{ return __builtin_tanh(__x); } 
# 536
}
# 555 "/usr/include/c++/4.8.2/cmath" 3
namespace std __attribute((__visibility__("default"))) { 
# 805 "/usr/include/c++/4.8.2/cmath" 3
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 808
fpclassify(_Tp __f) 
# 809
{ 
# 810
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 811
return __builtin_fpclassify(0, 1, 4, 3, 2, (__type)__f); 
# 813
} 
# 815
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 818
isfinite(_Tp __f) 
# 819
{ 
# 820
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 821
return __builtin_isfinite((__type)__f); 
# 822
} 
# 824
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 827
isinf(_Tp __f) 
# 828
{ 
# 829
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 830
return __builtin_isinf((__type)__f); 
# 831
} 
# 833
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 836
isnan(_Tp __f) 
# 837
{ 
# 838
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 839
return __builtin_isnan((__type)__f); 
# 840
} 
# 842
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 845
isnormal(_Tp __f) 
# 846
{ 
# 847
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 848
return __builtin_isnormal((__type)__f); 
# 849
} 
# 851
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 854
signbit(_Tp __f) 
# 855
{ 
# 856
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 857
return __builtin_signbit((__type)__f); 
# 858
} 
# 860
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 863
isgreater(_Tp __f1, _Tp __f2) 
# 864
{ 
# 865
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 866
return __builtin_isgreater((__type)__f1, (__type)__f2); 
# 867
} 
# 869
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 872
isgreaterequal(_Tp __f1, _Tp __f2) 
# 873
{ 
# 874
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 875
return __builtin_isgreaterequal((__type)__f1, (__type)__f2); 
# 876
} 
# 878
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 881
isless(_Tp __f1, _Tp __f2) 
# 882
{ 
# 883
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 884
return __builtin_isless((__type)__f1, (__type)__f2); 
# 885
} 
# 887
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 890
islessequal(_Tp __f1, _Tp __f2) 
# 891
{ 
# 892
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 893
return __builtin_islessequal((__type)__f1, (__type)__f2); 
# 894
} 
# 896
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 899
islessgreater(_Tp __f1, _Tp __f2) 
# 900
{ 
# 901
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 902
return __builtin_islessgreater((__type)__f1, (__type)__f2); 
# 903
} 
# 905
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 908
isunordered(_Tp __f1, _Tp __f2) 
# 909
{ 
# 910
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 911
return __builtin_isunordered((__type)__f1, (__type)__f2); 
# 912
} 
# 917
}
# 114 "/usr/include/c++/4.8.2/cstdlib" 3
namespace std __attribute((__visibility__("default"))) { 
# 118
using ::div_t;
# 119
using ::ldiv_t;
# 121
using ::abort;
# 122
using ::abs;
# 123
using ::atexit;
# 129
using ::atof;
# 130
using ::atoi;
# 131
using ::atol;
# 132
using ::bsearch;
# 133
using ::calloc;
# 134
using ::div;
# 135
using ::exit;
# 136
using ::free;
# 137
using ::getenv;
# 138
using ::labs;
# 139
using ::ldiv;
# 140
using ::malloc;
# 142
using ::mblen;
# 143
using ::mbstowcs;
# 144
using ::mbtowc;
# 146
using ::qsort;
# 152
using ::rand;
# 153
using ::realloc;
# 154
using ::srand;
# 155
using ::strtod;
# 156
using ::strtol;
# 157
using ::strtoul;
# 158
using ::system;
# 160
using ::wcstombs;
# 161
using ::wctomb;
# 166
inline long abs(long __i) { return __builtin_labs(__i); } 
# 169
inline ldiv_t div(long __i, long __j) { return ldiv(__i, __j); } 
# 174
inline long long abs(long long __x) { return __builtin_llabs(__x); } 
# 179
inline __int128_t abs(__int128_t __x) { return (__x >= (0)) ? __x : (-__x); } 
# 183
}
# 196 "/usr/include/c++/4.8.2/cstdlib" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 201
using ::lldiv_t;
# 207
using ::_Exit;
# 211
using ::llabs;
# 214
inline lldiv_t div(long long __n, long long __d) 
# 215
{ lldiv_t __q; (__q.quot) = (__n / __d); (__q.rem) = (__n % __d); return __q; } 
# 217
using ::lldiv;
# 228 "/usr/include/c++/4.8.2/cstdlib" 3
using ::atoll;
# 229
using ::strtoll;
# 230
using ::strtoull;
# 232
using ::strtof;
# 233
using ::strtold;
# 236
}
# 238
namespace std { 
# 241
using __gnu_cxx::lldiv_t;
# 243
using __gnu_cxx::_Exit;
# 245
using __gnu_cxx::llabs;
# 246
using __gnu_cxx::div;
# 247
using __gnu_cxx::lldiv;
# 249
using __gnu_cxx::atoll;
# 250
using __gnu_cxx::strtof;
# 251
using __gnu_cxx::strtoll;
# 252
using __gnu_cxx::strtoull;
# 253
using __gnu_cxx::strtold;
# 254
}
# 8984 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
__attribute((always_inline)) inline int signbit(float x); 
# 8988
__attribute((always_inline)) inline int signbit(double x); 
# 8990
__attribute((always_inline)) inline int signbit(long double x); 
# 8992
__attribute((always_inline)) inline int isfinite(float x); 
# 8996
__attribute((always_inline)) inline int isfinite(double x); 
# 8998
__attribute((always_inline)) inline int isfinite(long double x); 
# 9005
__attribute((always_inline)) inline int isnan(float x); 
# 9013
extern "C" __attribute((always_inline)) inline int isnan(double x) throw(); 
# 9018
__attribute((always_inline)) inline int isnan(long double x); 
# 9026
__attribute((always_inline)) inline int isinf(float x); 
# 9035 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern "C" __attribute((always_inline)) inline int isinf(double x) throw(); 
# 9040
__attribute((always_inline)) inline int isinf(long double x); 
# 9098 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std { 
# 9100
template< class T> extern T __pow_helper(T, int); 
# 9101
template< class T> extern T __cmath_power(T, unsigned); 
# 9102
}
# 9104
using std::abs;
# 9105
using std::fabs;
# 9106
using std::ceil;
# 9107
using std::floor;
# 9108
using std::sqrt;
# 9110
using std::pow;
# 9112
using std::log;
# 9113
using std::log10;
# 9114
using std::fmod;
# 9115
using std::modf;
# 9116
using std::exp;
# 9117
using std::frexp;
# 9118
using std::ldexp;
# 9119
using std::asin;
# 9120
using std::sin;
# 9121
using std::sinh;
# 9122
using std::acos;
# 9123
using std::cos;
# 9124
using std::cosh;
# 9125
using std::atan;
# 9126
using std::atan2;
# 9127
using std::tan;
# 9128
using std::tanh;
# 9493 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std { 
# 9502 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern inline long long abs(long long); 
# 9512 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern inline long abs(long); 
# 9513
extern inline float abs(float); 
# 9514
extern inline double abs(double); 
# 9515
extern inline float fabs(float); 
# 9516
extern inline float ceil(float); 
# 9517
extern inline float floor(float); 
# 9518
extern inline float sqrt(float); 
# 9519
extern inline float pow(float, float); 
# 9528 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern inline float pow(float, int); 
# 9529
extern inline double pow(double, int); 
# 9534
extern inline float log(float); 
# 9535
extern inline float log10(float); 
# 9536
extern inline float fmod(float, float); 
# 9537
extern inline float modf(float, float *); 
# 9538
extern inline float exp(float); 
# 9539
extern inline float frexp(float, int *); 
# 9540
extern inline float ldexp(float, int); 
# 9541
extern inline float asin(float); 
# 9542
extern inline float sin(float); 
# 9543
extern inline float sinh(float); 
# 9544
extern inline float acos(float); 
# 9545
extern inline float cos(float); 
# 9546
extern inline float cosh(float); 
# 9547
extern inline float atan(float); 
# 9548
extern inline float atan2(float, float); 
# 9549
extern inline float tan(float); 
# 9550
extern inline float tanh(float); 
# 9624 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
}
# 9761 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline float logb(float a); 
# 9763
static inline int ilogb(float a); 
# 9765
static inline float scalbn(float a, int b); 
# 9767
static inline float scalbln(float a, long b); 
# 9769
static inline float exp2(float a); 
# 9771
static inline float expm1(float a); 
# 9773
static inline float log2(float a); 
# 9775
static inline float log1p(float a); 
# 9777
static inline float acosh(float a); 
# 9779
static inline float asinh(float a); 
# 9781
static inline float atanh(float a); 
# 9783
static inline float hypot(float a, float b); 
# 9785
static inline float cbrt(float a); 
# 9787
static inline float erf(float a); 
# 9789
static inline float erfc(float a); 
# 9791
static inline float lgamma(float a); 
# 9793
static inline float tgamma(float a); 
# 9795
static inline float copysign(float a, float b); 
# 9797
static inline float nextafter(float a, float b); 
# 9799
static inline float remainder(float a, float b); 
# 9801
static inline float remquo(float a, float b, int * quo); 
# 9803
static inline float round(float a); 
# 9805
static inline long lround(float a); 
# 9807
static inline long long llround(float a); 
# 9809
static inline float trunc(float a); 
# 9811
static inline float rint(float a); 
# 9813
static inline long lrint(float a); 
# 9815
static inline long long llrint(float a); 
# 9817
static inline float nearbyint(float a); 
# 9819
static inline float fdim(float a, float b); 
# 9821
static inline float fma(float a, float b, float c); 
# 9823
static inline float fmax(float a, float b); 
# 9825
static inline float fmin(float a, float b); 
# 9864 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline float exp10(float a); 
# 9866
static inline float rsqrt(float a); 
# 9868
static inline float rcbrt(float a); 
# 9870
static inline float sinpi(float a); 
# 9872
static inline float cospi(float a); 
# 9874
static inline void sincospi(float a, float * sptr, float * cptr); 
# 9876
static inline void sincos(float a, float * sptr, float * cptr); 
# 9878
static inline float j0(float a); 
# 9880
static inline float j1(float a); 
# 9882
static inline float jn(int n, float a); 
# 9884
static inline float y0(float a); 
# 9886
static inline float y1(float a); 
# 9888
static inline float yn(int n, float a); 
# 9890
static inline float cyl_bessel_i0(float a); 
# 9892
static inline float cyl_bessel_i1(float a); 
# 9894
static inline float erfinv(float a); 
# 9896
static inline float erfcinv(float a); 
# 9898
static inline float normcdfinv(float a); 
# 9900
static inline float normcdf(float a); 
# 9902
static inline float erfcx(float a); 
# 9904
static inline double copysign(double a, float b); 
# 9906
static inline double copysign(float a, double b); 
# 9908
static inline unsigned min(unsigned a, unsigned b); 
# 9910
static inline unsigned min(int a, unsigned b); 
# 9912
static inline unsigned min(unsigned a, int b); 
# 9914
static inline long min(long a, long b); 
# 9916
static inline unsigned long min(unsigned long a, unsigned long b); 
# 9918
static inline unsigned long min(long a, unsigned long b); 
# 9920
static inline unsigned long min(unsigned long a, long b); 
# 9922
static inline long long min(long long a, long long b); 
# 9924
static inline unsigned long long min(unsigned long long a, unsigned long long b); 
# 9926
static inline unsigned long long min(long long a, unsigned long long b); 
# 9928
static inline unsigned long long min(unsigned long long a, long long b); 
# 9930
static inline float min(float a, float b); 
# 9932
static inline double min(double a, double b); 
# 9934
static inline double min(float a, double b); 
# 9936
static inline double min(double a, float b); 
# 9938
static inline unsigned max(unsigned a, unsigned b); 
# 9940
static inline unsigned max(int a, unsigned b); 
# 9942
static inline unsigned max(unsigned a, int b); 
# 9944
static inline long max(long a, long b); 
# 9946
static inline unsigned long max(unsigned long a, unsigned long b); 
# 9948
static inline unsigned long max(long a, unsigned long b); 
# 9950
static inline unsigned long max(unsigned long a, long b); 
# 9952
static inline long long max(long long a, long long b); 
# 9954
static inline unsigned long long max(unsigned long long a, unsigned long long b); 
# 9956
static inline unsigned long long max(long long a, unsigned long long b); 
# 9958
static inline unsigned long long max(unsigned long long a, long long b); 
# 9960
static inline float max(float a, float b); 
# 9962
static inline double max(double a, double b); 
# 9964
static inline double max(float a, double b); 
# 9966
static inline double max(double a, float b); 
# 327 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
__attribute((always_inline)) inline int signbit(float x) { return __signbitf(x); } 
# 331
__attribute((always_inline)) inline int signbit(double x) { return __signbit(x); } 
# 333
__attribute((always_inline)) inline int signbit(long double x) { return __signbitl(x); } 
# 344 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
__attribute((always_inline)) inline int isfinite(float x) { return __finitef(x); } 
# 359 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
__attribute((always_inline)) inline int isfinite(double x) { return __finite(x); } 
# 372 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
__attribute((always_inline)) inline int isfinite(long double x) { return __finitel(x); } 
# 375
__attribute((always_inline)) inline int isnan(float x) { return __isnanf(x); } 
# 379
__attribute((always_inline)) inline int isnan(double x) throw() { return __isnan(x); } 
# 381
__attribute((always_inline)) inline int isnan(long double x) { return __isnanl(x); } 
# 383
__attribute((always_inline)) inline int isinf(float x) { return __isinff(x); } 
# 387
__attribute((always_inline)) inline int isinf(double x) throw() { return __isinf(x); } 
# 389
__attribute((always_inline)) inline int isinf(long double x) { return __isinfl(x); } 
# 585 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
static inline float logb(float a) 
# 586
{ 
# 587
return logbf(a); 
# 588
} 
# 590
static inline int ilogb(float a) 
# 591
{ 
# 592
return ilogbf(a); 
# 593
} 
# 595
static inline float scalbn(float a, int b) 
# 596
{ 
# 597
return scalbnf(a, b); 
# 598
} 
# 600
static inline float scalbln(float a, long b) 
# 601
{ 
# 602
return scalblnf(a, b); 
# 603
} 
# 605
static inline float exp2(float a) 
# 606
{ 
# 607
return exp2f(a); 
# 608
} 
# 610
static inline float expm1(float a) 
# 611
{ 
# 612
return expm1f(a); 
# 613
} 
# 615
static inline float log2(float a) 
# 616
{ 
# 617
return log2f(a); 
# 618
} 
# 620
static inline float log1p(float a) 
# 621
{ 
# 622
return log1pf(a); 
# 623
} 
# 625
static inline float acosh(float a) 
# 626
{ 
# 627
return acoshf(a); 
# 628
} 
# 630
static inline float asinh(float a) 
# 631
{ 
# 632
return asinhf(a); 
# 633
} 
# 635
static inline float atanh(float a) 
# 636
{ 
# 637
return atanhf(a); 
# 638
} 
# 640
static inline float hypot(float a, float b) 
# 641
{ 
# 642
return hypotf(a, b); 
# 643
} 
# 645
static inline float cbrt(float a) 
# 646
{ 
# 647
return cbrtf(a); 
# 648
} 
# 650
static inline float erf(float a) 
# 651
{ 
# 652
return erff(a); 
# 653
} 
# 655
static inline float erfc(float a) 
# 656
{ 
# 657
return erfcf(a); 
# 658
} 
# 660
static inline float lgamma(float a) 
# 661
{ 
# 662
return lgammaf(a); 
# 663
} 
# 665
static inline float tgamma(float a) 
# 666
{ 
# 667
return tgammaf(a); 
# 668
} 
# 670
static inline float copysign(float a, float b) 
# 671
{ 
# 672
return copysignf(a, b); 
# 673
} 
# 675
static inline float nextafter(float a, float b) 
# 676
{ 
# 677
return nextafterf(a, b); 
# 678
} 
# 680
static inline float remainder(float a, float b) 
# 681
{ 
# 682
return remainderf(a, b); 
# 683
} 
# 685
static inline float remquo(float a, float b, int *quo) 
# 686
{ 
# 687
return remquof(a, b, quo); 
# 688
} 
# 690
static inline float round(float a) 
# 691
{ 
# 692
return roundf(a); 
# 693
} 
# 695
static inline long lround(float a) 
# 696
{ 
# 697
return lroundf(a); 
# 698
} 
# 700
static inline long long llround(float a) 
# 701
{ 
# 702
return llroundf(a); 
# 703
} 
# 705
static inline float trunc(float a) 
# 706
{ 
# 707
return truncf(a); 
# 708
} 
# 710
static inline float rint(float a) 
# 711
{ 
# 712
return rintf(a); 
# 713
} 
# 715
static inline long lrint(float a) 
# 716
{ 
# 717
return lrintf(a); 
# 718
} 
# 720
static inline long long llrint(float a) 
# 721
{ 
# 722
return llrintf(a); 
# 723
} 
# 725
static inline float nearbyint(float a) 
# 726
{ 
# 727
return nearbyintf(a); 
# 728
} 
# 730
static inline float fdim(float a, float b) 
# 731
{ 
# 732
return fdimf(a, b); 
# 733
} 
# 735
static inline float fma(float a, float b, float c) 
# 736
{ 
# 737
return fmaf(a, b, c); 
# 738
} 
# 740
static inline float fmax(float a, float b) 
# 741
{ 
# 742
return fmaxf(a, b); 
# 743
} 
# 745
static inline float fmin(float a, float b) 
# 746
{ 
# 747
return fminf(a, b); 
# 748
} 
# 756
static inline float exp10(float a) 
# 757
{ 
# 758
return exp10f(a); 
# 759
} 
# 761
static inline float rsqrt(float a) 
# 762
{ 
# 763
return rsqrtf(a); 
# 764
} 
# 766
static inline float rcbrt(float a) 
# 767
{ 
# 768
return rcbrtf(a); 
# 769
} 
# 771
static inline float sinpi(float a) 
# 772
{ 
# 773
return sinpif(a); 
# 774
} 
# 776
static inline float cospi(float a) 
# 777
{ 
# 778
return cospif(a); 
# 779
} 
# 781
static inline void sincospi(float a, float *sptr, float *cptr) 
# 782
{ 
# 783
sincospif(a, sptr, cptr); 
# 784
} 
# 786
static inline void sincos(float a, float *sptr, float *cptr) 
# 787
{ 
# 788
sincosf(a, sptr, cptr); 
# 789
} 
# 791
static inline float j0(float a) 
# 792
{ 
# 793
return j0f(a); 
# 794
} 
# 796
static inline float j1(float a) 
# 797
{ 
# 798
return j1f(a); 
# 799
} 
# 801
static inline float jn(int n, float a) 
# 802
{ 
# 803
return jnf(n, a); 
# 804
} 
# 806
static inline float y0(float a) 
# 807
{ 
# 808
return y0f(a); 
# 809
} 
# 811
static inline float y1(float a) 
# 812
{ 
# 813
return y1f(a); 
# 814
} 
# 816
static inline float yn(int n, float a) 
# 817
{ 
# 818
return ynf(n, a); 
# 819
} 
# 821
static inline float cyl_bessel_i0(float a) 
# 822
{ 
# 823
return cyl_bessel_i0f(a); 
# 824
} 
# 826
static inline float cyl_bessel_i1(float a) 
# 827
{ 
# 828
return cyl_bessel_i1f(a); 
# 829
} 
# 831
static inline float erfinv(float a) 
# 832
{ 
# 833
return erfinvf(a); 
# 834
} 
# 836
static inline float erfcinv(float a) 
# 837
{ 
# 838
return erfcinvf(a); 
# 839
} 
# 841
static inline float normcdfinv(float a) 
# 842
{ 
# 843
return normcdfinvf(a); 
# 844
} 
# 846
static inline float normcdf(float a) 
# 847
{ 
# 848
return normcdff(a); 
# 849
} 
# 851
static inline float erfcx(float a) 
# 852
{ 
# 853
return erfcxf(a); 
# 854
} 
# 856
static inline double copysign(double a, float b) 
# 857
{ 
# 858
return copysign(a, (double)b); 
# 859
} 
# 861
static inline double copysign(float a, double b) 
# 862
{ 
# 863
return copysign((double)a, b); 
# 864
} 
# 866
static inline unsigned min(unsigned a, unsigned b) 
# 867
{ 
# 868
return umin(a, b); 
# 869
} 
# 871
static inline unsigned min(int a, unsigned b) 
# 872
{ 
# 873
return umin((unsigned)a, b); 
# 874
} 
# 876
static inline unsigned min(unsigned a, int b) 
# 877
{ 
# 878
return umin(a, (unsigned)b); 
# 879
} 
# 881
static inline long min(long a, long b) 
# 882
{ 
# 888
if (sizeof(long) == sizeof(int)) { 
# 892
return (long)min((int)a, (int)b); 
# 893
} else { 
# 894
return (long)llmin((long long)a, (long long)b); 
# 895
}  
# 896
} 
# 898
static inline unsigned long min(unsigned long a, unsigned long b) 
# 899
{ 
# 903
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 907
return (unsigned long)umin((unsigned)a, (unsigned)b); 
# 908
} else { 
# 909
return (unsigned long)ullmin((unsigned long long)a, (unsigned long long)b); 
# 910
}  
# 911
} 
# 913
static inline unsigned long min(long a, unsigned long b) 
# 914
{ 
# 918
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 922
return (unsigned long)umin((unsigned)a, (unsigned)b); 
# 923
} else { 
# 924
return (unsigned long)ullmin((unsigned long long)a, (unsigned long long)b); 
# 925
}  
# 926
} 
# 928
static inline unsigned long min(unsigned long a, long b) 
# 929
{ 
# 933
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 937
return (unsigned long)umin((unsigned)a, (unsigned)b); 
# 938
} else { 
# 939
return (unsigned long)ullmin((unsigned long long)a, (unsigned long long)b); 
# 940
}  
# 941
} 
# 943
static inline long long min(long long a, long long b) 
# 944
{ 
# 945
return llmin(a, b); 
# 946
} 
# 948
static inline unsigned long long min(unsigned long long a, unsigned long long b) 
# 949
{ 
# 950
return ullmin(a, b); 
# 951
} 
# 953
static inline unsigned long long min(long long a, unsigned long long b) 
# 954
{ 
# 955
return ullmin((unsigned long long)a, b); 
# 956
} 
# 958
static inline unsigned long long min(unsigned long long a, long long b) 
# 959
{ 
# 960
return ullmin(a, (unsigned long long)b); 
# 961
} 
# 963
static inline float min(float a, float b) 
# 964
{ 
# 965
return fminf(a, b); 
# 966
} 
# 968
static inline double min(double a, double b) 
# 969
{ 
# 970
return fmin(a, b); 
# 971
} 
# 973
static inline double min(float a, double b) 
# 974
{ 
# 975
return fmin((double)a, b); 
# 976
} 
# 978
static inline double min(double a, float b) 
# 979
{ 
# 980
return fmin(a, (double)b); 
# 981
} 
# 983
static inline unsigned max(unsigned a, unsigned b) 
# 984
{ 
# 985
return umax(a, b); 
# 986
} 
# 988
static inline unsigned max(int a, unsigned b) 
# 989
{ 
# 990
return umax((unsigned)a, b); 
# 991
} 
# 993
static inline unsigned max(unsigned a, int b) 
# 994
{ 
# 995
return umax(a, (unsigned)b); 
# 996
} 
# 998
static inline long max(long a, long b) 
# 999
{ 
# 1004
if (sizeof(long) == sizeof(int)) { 
# 1008
return (long)max((int)a, (int)b); 
# 1009
} else { 
# 1010
return (long)llmax((long long)a, (long long)b); 
# 1011
}  
# 1012
} 
# 1014
static inline unsigned long max(unsigned long a, unsigned long b) 
# 1015
{ 
# 1019
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1023
return (unsigned long)umax((unsigned)a, (unsigned)b); 
# 1024
} else { 
# 1025
return (unsigned long)ullmax((unsigned long long)a, (unsigned long long)b); 
# 1026
}  
# 1027
} 
# 1029
static inline unsigned long max(long a, unsigned long b) 
# 1030
{ 
# 1034
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1038
return (unsigned long)umax((unsigned)a, (unsigned)b); 
# 1039
} else { 
# 1040
return (unsigned long)ullmax((unsigned long long)a, (unsigned long long)b); 
# 1041
}  
# 1042
} 
# 1044
static inline unsigned long max(unsigned long a, long b) 
# 1045
{ 
# 1049
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1053
return (unsigned long)umax((unsigned)a, (unsigned)b); 
# 1054
} else { 
# 1055
return (unsigned long)ullmax((unsigned long long)a, (unsigned long long)b); 
# 1056
}  
# 1057
} 
# 1059
static inline long long max(long long a, long long b) 
# 1060
{ 
# 1061
return llmax(a, b); 
# 1062
} 
# 1064
static inline unsigned long long max(unsigned long long a, unsigned long long b) 
# 1065
{ 
# 1066
return ullmax(a, b); 
# 1067
} 
# 1069
static inline unsigned long long max(long long a, unsigned long long b) 
# 1070
{ 
# 1071
return ullmax((unsigned long long)a, b); 
# 1072
} 
# 1074
static inline unsigned long long max(unsigned long long a, long long b) 
# 1075
{ 
# 1076
return ullmax(a, (unsigned long long)b); 
# 1077
} 
# 1079
static inline float max(float a, float b) 
# 1080
{ 
# 1081
return fmaxf(a, b); 
# 1082
} 
# 1084
static inline double max(double a, double b) 
# 1085
{ 
# 1086
return fmax(a, b); 
# 1087
} 
# 1089
static inline double max(float a, double b) 
# 1090
{ 
# 1091
return fmax((double)a, b); 
# 1092
} 
# 1094
static inline double max(double a, float b) 
# 1095
{ 
# 1096
return fmax(a, (double)b); 
# 1097
} 
# 1108 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
inline int min(int a, int b) 
# 1109
{ 
# 1110
return (a < b) ? a : b; 
# 1111
} 
# 1113
inline unsigned umin(unsigned a, unsigned b) 
# 1114
{ 
# 1115
return (a < b) ? a : b; 
# 1116
} 
# 1118
inline long long llmin(long long a, long long b) 
# 1119
{ 
# 1120
return (a < b) ? a : b; 
# 1121
} 
# 1123
inline unsigned long long ullmin(unsigned long long a, unsigned long long 
# 1124
b) 
# 1125
{ 
# 1126
return (a < b) ? a : b; 
# 1127
} 
# 1129
inline int max(int a, int b) 
# 1130
{ 
# 1131
return (a > b) ? a : b; 
# 1132
} 
# 1134
inline unsigned umax(unsigned a, unsigned b) 
# 1135
{ 
# 1136
return (a > b) ? a : b; 
# 1137
} 
# 1139
inline long long llmax(long long a, long long b) 
# 1140
{ 
# 1141
return (a > b) ? a : b; 
# 1142
} 
# 1144
inline unsigned long long ullmax(unsigned long long a, unsigned long long 
# 1145
b) 
# 1146
{ 
# 1147
return (a > b) ? a : b; 
# 1148
} 
# 74 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_surface_types.h"
template< class T, int dim = 1> 
# 75
struct surface : public surfaceReference { 
# 78
surface() 
# 79
{ 
# 80
(channelDesc) = cudaCreateChannelDesc< T> (); 
# 81
} 
# 83
surface(cudaChannelFormatDesc desc) 
# 84
{ 
# 85
(channelDesc) = desc; 
# 86
} 
# 88
}; 
# 90
template< int dim> 
# 91
struct surface< void, dim>  : public surfaceReference { 
# 94
surface() 
# 95
{ 
# 96
(channelDesc) = cudaCreateChannelDesc< void> (); 
# 97
} 
# 99
}; 
# 74 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_texture_types.h"
template< class T, int texType = 1, cudaTextureReadMode mode = cudaReadModeElementType> 
# 75
struct texture : public textureReference { 
# 78
texture(int norm = 0, cudaTextureFilterMode 
# 79
fMode = cudaFilterModePoint, cudaTextureAddressMode 
# 80
aMode = cudaAddressModeClamp) 
# 81
{ 
# 82
(normalized) = norm; 
# 83
(filterMode) = fMode; 
# 84
((addressMode)[0]) = aMode; 
# 85
((addressMode)[1]) = aMode; 
# 86
((addressMode)[2]) = aMode; 
# 87
(channelDesc) = cudaCreateChannelDesc< T> (); 
# 88
(sRGB) = 0; 
# 89
} 
# 91
texture(int norm, cudaTextureFilterMode 
# 92
fMode, cudaTextureAddressMode 
# 93
aMode, cudaChannelFormatDesc 
# 94
desc) 
# 95
{ 
# 96
(normalized) = norm; 
# 97
(filterMode) = fMode; 
# 98
((addressMode)[0]) = aMode; 
# 99
((addressMode)[1]) = aMode; 
# 100
((addressMode)[2]) = aMode; 
# 101
(channelDesc) = desc; 
# 102
(sRGB) = 0; 
# 103
} 
# 105
}; 
# 89 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.h"
extern "C" {
# 3217 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.h"
}
# 3225
__attribute__((unused)) static inline int mulhi(int a, int b); 
# 3227
__attribute__((unused)) static inline unsigned mulhi(unsigned a, unsigned b); 
# 3229
__attribute__((unused)) static inline unsigned mulhi(int a, unsigned b); 
# 3231
__attribute__((unused)) static inline unsigned mulhi(unsigned a, int b); 
# 3233
__attribute__((unused)) static inline long long mul64hi(long long a, long long b); 
# 3235
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, unsigned long long b); 
# 3237
__attribute__((unused)) static inline unsigned long long mul64hi(long long a, unsigned long long b); 
# 3239
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, long long b); 
# 3241
__attribute__((unused)) static inline int float_as_int(float a); 
# 3243
__attribute__((unused)) static inline float int_as_float(int a); 
# 3245
__attribute__((unused)) static inline unsigned float_as_uint(float a); 
# 3247
__attribute__((unused)) static inline float uint_as_float(unsigned a); 
# 3249
__attribute__((unused)) static inline float saturate(float a); 
# 3251
__attribute__((unused)) static inline int mul24(int a, int b); 
# 3253
__attribute__((unused)) static inline unsigned umul24(unsigned a, unsigned b); 
# 3255
__attribute__((unused)) static inline int float2int(float a, cudaRoundMode mode = cudaRoundZero); 
# 3257
__attribute__((unused)) static inline unsigned float2uint(float a, cudaRoundMode mode = cudaRoundZero); 
# 3259
__attribute__((unused)) static inline float int2float(int a, cudaRoundMode mode = cudaRoundNearest); 
# 3261
__attribute__((unused)) static inline float uint2float(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
# 90 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int mulhi(int a, int b) 
# 91
{int volatile ___ = 1;(void)a;(void)b;
# 93
::exit(___);}
#if 0
# 91
{ 
# 92
return __mulhi(a, b); 
# 93
} 
#endif
# 95 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(unsigned a, unsigned b) 
# 96
{int volatile ___ = 1;(void)a;(void)b;
# 98
::exit(___);}
#if 0
# 96
{ 
# 97
return __umulhi(a, b); 
# 98
} 
#endif
# 100 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(int a, unsigned b) 
# 101
{int volatile ___ = 1;(void)a;(void)b;
# 103
::exit(___);}
#if 0
# 101
{ 
# 102
return __umulhi((unsigned)a, b); 
# 103
} 
#endif
# 105 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(unsigned a, int b) 
# 106
{int volatile ___ = 1;(void)a;(void)b;
# 108
::exit(___);}
#if 0
# 106
{ 
# 107
return __umulhi(a, (unsigned)b); 
# 108
} 
#endif
# 110 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline long long mul64hi(long long a, long long b) 
# 111
{int volatile ___ = 1;(void)a;(void)b;
# 113
::exit(___);}
#if 0
# 111
{ 
# 112
return __mul64hi(a, b); 
# 113
} 
#endif
# 115 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, unsigned long long b) 
# 116
{int volatile ___ = 1;(void)a;(void)b;
# 118
::exit(___);}
#if 0
# 116
{ 
# 117
return __umul64hi(a, b); 
# 118
} 
#endif
# 120 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(long long a, unsigned long long b) 
# 121
{int volatile ___ = 1;(void)a;(void)b;
# 123
::exit(___);}
#if 0
# 121
{ 
# 122
return __umul64hi((unsigned long long)a, b); 
# 123
} 
#endif
# 125 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, long long b) 
# 126
{int volatile ___ = 1;(void)a;(void)b;
# 128
::exit(___);}
#if 0
# 126
{ 
# 127
return __umul64hi(a, (unsigned long long)b); 
# 128
} 
#endif
# 130 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int float_as_int(float a) 
# 131
{int volatile ___ = 1;(void)a;
# 133
::exit(___);}
#if 0
# 131
{ 
# 132
return __float_as_int(a); 
# 133
} 
#endif
# 135 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float int_as_float(int a) 
# 136
{int volatile ___ = 1;(void)a;
# 138
::exit(___);}
#if 0
# 136
{ 
# 137
return __int_as_float(a); 
# 138
} 
#endif
# 140 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned float_as_uint(float a) 
# 141
{int volatile ___ = 1;(void)a;
# 143
::exit(___);}
#if 0
# 141
{ 
# 142
return __float_as_uint(a); 
# 143
} 
#endif
# 145 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float uint_as_float(unsigned a) 
# 146
{int volatile ___ = 1;(void)a;
# 148
::exit(___);}
#if 0
# 146
{ 
# 147
return __uint_as_float(a); 
# 148
} 
#endif
# 149 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float saturate(float a) 
# 150
{int volatile ___ = 1;(void)a;
# 152
::exit(___);}
#if 0
# 150
{ 
# 151
return __saturatef(a); 
# 152
} 
#endif
# 154 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int mul24(int a, int b) 
# 155
{int volatile ___ = 1;(void)a;(void)b;
# 157
::exit(___);}
#if 0
# 155
{ 
# 156
return __mul24(a, b); 
# 157
} 
#endif
# 159 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned umul24(unsigned a, unsigned b) 
# 160
{int volatile ___ = 1;(void)a;(void)b;
# 162
::exit(___);}
#if 0
# 160
{ 
# 161
return __umul24(a, b); 
# 162
} 
#endif
# 164 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int float2int(float a, cudaRoundMode mode) 
# 165
{int volatile ___ = 1;(void)a;(void)mode;
# 170
::exit(___);}
#if 0
# 165
{ 
# 166
return (mode == (cudaRoundNearest)) ? __float2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2int_rd(a) : __float2int_rz(a))); 
# 170
} 
#endif
# 172 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned float2uint(float a, cudaRoundMode mode) 
# 173
{int volatile ___ = 1;(void)a;(void)mode;
# 178
::exit(___);}
#if 0
# 173
{ 
# 174
return (mode == (cudaRoundNearest)) ? __float2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2uint_rd(a) : __float2uint_rz(a))); 
# 178
} 
#endif
# 180 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float int2float(int a, cudaRoundMode mode) 
# 181
{int volatile ___ = 1;(void)a;(void)mode;
# 186
::exit(___);}
#if 0
# 181
{ 
# 182
return (mode == (cudaRoundZero)) ? __int2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __int2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __int2float_rd(a) : __int2float_rn(a))); 
# 186
} 
#endif
# 188 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float uint2float(unsigned a, cudaRoundMode mode) 
# 189
{int volatile ___ = 1;(void)a;(void)mode;
# 194
::exit(___);}
#if 0
# 189
{ 
# 190
return (mode == (cudaRoundZero)) ? __uint2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __uint2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __uint2float_rd(a) : __uint2float_rn(a))); 
# 194
} 
#endif
# 106 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 106
{ } 
#endif
# 108 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 108
{ } 
#endif
# 110 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 110
{ } 
#endif
# 112 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 112
{ } 
#endif
# 114 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 114
{ } 
#endif
# 116 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 116
{ } 
#endif
# 118 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 118
{ } 
#endif
# 120 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 120
{ } 
#endif
# 122 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 122
{ } 
#endif
# 124 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 126
{ } 
#endif
# 128 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 128
{ } 
#endif
# 130 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 130
{ } 
#endif
# 132 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 132
{ } 
#endif
# 134 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 134
{ } 
#endif
# 136 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 136
{ } 
#endif
# 138 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 138
{ } 
#endif
# 140 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 140
{ } 
#endif
# 142 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 142
{ } 
#endif
# 144 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 144
{ } 
#endif
# 146 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 146
{ } 
#endif
# 171 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
extern "C" {
# 180
}
# 189 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 189
{ } 
#endif
# 191 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 191
{ } 
#endif
# 193 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS(unsigned long long *address, unsigned long long compare, unsigned long long val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 193
{ } 
#endif
# 195 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute((deprecated("__any() is deprecated in favor of __any_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool any(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 195
{ } 
#endif
# 197 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute((deprecated("__all() is deprecated in favor of __all_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool all(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 197
{ } 
#endif
# 87 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern "C" {
# 1139 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
}
# 1147
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode); 
# 1149
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1151
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1153
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1155
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode = cudaRoundZero); 
# 1157
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero); 
# 1159
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode = cudaRoundZero); 
# 1161
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode = cudaRoundZero); 
# 1163
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1165
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1167
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest); 
# 1169
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
# 1171
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest); 
# 93 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode) 
# 94
{int volatile ___ = 1;(void)a;(void)b;(void)c;(void)mode;
# 99
::exit(___);}
#if 0
# 94
{ 
# 95
return (mode == (cudaRoundZero)) ? __fma_rz(a, b, c) : ((mode == (cudaRoundPosInf)) ? __fma_ru(a, b, c) : ((mode == (cudaRoundMinInf)) ? __fma_rd(a, b, c) : __fma_rn(a, b, c))); 
# 99
} 
#endif
# 101 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode) 
# 102
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 107
::exit(___);}
#if 0
# 102
{ 
# 103
return (mode == (cudaRoundZero)) ? __dmul_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dmul_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dmul_rd(a, b) : __dmul_rn(a, b))); 
# 107
} 
#endif
# 109 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode) 
# 110
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 115
::exit(___);}
#if 0
# 110
{ 
# 111
return (mode == (cudaRoundZero)) ? __dadd_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dadd_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dadd_rd(a, b) : __dadd_rn(a, b))); 
# 115
} 
#endif
# 117 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode) 
# 118
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 123
::exit(___);}
#if 0
# 118
{ 
# 119
return (mode == (cudaRoundZero)) ? __dsub_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dsub_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dsub_rd(a, b) : __dsub_rn(a, b))); 
# 123
} 
#endif
# 125 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode) 
# 126
{int volatile ___ = 1;(void)a;(void)mode;
# 131
::exit(___);}
#if 0
# 126
{ 
# 127
return (mode == (cudaRoundNearest)) ? __double2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2int_rd(a) : __double2int_rz(a))); 
# 131
} 
#endif
# 133 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode) 
# 134
{int volatile ___ = 1;(void)a;(void)mode;
# 139
::exit(___);}
#if 0
# 134
{ 
# 135
return (mode == (cudaRoundNearest)) ? __double2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2uint_rd(a) : __double2uint_rz(a))); 
# 139
} 
#endif
# 141 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode) 
# 142
{int volatile ___ = 1;(void)a;(void)mode;
# 147
::exit(___);}
#if 0
# 142
{ 
# 143
return (mode == (cudaRoundNearest)) ? __double2ll_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ll_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ll_rd(a) : __double2ll_rz(a))); 
# 147
} 
#endif
# 149 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode) 
# 150
{int volatile ___ = 1;(void)a;(void)mode;
# 155
::exit(___);}
#if 0
# 150
{ 
# 151
return (mode == (cudaRoundNearest)) ? __double2ull_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ull_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ull_rd(a) : __double2ull_rz(a))); 
# 155
} 
#endif
# 157 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode) 
# 158
{int volatile ___ = 1;(void)a;(void)mode;
# 163
::exit(___);}
#if 0
# 158
{ 
# 159
return (mode == (cudaRoundZero)) ? __ll2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ll2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ll2double_rd(a) : __ll2double_rn(a))); 
# 163
} 
#endif
# 165 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode) 
# 166
{int volatile ___ = 1;(void)a;(void)mode;
# 171
::exit(___);}
#if 0
# 166
{ 
# 167
return (mode == (cudaRoundZero)) ? __ull2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ull2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ull2double_rd(a) : __ull2double_rn(a))); 
# 171
} 
#endif
# 173 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode) 
# 174
{int volatile ___ = 1;(void)a;(void)mode;
# 176
::exit(___);}
#if 0
# 174
{ 
# 175
return (double)a; 
# 176
} 
#endif
# 178 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode) 
# 179
{int volatile ___ = 1;(void)a;(void)mode;
# 181
::exit(___);}
#if 0
# 179
{ 
# 180
return (double)a; 
# 181
} 
#endif
# 183 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode) 
# 184
{int volatile ___ = 1;(void)a;(void)mode;
# 186
::exit(___);}
#if 0
# 184
{ 
# 185
return (double)a; 
# 186
} 
#endif
# 89 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_20_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 89
{ } 
#endif
# 100 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 100
{ } 
#endif
# 102 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 102
{ } 
#endif
# 104 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 104
{ } 
#endif
# 106 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 106
{ } 
#endif
# 108 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 108
{ } 
#endif
# 110 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 110
{ } 
#endif
# 112 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 112
{ } 
#endif
# 114 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 114
{ } 
#endif
# 116 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 116
{ } 
#endif
# 118 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 118
{ } 
#endif
# 303 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 303
{ } 
#endif
# 306 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 306
{ } 
#endif
# 309 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 309
{ } 
#endif
# 312 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 312
{ } 
#endif
# 315 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 315
{ } 
#endif
# 318 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 318
{ } 
#endif
# 321 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 321
{ } 
#endif
# 324 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 324
{ } 
#endif
# 327 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 327
{ } 
#endif
# 330 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd_block(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 330
{ } 
#endif
# 333 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd_system(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 333
{ } 
#endif
# 336 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 336
{ } 
#endif
# 339 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 339
{ } 
#endif
# 342 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 342
{ } 
#endif
# 345 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 345
{ } 
#endif
# 348 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 348
{ } 
#endif
# 351 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 351
{ } 
#endif
# 354 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 354
{ } 
#endif
# 357 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 357
{ } 
#endif
# 360 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 360
{ } 
#endif
# 363 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 363
{ } 
#endif
# 366 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 366
{ } 
#endif
# 369 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 369
{ } 
#endif
# 372 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 372
{ } 
#endif
# 375 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 375
{ } 
#endif
# 378 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 378
{ } 
#endif
# 381 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 381
{ } 
#endif
# 384 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 384
{ } 
#endif
# 387 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 387
{ } 
#endif
# 390 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 390
{ } 
#endif
# 393 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 393
{ } 
#endif
# 396 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 396
{ } 
#endif
# 399 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 399
{ } 
#endif
# 402 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 402
{ } 
#endif
# 405 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 405
{ } 
#endif
# 408 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 408
{ } 
#endif
# 411 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 411
{ } 
#endif
# 414 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 414
{ } 
#endif
# 417 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 417
{ } 
#endif
# 420 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 420
{ } 
#endif
# 423 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 423
{ } 
#endif
# 426 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 426
{ } 
#endif
# 429 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 429
{ } 
#endif
# 432 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS_block(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 432
{ } 
#endif
# 435 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS_system(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 435
{ } 
#endif
# 438 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS_block(unsigned *address, unsigned compare, unsigned 
# 439
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 439
{ } 
#endif
# 442 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS_system(unsigned *address, unsigned compare, unsigned 
# 443
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 443
{ } 
#endif
# 446 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS_block(unsigned long long *address, unsigned long long 
# 447
compare, unsigned long long 
# 448
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 448
{ } 
#endif
# 451 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS_system(unsigned long long *address, unsigned long long 
# 452
compare, unsigned long long 
# 453
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 453
{ } 
#endif
# 456 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 456
{ } 
#endif
# 459 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 459
{ } 
#endif
# 462 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 462
{ } 
#endif
# 465 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 465
{ } 
#endif
# 468 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 468
{ } 
#endif
# 471 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 471
{ } 
#endif
# 474 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 474
{ } 
#endif
# 477 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 477
{ } 
#endif
# 480 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 480
{ } 
#endif
# 483 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 483
{ } 
#endif
# 486 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 486
{ } 
#endif
# 489 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 489
{ } 
#endif
# 492 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 492
{ } 
#endif
# 495 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 495
{ } 
#endif
# 498 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 498
{ } 
#endif
# 501 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 501
{ } 
#endif
# 504 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 504
{ } 
#endif
# 507 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 507
{ } 
#endif
# 510 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 510
{ } 
#endif
# 513 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 513
{ } 
#endif
# 516 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 516
{ } 
#endif
# 519 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 519
{ } 
#endif
# 522 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 522
{ } 
#endif
# 525 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 525
{ } 
#endif
# 90 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern "C" {
# 1475 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
}
# 1482
__attribute((deprecated("__ballot() is deprecated in favor of __ballot_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to" " suppress this warning)."))) __attribute__((unused)) static inline unsigned ballot(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1482
{ } 
#endif
# 1484 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline int syncthreads_count(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1484
{ } 
#endif
# 1486 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_and(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1486
{ } 
#endif
# 1488 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_or(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1488
{ } 
#endif
# 1493 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isGlobal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1493
{ } 
#endif
# 1494 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1494
{ } 
#endif
# 1495 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isConstant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1495
{ } 
#endif
# 1496 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isLocal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1496
{ } 
#endif
# 102 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __fns(unsigned mask, unsigned base, int offset) {int volatile ___ = 1;(void)mask;(void)base;(void)offset;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __barrier_sync(unsigned id) {int volatile ___ = 1;(void)id;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __barrier_sync_count(unsigned id, unsigned cnt) {int volatile ___ = 1;(void)id;(void)cnt;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __syncwarp(unsigned mask = 4294967295U) {int volatile ___ = 1;(void)mask;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __all_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __any_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __uni_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __ballot_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __activemask() {int volatile ___ = 1;::exit(___);}
#if 0
# 110
{ } 
#endif
# 119 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline int __shfl(int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 119
{ } 
#endif
# 120 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned __shfl(unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 120
{ } 
#endif
# 121 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_up(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 121
{ } 
#endif
# 122 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_up(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 122
{ } 
#endif
# 123 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_down(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 123
{ } 
#endif
# 124 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_down(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 124
{ } 
#endif
# 125 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_xor(int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 125
{ } 
#endif
# 126 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_xor(unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 126
{ } 
#endif
# 127 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline float __shfl(float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 127
{ } 
#endif
# 128 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_up(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_down(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_xor(float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 130
{ } 
#endif
# 133 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_sync(unsigned mask, int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_sync(unsigned mask, unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 134
{ } 
#endif
# 135 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_up_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_up_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 136
{ } 
#endif
# 137 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_down_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 137
{ } 
#endif
# 138 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_down_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 138
{ } 
#endif
# 139 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 139
{ } 
#endif
# 140 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 140
{ } 
#endif
# 141 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_sync(unsigned mask, float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_up_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 143
{ } 
#endif
# 144 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 144
{ } 
#endif
# 148 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl(unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long long __shfl(long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 149
{ } 
#endif
# 150 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_up(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 150
{ } 
#endif
# 151 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_up(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 151
{ } 
#endif
# 152 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_down(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 152
{ } 
#endif
# 153 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_down(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 153
{ } 
#endif
# 154 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_xor(long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 154
{ } 
#endif
# 155 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 155
{ } 
#endif
# 156 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline double __shfl(double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 156
{ } 
#endif
# 157 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_up(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 157
{ } 
#endif
# 158 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_down(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 158
{ } 
#endif
# 159 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_xor(double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 159
{ } 
#endif
# 162 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_sync(unsigned mask, long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 162
{ } 
#endif
# 163 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 163
{ } 
#endif
# 164 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_up_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_up_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 165
{ } 
#endif
# 166 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_down_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 166
{ } 
#endif
# 167 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_down_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 167
{ } 
#endif
# 168 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_xor_sync(unsigned mask, long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 168
{ } 
#endif
# 169 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_sync(unsigned mask, double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_up_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_down_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 173
{ } 
#endif
# 177 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long __shfl(long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long __shfl(unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_up(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 179
{ } 
#endif
# 180 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_up(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 180
{ } 
#endif
# 181 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_down(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 181
{ } 
#endif
# 182 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_down(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 182
{ } 
#endif
# 183 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_xor(long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 183
{ } 
#endif
# 184 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_xor(unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 184
{ } 
#endif
# 187 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_sync(unsigned mask, long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 187
{ } 
#endif
# 188 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 188
{ } 
#endif
# 189 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_up_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 190
{ } 
#endif
# 191 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_down_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 191
{ } 
#endif
# 192 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 192
{ } 
#endif
# 193 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 193
{ } 
#endif
# 194 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 194
{ } 
#endif
# 87 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 87
{ } 
#endif
# 88 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 88
{ } 
#endif
# 90 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 90
{ } 
#endif
# 91 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 91
{ } 
#endif
# 92 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 101
{ } 
#endif
# 103 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 113
{ } 
#endif
# 115 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 115
{ } 
#endif
# 116 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 116
{ } 
#endif
# 117 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 117
{ } 
#endif
# 118 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 118
{ } 
#endif
# 119 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 119
{ } 
#endif
# 123 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 123
{ } 
#endif
# 124 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 126
{ } 
#endif
# 127 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 127
{ } 
#endif
# 128 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 130
{ } 
#endif
# 131 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 131
{ } 
#endif
# 132 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 132
{ } 
#endif
# 133 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 134
{ } 
#endif
# 135 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 136
{ } 
#endif
# 137 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 137
{ } 
#endif
# 139 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 139
{ } 
#endif
# 140 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 140
{ } 
#endif
# 141 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 143
{ } 
#endif
# 144 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 144
{ } 
#endif
# 145 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 145
{ } 
#endif
# 146 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 146
{ } 
#endif
# 147 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 147
{ } 
#endif
# 148 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 149
{ } 
#endif
# 151 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 151
{ } 
#endif
# 152 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 152
{ } 
#endif
# 153 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 153
{ } 
#endif
# 154 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 154
{ } 
#endif
# 155 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 155
{ } 
#endif
# 159 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldca(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 159
{ } 
#endif
# 160 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldca(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 160
{ } 
#endif
# 162 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldca(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 162
{ } 
#endif
# 163 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldca(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 163
{ } 
#endif
# 164 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldca(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldca(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 165
{ } 
#endif
# 166 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldca(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 166
{ } 
#endif
# 167 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldca(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 167
{ } 
#endif
# 168 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldca(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 168
{ } 
#endif
# 169 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldca(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldca(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldca(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldca(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldca(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 173
{ } 
#endif
# 175 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldca(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 175
{ } 
#endif
# 176 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldca(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 176
{ } 
#endif
# 177 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldca(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldca(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldca(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 179
{ } 
#endif
# 180 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldca(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 180
{ } 
#endif
# 181 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldca(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 181
{ } 
#endif
# 182 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldca(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 182
{ } 
#endif
# 183 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldca(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 183
{ } 
#endif
# 184 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldca(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 184
{ } 
#endif
# 185 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldca(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 185
{ } 
#endif
# 187 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldca(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 187
{ } 
#endif
# 188 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldca(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 188
{ } 
#endif
# 189 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldca(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldca(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 190
{ } 
#endif
# 191 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldca(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 191
{ } 
#endif
# 195 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcs(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 195
{ } 
#endif
# 196 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcs(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 196
{ } 
#endif
# 198 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcs(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 198
{ } 
#endif
# 199 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcs(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 199
{ } 
#endif
# 200 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcs(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 200
{ } 
#endif
# 201 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcs(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 201
{ } 
#endif
# 202 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcs(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 202
{ } 
#endif
# 203 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcs(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 203
{ } 
#endif
# 204 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcs(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 204
{ } 
#endif
# 205 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcs(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 205
{ } 
#endif
# 206 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcs(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 206
{ } 
#endif
# 207 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcs(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 207
{ } 
#endif
# 208 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcs(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 208
{ } 
#endif
# 209 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcs(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 209
{ } 
#endif
# 211 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcs(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 211
{ } 
#endif
# 212 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcs(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 212
{ } 
#endif
# 213 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcs(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 213
{ } 
#endif
# 214 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcs(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 214
{ } 
#endif
# 215 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcs(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 215
{ } 
#endif
# 216 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcs(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 216
{ } 
#endif
# 217 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcs(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 217
{ } 
#endif
# 218 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcs(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 218
{ } 
#endif
# 219 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcs(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 219
{ } 
#endif
# 220 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcs(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 220
{ } 
#endif
# 221 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcs(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 221
{ } 
#endif
# 223 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcs(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 223
{ } 
#endif
# 224 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcs(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 224
{ } 
#endif
# 225 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcs(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 225
{ } 
#endif
# 226 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcs(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 226
{ } 
#endif
# 227 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcs(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 227
{ } 
#endif
# 244 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 244
{ } 
#endif
# 256 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 256
{ } 
#endif
# 269 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 269
{ } 
#endif
# 281 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 281
{ } 
#endif
# 89 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_lo(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 89
{ } 
#endif
# 90 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 90
{ } 
#endif
# 92 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_lo(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 93
{ } 
#endif
# 95 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_hi(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 96
{ } 
#endif
# 98 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_hi(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 99
{ } 
#endif
# 106 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp4a(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 107
{ } 
#endif
# 109 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp4a(char4 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp4a(uchar4 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 110
{ } 
#endif
# 93 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, float value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, double value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 100
{ } 
#endif
# 102 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, int value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, float value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, double value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 109
{ } 
#endif
# 111 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline void __nanosleep(unsigned ns) {int volatile ___ = 1;(void)ns;::exit(___);}
#if 0
# 111
{ } 
#endif
# 113 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned short atomicCAS(unsigned short *address, unsigned short compare, unsigned short val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 113
{ } 
#endif
# 114 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 115
__attribute((always_inline)) __attribute__((unused)) static inline void surf1Dread(T *res, surface< void, 1>  surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 116
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)s;(void)mode;
# 120
::exit(___);}
#if 0
# 116
{ 
# 120
} 
#endif
# 122 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 123
__attribute((always_inline)) __attribute__((unused)) static inline T surf1Dread(surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 124
{int volatile ___ = 1;(void)surf;(void)x;(void)mode;
# 130
::exit(___);}
#if 0
# 124
{ 
# 130
} 
#endif
# 132 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 133
__attribute((always_inline)) __attribute__((unused)) static inline void surf1Dread(T *res, surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 134
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)mode;
# 138
::exit(___);}
#if 0
# 134
{ 
# 138
} 
#endif
# 141 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 142
__attribute((always_inline)) __attribute__((unused)) static inline void surf2Dread(T *res, surface< void, 2>  surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 143
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)s;(void)mode;
# 147
::exit(___);}
#if 0
# 143
{ 
# 147
} 
#endif
# 149 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 150
__attribute((always_inline)) __attribute__((unused)) static inline T surf2Dread(surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 151
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)mode;
# 157
::exit(___);}
#if 0
# 151
{ 
# 157
} 
#endif
# 159 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 160
__attribute((always_inline)) __attribute__((unused)) static inline void surf2Dread(T *res, surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 161
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)mode;
# 165
::exit(___);}
#if 0
# 161
{ 
# 165
} 
#endif
# 168 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 169
__attribute((always_inline)) __attribute__((unused)) static inline void surf3Dread(T *res, surface< void, 3>  surf, int x, int y, int z, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 170
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
# 174
::exit(___);}
#if 0
# 170
{ 
# 174
} 
#endif
# 176 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 177
__attribute((always_inline)) __attribute__((unused)) static inline T surf3Dread(surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 178
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 184
::exit(___);}
#if 0
# 178
{ 
# 184
} 
#endif
# 186 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 187
__attribute((always_inline)) __attribute__((unused)) static inline void surf3Dread(T *res, surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 188
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 192
::exit(___);}
#if 0
# 188
{ 
# 192
} 
#endif
# 196 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 197
__attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredread(T *res, surface< void, 241>  surf, int x, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 198
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
# 202
::exit(___);}
#if 0
# 198
{ 
# 202
} 
#endif
# 204 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 205
__attribute((always_inline)) __attribute__((unused)) static inline T surf1DLayeredread(surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 206
{int volatile ___ = 1;(void)surf;(void)x;(void)layer;(void)mode;
# 212
::exit(___);}
#if 0
# 206
{ 
# 212
} 
#endif
# 215 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 216
__attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredread(T *res, surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 217
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)mode;
# 221
::exit(___);}
#if 0
# 217
{ 
# 221
} 
#endif
# 224 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 225
__attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredread(T *res, surface< void, 242>  surf, int x, int y, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 226
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
# 230
::exit(___);}
#if 0
# 226
{ 
# 230
} 
#endif
# 232 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 233
__attribute((always_inline)) __attribute__((unused)) static inline T surf2DLayeredread(surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 234
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 240
::exit(___);}
#if 0
# 234
{ 
# 240
} 
#endif
# 243 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 244
__attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredread(T *res, surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 245
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 249
::exit(___);}
#if 0
# 245
{ 
# 249
} 
#endif
# 252 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 253
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapread(T *res, surface< void, 12>  surf, int x, int y, int face, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 254
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
# 258
::exit(___);}
#if 0
# 254
{ 
# 258
} 
#endif
# 260 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 261
__attribute((always_inline)) __attribute__((unused)) static inline T surfCubemapread(surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 262
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 269
::exit(___);}
#if 0
# 262
{ 
# 269
} 
#endif
# 271 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 272
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapread(T *res, surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 273
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 277
::exit(___);}
#if 0
# 273
{ 
# 277
} 
#endif
# 280 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 281
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredread(T *res, surface< void, 252>  surf, int x, int y, int layerFace, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 282
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
# 286
::exit(___);}
#if 0
# 282
{ 
# 286
} 
#endif
# 288 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 289
__attribute((always_inline)) __attribute__((unused)) static inline T surfCubemapLayeredread(surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 290
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 296
::exit(___);}
#if 0
# 290
{ 
# 296
} 
#endif
# 298 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 299
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredread(T *res, surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 300
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 304
::exit(___);}
#if 0
# 300
{ 
# 304
} 
#endif
# 307 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 308
__attribute((always_inline)) __attribute__((unused)) static inline void surf1Dwrite(T val, surface< void, 1>  surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 309
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)s;(void)mode;
# 313
::exit(___);}
#if 0
# 309
{ 
# 313
} 
#endif
# 315 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 316
__attribute((always_inline)) __attribute__((unused)) static inline void surf1Dwrite(T val, surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 317
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)mode;
# 321
::exit(___);}
#if 0
# 317
{ 
# 321
} 
#endif
# 325 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 326
__attribute((always_inline)) __attribute__((unused)) static inline void surf2Dwrite(T val, surface< void, 2>  surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 327
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)s;(void)mode;
# 331
::exit(___);}
#if 0
# 327
{ 
# 331
} 
#endif
# 333 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 334
__attribute((always_inline)) __attribute__((unused)) static inline void surf2Dwrite(T val, surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 335
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)mode;
# 339
::exit(___);}
#if 0
# 335
{ 
# 339
} 
#endif
# 342 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 343
__attribute((always_inline)) __attribute__((unused)) static inline void surf3Dwrite(T val, surface< void, 3>  surf, int x, int y, int z, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 344
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
# 348
::exit(___);}
#if 0
# 344
{ 
# 348
} 
#endif
# 350 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 351
__attribute((always_inline)) __attribute__((unused)) static inline void surf3Dwrite(T val, surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 352
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 356
::exit(___);}
#if 0
# 352
{ 
# 356
} 
#endif
# 359 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 360
__attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredwrite(T val, surface< void, 241>  surf, int x, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 361
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
# 365
::exit(___);}
#if 0
# 361
{ 
# 365
} 
#endif
# 367 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 368
__attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredwrite(T val, surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 369
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)mode;
# 373
::exit(___);}
#if 0
# 369
{ 
# 373
} 
#endif
# 376 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 377
__attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredwrite(T val, surface< void, 242>  surf, int x, int y, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 378
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
# 382
::exit(___);}
#if 0
# 378
{ 
# 382
} 
#endif
# 384 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 385
__attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredwrite(T val, surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 386
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 390
::exit(___);}
#if 0
# 386
{ 
# 390
} 
#endif
# 393 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 394
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapwrite(T val, surface< void, 12>  surf, int x, int y, int face, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 395
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
# 399
::exit(___);}
#if 0
# 395
{ 
# 399
} 
#endif
# 401 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 402
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapwrite(T val, surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 403
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 407
::exit(___);}
#if 0
# 403
{ 
# 407
} 
#endif
# 411 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 412
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredwrite(T val, surface< void, 252>  surf, int x, int y, int layerFace, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 413
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
# 417
::exit(___);}
#if 0
# 413
{ 
# 417
} 
#endif
# 419 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 420
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredwrite(T val, surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 421
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 425
::exit(___);}
#if 0
# 421
{ 
# 425
} 
#endif
# 66 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 67
struct __nv_tex_rmet_ret { }; 
# 69
template<> struct __nv_tex_rmet_ret< char>  { typedef char type; }; 
# 70
template<> struct __nv_tex_rmet_ret< signed char>  { typedef signed char type; }; 
# 71
template<> struct __nv_tex_rmet_ret< unsigned char>  { typedef unsigned char type; }; 
# 72
template<> struct __nv_tex_rmet_ret< char1>  { typedef char1 type; }; 
# 73
template<> struct __nv_tex_rmet_ret< uchar1>  { typedef uchar1 type; }; 
# 74
template<> struct __nv_tex_rmet_ret< char2>  { typedef char2 type; }; 
# 75
template<> struct __nv_tex_rmet_ret< uchar2>  { typedef uchar2 type; }; 
# 76
template<> struct __nv_tex_rmet_ret< char4>  { typedef char4 type; }; 
# 77
template<> struct __nv_tex_rmet_ret< uchar4>  { typedef uchar4 type; }; 
# 79
template<> struct __nv_tex_rmet_ret< short>  { typedef short type; }; 
# 80
template<> struct __nv_tex_rmet_ret< unsigned short>  { typedef unsigned short type; }; 
# 81
template<> struct __nv_tex_rmet_ret< short1>  { typedef short1 type; }; 
# 82
template<> struct __nv_tex_rmet_ret< ushort1>  { typedef ushort1 type; }; 
# 83
template<> struct __nv_tex_rmet_ret< short2>  { typedef short2 type; }; 
# 84
template<> struct __nv_tex_rmet_ret< ushort2>  { typedef ushort2 type; }; 
# 85
template<> struct __nv_tex_rmet_ret< short4>  { typedef short4 type; }; 
# 86
template<> struct __nv_tex_rmet_ret< ushort4>  { typedef ushort4 type; }; 
# 88
template<> struct __nv_tex_rmet_ret< int>  { typedef int type; }; 
# 89
template<> struct __nv_tex_rmet_ret< unsigned>  { typedef unsigned type; }; 
# 90
template<> struct __nv_tex_rmet_ret< int1>  { typedef int1 type; }; 
# 91
template<> struct __nv_tex_rmet_ret< uint1>  { typedef uint1 type; }; 
# 92
template<> struct __nv_tex_rmet_ret< int2>  { typedef int2 type; }; 
# 93
template<> struct __nv_tex_rmet_ret< uint2>  { typedef uint2 type; }; 
# 94
template<> struct __nv_tex_rmet_ret< int4>  { typedef int4 type; }; 
# 95
template<> struct __nv_tex_rmet_ret< uint4>  { typedef uint4 type; }; 
# 107 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template<> struct __nv_tex_rmet_ret< float>  { typedef float type; }; 
# 108
template<> struct __nv_tex_rmet_ret< float1>  { typedef float1 type; }; 
# 109
template<> struct __nv_tex_rmet_ret< float2>  { typedef float2 type; }; 
# 110
template<> struct __nv_tex_rmet_ret< float4>  { typedef float4 type; }; 
# 113
template< class T> struct __nv_tex_rmet_cast { typedef T *type; }; 
# 125 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 126
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1Dfetch(texture< T, 1, cudaReadModeElementType>  t, int x) 
# 127
{int volatile ___ = 1;(void)t;(void)x;
# 133
::exit(___);}
#if 0
# 127
{ 
# 133
} 
#endif
# 135 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 136
struct __nv_tex_rmnf_ret { }; 
# 138
template<> struct __nv_tex_rmnf_ret< char>  { typedef float type; }; 
# 139
template<> struct __nv_tex_rmnf_ret< signed char>  { typedef float type; }; 
# 140
template<> struct __nv_tex_rmnf_ret< unsigned char>  { typedef float type; }; 
# 141
template<> struct __nv_tex_rmnf_ret< short>  { typedef float type; }; 
# 142
template<> struct __nv_tex_rmnf_ret< unsigned short>  { typedef float type; }; 
# 143
template<> struct __nv_tex_rmnf_ret< char1>  { typedef float1 type; }; 
# 144
template<> struct __nv_tex_rmnf_ret< uchar1>  { typedef float1 type; }; 
# 145
template<> struct __nv_tex_rmnf_ret< short1>  { typedef float1 type; }; 
# 146
template<> struct __nv_tex_rmnf_ret< ushort1>  { typedef float1 type; }; 
# 147
template<> struct __nv_tex_rmnf_ret< char2>  { typedef float2 type; }; 
# 148
template<> struct __nv_tex_rmnf_ret< uchar2>  { typedef float2 type; }; 
# 149
template<> struct __nv_tex_rmnf_ret< short2>  { typedef float2 type; }; 
# 150
template<> struct __nv_tex_rmnf_ret< ushort2>  { typedef float2 type; }; 
# 151
template<> struct __nv_tex_rmnf_ret< char4>  { typedef float4 type; }; 
# 152
template<> struct __nv_tex_rmnf_ret< uchar4>  { typedef float4 type; }; 
# 153
template<> struct __nv_tex_rmnf_ret< short4>  { typedef float4 type; }; 
# 154
template<> struct __nv_tex_rmnf_ret< ushort4>  { typedef float4 type; }; 
# 156
template< class T> 
# 157
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1Dfetch(texture< T, 1, cudaReadModeNormalizedFloat>  t, int x) 
# 158
{int volatile ___ = 1;(void)t;(void)x;
# 165
::exit(___);}
#if 0
# 158
{ 
# 165
} 
#endif
# 168 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 169
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1D(texture< T, 1, cudaReadModeElementType>  t, float x) 
# 170
{int volatile ___ = 1;(void)t;(void)x;
# 176
::exit(___);}
#if 0
# 170
{ 
# 176
} 
#endif
# 178 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 179
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1D(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x) 
# 180
{int volatile ___ = 1;(void)t;(void)x;
# 187
::exit(___);}
#if 0
# 180
{ 
# 187
} 
#endif
# 191 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 192
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2D(texture< T, 2, cudaReadModeElementType>  t, float x, float y) 
# 193
{int volatile ___ = 1;(void)t;(void)x;(void)y;
# 200
::exit(___);}
#if 0
# 193
{ 
# 200
} 
#endif
# 202 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 203
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2D(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y) 
# 204
{int volatile ___ = 1;(void)t;(void)x;(void)y;
# 211
::exit(___);}
#if 0
# 204
{ 
# 211
} 
#endif
# 215 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 216
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayered(texture< T, 241, cudaReadModeElementType>  t, float x, int layer) 
# 217
{int volatile ___ = 1;(void)t;(void)x;(void)layer;
# 223
::exit(___);}
#if 0
# 217
{ 
# 223
} 
#endif
# 225 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 226
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayered(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer) 
# 227
{int volatile ___ = 1;(void)t;(void)x;(void)layer;
# 234
::exit(___);}
#if 0
# 227
{ 
# 234
} 
#endif
# 238 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 239
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayered(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer) 
# 240
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;
# 246
::exit(___);}
#if 0
# 240
{ 
# 246
} 
#endif
# 248 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 249
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayered(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer) 
# 250
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;
# 257
::exit(___);}
#if 0
# 250
{ 
# 257
} 
#endif
# 260 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 261
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3D(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z) 
# 262
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 268
::exit(___);}
#if 0
# 262
{ 
# 268
} 
#endif
# 270 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 271
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3D(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z) 
# 272
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 279
::exit(___);}
#if 0
# 272
{ 
# 279
} 
#endif
# 282 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 283
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemap(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z) 
# 284
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 290
::exit(___);}
#if 0
# 284
{ 
# 290
} 
#endif
# 292 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 293
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemap(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z) 
# 294
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 301
::exit(___);}
#if 0
# 294
{ 
# 301
} 
#endif
# 304 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 305
struct __nv_tex2dgather_ret { }; 
# 306
template<> struct __nv_tex2dgather_ret< char>  { typedef char4 type; }; 
# 307
template<> struct __nv_tex2dgather_ret< signed char>  { typedef char4 type; }; 
# 308
template<> struct __nv_tex2dgather_ret< char1>  { typedef char4 type; }; 
# 309
template<> struct __nv_tex2dgather_ret< char2>  { typedef char4 type; }; 
# 310
template<> struct __nv_tex2dgather_ret< char3>  { typedef char4 type; }; 
# 311
template<> struct __nv_tex2dgather_ret< char4>  { typedef char4 type; }; 
# 312
template<> struct __nv_tex2dgather_ret< unsigned char>  { typedef uchar4 type; }; 
# 313
template<> struct __nv_tex2dgather_ret< uchar1>  { typedef uchar4 type; }; 
# 314
template<> struct __nv_tex2dgather_ret< uchar2>  { typedef uchar4 type; }; 
# 315
template<> struct __nv_tex2dgather_ret< uchar3>  { typedef uchar4 type; }; 
# 316
template<> struct __nv_tex2dgather_ret< uchar4>  { typedef uchar4 type; }; 
# 318
template<> struct __nv_tex2dgather_ret< short>  { typedef short4 type; }; 
# 319
template<> struct __nv_tex2dgather_ret< short1>  { typedef short4 type; }; 
# 320
template<> struct __nv_tex2dgather_ret< short2>  { typedef short4 type; }; 
# 321
template<> struct __nv_tex2dgather_ret< short3>  { typedef short4 type; }; 
# 322
template<> struct __nv_tex2dgather_ret< short4>  { typedef short4 type; }; 
# 323
template<> struct __nv_tex2dgather_ret< unsigned short>  { typedef ushort4 type; }; 
# 324
template<> struct __nv_tex2dgather_ret< ushort1>  { typedef ushort4 type; }; 
# 325
template<> struct __nv_tex2dgather_ret< ushort2>  { typedef ushort4 type; }; 
# 326
template<> struct __nv_tex2dgather_ret< ushort3>  { typedef ushort4 type; }; 
# 327
template<> struct __nv_tex2dgather_ret< ushort4>  { typedef ushort4 type; }; 
# 329
template<> struct __nv_tex2dgather_ret< int>  { typedef int4 type; }; 
# 330
template<> struct __nv_tex2dgather_ret< int1>  { typedef int4 type; }; 
# 331
template<> struct __nv_tex2dgather_ret< int2>  { typedef int4 type; }; 
# 332
template<> struct __nv_tex2dgather_ret< int3>  { typedef int4 type; }; 
# 333
template<> struct __nv_tex2dgather_ret< int4>  { typedef int4 type; }; 
# 334
template<> struct __nv_tex2dgather_ret< unsigned>  { typedef uint4 type; }; 
# 335
template<> struct __nv_tex2dgather_ret< uint1>  { typedef uint4 type; }; 
# 336
template<> struct __nv_tex2dgather_ret< uint2>  { typedef uint4 type; }; 
# 337
template<> struct __nv_tex2dgather_ret< uint3>  { typedef uint4 type; }; 
# 338
template<> struct __nv_tex2dgather_ret< uint4>  { typedef uint4 type; }; 
# 340
template<> struct __nv_tex2dgather_ret< float>  { typedef float4 type; }; 
# 341
template<> struct __nv_tex2dgather_ret< float1>  { typedef float4 type; }; 
# 342
template<> struct __nv_tex2dgather_ret< float2>  { typedef float4 type; }; 
# 343
template<> struct __nv_tex2dgather_ret< float3>  { typedef float4 type; }; 
# 344
template<> struct __nv_tex2dgather_ret< float4>  { typedef float4 type; }; 
# 346
template< class T> 
# 347
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex2dgather_ret< T> ::type tex2Dgather(texture< T, 2, cudaReadModeElementType>  t, float x, float y, int comp = 0) 
# 348
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;
# 355
::exit(___);}
#if 0
# 348
{ 
# 355
} 
#endif
# 358 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> struct __nv_tex2dgather_rmnf_ret { }; 
# 359
template<> struct __nv_tex2dgather_rmnf_ret< char>  { typedef float4 type; }; 
# 360
template<> struct __nv_tex2dgather_rmnf_ret< signed char>  { typedef float4 type; }; 
# 361
template<> struct __nv_tex2dgather_rmnf_ret< unsigned char>  { typedef float4 type; }; 
# 362
template<> struct __nv_tex2dgather_rmnf_ret< char1>  { typedef float4 type; }; 
# 363
template<> struct __nv_tex2dgather_rmnf_ret< uchar1>  { typedef float4 type; }; 
# 364
template<> struct __nv_tex2dgather_rmnf_ret< char2>  { typedef float4 type; }; 
# 365
template<> struct __nv_tex2dgather_rmnf_ret< uchar2>  { typedef float4 type; }; 
# 366
template<> struct __nv_tex2dgather_rmnf_ret< char3>  { typedef float4 type; }; 
# 367
template<> struct __nv_tex2dgather_rmnf_ret< uchar3>  { typedef float4 type; }; 
# 368
template<> struct __nv_tex2dgather_rmnf_ret< char4>  { typedef float4 type; }; 
# 369
template<> struct __nv_tex2dgather_rmnf_ret< uchar4>  { typedef float4 type; }; 
# 370
template<> struct __nv_tex2dgather_rmnf_ret< signed short>  { typedef float4 type; }; 
# 371
template<> struct __nv_tex2dgather_rmnf_ret< unsigned short>  { typedef float4 type; }; 
# 372
template<> struct __nv_tex2dgather_rmnf_ret< short1>  { typedef float4 type; }; 
# 373
template<> struct __nv_tex2dgather_rmnf_ret< ushort1>  { typedef float4 type; }; 
# 374
template<> struct __nv_tex2dgather_rmnf_ret< short2>  { typedef float4 type; }; 
# 375
template<> struct __nv_tex2dgather_rmnf_ret< ushort2>  { typedef float4 type; }; 
# 376
template<> struct __nv_tex2dgather_rmnf_ret< short3>  { typedef float4 type; }; 
# 377
template<> struct __nv_tex2dgather_rmnf_ret< ushort3>  { typedef float4 type; }; 
# 378
template<> struct __nv_tex2dgather_rmnf_ret< short4>  { typedef float4 type; }; 
# 379
template<> struct __nv_tex2dgather_rmnf_ret< ushort4>  { typedef float4 type; }; 
# 381
template< class T> 
# 382
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex2dgather_rmnf_ret< T> ::type tex2Dgather(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, int comp = 0) 
# 383
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;
# 390
::exit(___);}
#if 0
# 383
{ 
# 390
} 
#endif
# 394 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 395
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLod(texture< T, 1, cudaReadModeElementType>  t, float x, float level) 
# 396
{int volatile ___ = 1;(void)t;(void)x;(void)level;
# 402
::exit(___);}
#if 0
# 396
{ 
# 402
} 
#endif
# 404 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 405
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLod(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float level) 
# 406
{int volatile ___ = 1;(void)t;(void)x;(void)level;
# 413
::exit(___);}
#if 0
# 406
{ 
# 413
} 
#endif
# 416 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 417
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLod(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float level) 
# 418
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;
# 424
::exit(___);}
#if 0
# 418
{ 
# 424
} 
#endif
# 426 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 427
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLod(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float level) 
# 428
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;
# 435
::exit(___);}
#if 0
# 428
{ 
# 435
} 
#endif
# 438 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 439
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayeredLod(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float level) 
# 440
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;
# 446
::exit(___);}
#if 0
# 440
{ 
# 446
} 
#endif
# 448 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 449
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayeredLod(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float level) 
# 450
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;
# 457
::exit(___);}
#if 0
# 450
{ 
# 457
} 
#endif
# 460 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 461
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayeredLod(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float level) 
# 462
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;
# 468
::exit(___);}
#if 0
# 462
{ 
# 468
} 
#endif
# 470 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 471
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayeredLod(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float level) 
# 472
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;
# 479
::exit(___);}
#if 0
# 472
{ 
# 479
} 
#endif
# 482 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 483
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3DLod(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float level) 
# 484
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 490
::exit(___);}
#if 0
# 484
{ 
# 490
} 
#endif
# 492 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 493
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3DLod(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) 
# 494
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 501
::exit(___);}
#if 0
# 494
{ 
# 501
} 
#endif
# 504 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 505
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLod(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float level) 
# 506
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 512
::exit(___);}
#if 0
# 506
{ 
# 512
} 
#endif
# 514 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 515
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLod(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) 
# 516
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 523
::exit(___);}
#if 0
# 516
{ 
# 523
} 
#endif
# 527 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 528
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayered(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer) 
# 529
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;
# 535
::exit(___);}
#if 0
# 529
{ 
# 535
} 
#endif
# 537 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 538
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayered(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer) 
# 539
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;
# 546
::exit(___);}
#if 0
# 539
{ 
# 546
} 
#endif
# 550 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 551
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayeredLod(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float level) 
# 552
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 558
::exit(___);}
#if 0
# 552
{ 
# 558
} 
#endif
# 560 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 561
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayeredLod(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float level) 
# 562
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 569
::exit(___);}
#if 0
# 562
{ 
# 569
} 
#endif
# 573 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 574
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapGrad(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 575
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 581
::exit(___);}
#if 0
# 575
{ 
# 581
} 
#endif
# 583 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 584
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapGrad(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 585
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 592
::exit(___);}
#if 0
# 585
{ 
# 592
} 
#endif
# 596 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 597
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayeredGrad(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 598
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 604
::exit(___);}
#if 0
# 598
{ 
# 604
} 
#endif
# 606 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 607
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayeredGrad(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 608
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 615
::exit(___);}
#if 0
# 608
{ 
# 615
} 
#endif
# 619 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 620
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DGrad(texture< T, 1, cudaReadModeElementType>  t, float x, float dPdx, float dPdy) 
# 621
{int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;
# 627
::exit(___);}
#if 0
# 621
{ 
# 627
} 
#endif
# 629 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 630
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DGrad(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float dPdx, float dPdy) 
# 631
{int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;
# 638
::exit(___);}
#if 0
# 631
{ 
# 638
} 
#endif
# 642 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 643
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DGrad(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float2 dPdx, float2 dPdy) 
# 644
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 650
::exit(___);}
#if 0
# 644
{ 
# 650
} 
#endif
# 652 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 653
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DGrad(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float2 dPdx, float2 dPdy) 
# 654
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 661
::exit(___);}
#if 0
# 654
{ 
# 661
} 
#endif
# 664 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 665
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayeredGrad(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float dPdx, float dPdy) 
# 666
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 672
::exit(___);}
#if 0
# 666
{ 
# 672
} 
#endif
# 674 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 675
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayeredGrad(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float dPdx, float dPdy) 
# 676
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 683
::exit(___);}
#if 0
# 676
{ 
# 683
} 
#endif
# 686 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 687
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayeredGrad(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 688
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 694
::exit(___);}
#if 0
# 688
{ 
# 694
} 
#endif
# 696 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 697
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayeredGrad(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 698
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 705
::exit(___);}
#if 0
# 698
{ 
# 705
} 
#endif
# 708 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 709
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3DGrad(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 710
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 716
::exit(___);}
#if 0
# 710
{ 
# 716
} 
#endif
# 718 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 719
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3DGrad(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 720
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 727
::exit(___);}
#if 0
# 720
{ 
# 727
} 
#endif
# 60 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> struct __nv_itex_trait { }; 
# 61
template<> struct __nv_itex_trait< char>  { typedef void type; }; 
# 62
template<> struct __nv_itex_trait< signed char>  { typedef void type; }; 
# 63
template<> struct __nv_itex_trait< char1>  { typedef void type; }; 
# 64
template<> struct __nv_itex_trait< char2>  { typedef void type; }; 
# 65
template<> struct __nv_itex_trait< char4>  { typedef void type; }; 
# 66
template<> struct __nv_itex_trait< unsigned char>  { typedef void type; }; 
# 67
template<> struct __nv_itex_trait< uchar1>  { typedef void type; }; 
# 68
template<> struct __nv_itex_trait< uchar2>  { typedef void type; }; 
# 69
template<> struct __nv_itex_trait< uchar4>  { typedef void type; }; 
# 70
template<> struct __nv_itex_trait< short>  { typedef void type; }; 
# 71
template<> struct __nv_itex_trait< short1>  { typedef void type; }; 
# 72
template<> struct __nv_itex_trait< short2>  { typedef void type; }; 
# 73
template<> struct __nv_itex_trait< short4>  { typedef void type; }; 
# 74
template<> struct __nv_itex_trait< unsigned short>  { typedef void type; }; 
# 75
template<> struct __nv_itex_trait< ushort1>  { typedef void type; }; 
# 76
template<> struct __nv_itex_trait< ushort2>  { typedef void type; }; 
# 77
template<> struct __nv_itex_trait< ushort4>  { typedef void type; }; 
# 78
template<> struct __nv_itex_trait< int>  { typedef void type; }; 
# 79
template<> struct __nv_itex_trait< int1>  { typedef void type; }; 
# 80
template<> struct __nv_itex_trait< int2>  { typedef void type; }; 
# 81
template<> struct __nv_itex_trait< int4>  { typedef void type; }; 
# 82
template<> struct __nv_itex_trait< unsigned>  { typedef void type; }; 
# 83
template<> struct __nv_itex_trait< uint1>  { typedef void type; }; 
# 84
template<> struct __nv_itex_trait< uint2>  { typedef void type; }; 
# 85
template<> struct __nv_itex_trait< uint4>  { typedef void type; }; 
# 96 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template<> struct __nv_itex_trait< float>  { typedef void type; }; 
# 97
template<> struct __nv_itex_trait< float1>  { typedef void type; }; 
# 98
template<> struct __nv_itex_trait< float2>  { typedef void type; }; 
# 99
template<> struct __nv_itex_trait< float4>  { typedef void type; }; 
# 103
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 104
tex1Dfetch(T *ptr, cudaTextureObject_t obj, int x) 
# 105
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 109
::exit(___);}
#if 0
# 105
{ 
# 109
} 
#endif
# 111 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 112
tex1Dfetch(cudaTextureObject_t texObject, int x) 
# 113
{int volatile ___ = 1;(void)texObject;(void)x;
# 119
::exit(___);}
#if 0
# 113
{ 
# 119
} 
#endif
# 121 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 122
tex1D(T *ptr, cudaTextureObject_t obj, float x) 
# 123
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 127
::exit(___);}
#if 0
# 123
{ 
# 127
} 
#endif
# 130 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 131
tex1D(cudaTextureObject_t texObject, float x) 
# 132
{int volatile ___ = 1;(void)texObject;(void)x;
# 138
::exit(___);}
#if 0
# 132
{ 
# 138
} 
#endif
# 141 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 142
tex2D(T *ptr, cudaTextureObject_t obj, float x, float y) 
# 143
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;
# 147
::exit(___);}
#if 0
# 143
{ 
# 147
} 
#endif
# 149 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 150
tex2D(cudaTextureObject_t texObject, float x, float y) 
# 151
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;
# 157
::exit(___);}
#if 0
# 151
{ 
# 157
} 
#endif
# 159 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 160
tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 161
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 165
::exit(___);}
#if 0
# 161
{ 
# 165
} 
#endif
# 167 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 168
tex3D(cudaTextureObject_t texObject, float x, float y, float z) 
# 169
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 175
::exit(___);}
#if 0
# 169
{ 
# 175
} 
#endif
# 177 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 178
tex1DLayered(T *ptr, cudaTextureObject_t obj, float x, int layer) 
# 179
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;
# 183
::exit(___);}
#if 0
# 179
{ 
# 183
} 
#endif
# 185 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 186
tex1DLayered(cudaTextureObject_t texObject, float x, int layer) 
# 187
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;
# 193
::exit(___);}
#if 0
# 187
{ 
# 193
} 
#endif
# 195 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 196
tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer) 
# 197
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;
# 201
::exit(___);}
#if 0
# 197
{ 
# 201
} 
#endif
# 203 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 204
tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer) 
# 205
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;
# 211
::exit(___);}
#if 0
# 205
{ 
# 211
} 
#endif
# 214 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 215
texCubemap(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 216
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 220
::exit(___);}
#if 0
# 216
{ 
# 220
} 
#endif
# 223 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 224
texCubemap(cudaTextureObject_t texObject, float x, float y, float z) 
# 225
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 231
::exit(___);}
#if 0
# 225
{ 
# 231
} 
#endif
# 234 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 235
texCubemapLayered(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer) 
# 236
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;
# 240
::exit(___);}
#if 0
# 236
{ 
# 240
} 
#endif
# 242 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 243
texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer) 
# 244
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;
# 250
::exit(___);}
#if 0
# 244
{ 
# 250
} 
#endif
# 252 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 253
tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, int comp = 0) 
# 254
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)comp;
# 258
::exit(___);}
#if 0
# 254
{ 
# 258
} 
#endif
# 260 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 261
tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0) 
# 262
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)comp;
# 268
::exit(___);}
#if 0
# 262
{ 
# 268
} 
#endif
# 272 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 273
tex1DLod(T *ptr, cudaTextureObject_t obj, float x, float level) 
# 274
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)level;
# 278
::exit(___);}
#if 0
# 274
{ 
# 278
} 
#endif
# 280 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 281
tex1DLod(cudaTextureObject_t texObject, float x, float level) 
# 282
{int volatile ___ = 1;(void)texObject;(void)x;(void)level;
# 288
::exit(___);}
#if 0
# 282
{ 
# 288
} 
#endif
# 291 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 292
tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level) 
# 293
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;
# 297
::exit(___);}
#if 0
# 293
{ 
# 297
} 
#endif
# 299 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 300
tex2DLod(cudaTextureObject_t texObject, float x, float y, float level) 
# 301
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;
# 307
::exit(___);}
#if 0
# 301
{ 
# 307
} 
#endif
# 310 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 311
tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 312
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 316
::exit(___);}
#if 0
# 312
{ 
# 316
} 
#endif
# 318 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 319
tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 320
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 326
::exit(___);}
#if 0
# 320
{ 
# 326
} 
#endif
# 329 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 330
tex1DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, int layer, float level) 
# 331
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)level;
# 335
::exit(___);}
#if 0
# 331
{ 
# 335
} 
#endif
# 337 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 338
tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level) 
# 339
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)level;
# 345
::exit(___);}
#if 0
# 339
{ 
# 345
} 
#endif
# 348 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 349
tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level) 
# 350
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;
# 354
::exit(___);}
#if 0
# 350
{ 
# 354
} 
#endif
# 356 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 357
tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level) 
# 358
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;
# 364
::exit(___);}
#if 0
# 358
{ 
# 364
} 
#endif
# 367 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 368
texCubemapLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 369
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 373
::exit(___);}
#if 0
# 369
{ 
# 373
} 
#endif
# 375 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 376
texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 377
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 383
::exit(___);}
#if 0
# 377
{ 
# 383
} 
#endif
# 386 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 387
texCubemapGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 388
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 392
::exit(___);}
#if 0
# 388
{ 
# 392
} 
#endif
# 394 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 395
texCubemapGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 396
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 402
::exit(___);}
#if 0
# 396
{ 
# 402
} 
#endif
# 404 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 405
texCubemapLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float level) 
# 406
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 410
::exit(___);}
#if 0
# 406
{ 
# 410
} 
#endif
# 412 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 413
texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level) 
# 414
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 420
::exit(___);}
#if 0
# 414
{ 
# 420
} 
#endif
# 422 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 423
tex1DGrad(T *ptr, cudaTextureObject_t obj, float x, float dPdx, float dPdy) 
# 424
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)dPdx;(void)dPdy;
# 428
::exit(___);}
#if 0
# 424
{ 
# 428
} 
#endif
# 430 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 431
tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy) 
# 432
{int volatile ___ = 1;(void)texObject;(void)x;(void)dPdx;(void)dPdy;
# 438
::exit(___);}
#if 0
# 432
{ 
# 438
} 
#endif
# 441 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 442
tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy) 
# 443
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 448
::exit(___);}
#if 0
# 443
{ 
# 448
} 
#endif
# 450 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 451
tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy) 
# 452
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 458
::exit(___);}
#if 0
# 452
{ 
# 458
} 
#endif
# 461 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 462
tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 463
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 467
::exit(___);}
#if 0
# 463
{ 
# 467
} 
#endif
# 469 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 470
tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 471
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 477
::exit(___);}
#if 0
# 471
{ 
# 477
} 
#endif
# 480 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 481
tex1DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy) 
# 482
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 486
::exit(___);}
#if 0
# 482
{ 
# 486
} 
#endif
# 488 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 489
tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy) 
# 490
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 496
::exit(___);}
#if 0
# 490
{ 
# 496
} 
#endif
# 499 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 500
tex2DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 501
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 505
::exit(___);}
#if 0
# 501
{ 
# 505
} 
#endif
# 507 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 508
tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 509
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 515
::exit(___);}
#if 0
# 509
{ 
# 515
} 
#endif
# 518 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 519
texCubemapLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 520
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 524
::exit(___);}
#if 0
# 520
{ 
# 524
} 
#endif
# 526 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 527
texCubemapLayeredGrad(cudaTextureObject_t texObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 528
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 534
::exit(___);}
#if 0
# 528
{ 
# 534
} 
#endif
# 59 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> struct __nv_isurf_trait { }; 
# 60
template<> struct __nv_isurf_trait< char>  { typedef void type; }; 
# 61
template<> struct __nv_isurf_trait< signed char>  { typedef void type; }; 
# 62
template<> struct __nv_isurf_trait< char1>  { typedef void type; }; 
# 63
template<> struct __nv_isurf_trait< unsigned char>  { typedef void type; }; 
# 64
template<> struct __nv_isurf_trait< uchar1>  { typedef void type; }; 
# 65
template<> struct __nv_isurf_trait< short>  { typedef void type; }; 
# 66
template<> struct __nv_isurf_trait< short1>  { typedef void type; }; 
# 67
template<> struct __nv_isurf_trait< unsigned short>  { typedef void type; }; 
# 68
template<> struct __nv_isurf_trait< ushort1>  { typedef void type; }; 
# 69
template<> struct __nv_isurf_trait< int>  { typedef void type; }; 
# 70
template<> struct __nv_isurf_trait< int1>  { typedef void type; }; 
# 71
template<> struct __nv_isurf_trait< unsigned>  { typedef void type; }; 
# 72
template<> struct __nv_isurf_trait< uint1>  { typedef void type; }; 
# 73
template<> struct __nv_isurf_trait< long long>  { typedef void type; }; 
# 74
template<> struct __nv_isurf_trait< longlong1>  { typedef void type; }; 
# 75
template<> struct __nv_isurf_trait< unsigned long long>  { typedef void type; }; 
# 76
template<> struct __nv_isurf_trait< ulonglong1>  { typedef void type; }; 
# 77
template<> struct __nv_isurf_trait< float>  { typedef void type; }; 
# 78
template<> struct __nv_isurf_trait< float1>  { typedef void type; }; 
# 80
template<> struct __nv_isurf_trait< char2>  { typedef void type; }; 
# 81
template<> struct __nv_isurf_trait< uchar2>  { typedef void type; }; 
# 82
template<> struct __nv_isurf_trait< short2>  { typedef void type; }; 
# 83
template<> struct __nv_isurf_trait< ushort2>  { typedef void type; }; 
# 84
template<> struct __nv_isurf_trait< int2>  { typedef void type; }; 
# 85
template<> struct __nv_isurf_trait< uint2>  { typedef void type; }; 
# 86
template<> struct __nv_isurf_trait< longlong2>  { typedef void type; }; 
# 87
template<> struct __nv_isurf_trait< ulonglong2>  { typedef void type; }; 
# 88
template<> struct __nv_isurf_trait< float2>  { typedef void type; }; 
# 90
template<> struct __nv_isurf_trait< char4>  { typedef void type; }; 
# 91
template<> struct __nv_isurf_trait< uchar4>  { typedef void type; }; 
# 92
template<> struct __nv_isurf_trait< short4>  { typedef void type; }; 
# 93
template<> struct __nv_isurf_trait< ushort4>  { typedef void type; }; 
# 94
template<> struct __nv_isurf_trait< int4>  { typedef void type; }; 
# 95
template<> struct __nv_isurf_trait< uint4>  { typedef void type; }; 
# 96
template<> struct __nv_isurf_trait< float4>  { typedef void type; }; 
# 99
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 100
surf1Dread(T *ptr, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 101
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)mode;
# 105
::exit(___);}
#if 0
# 101
{ 
# 105
} 
#endif
# 107 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 108
surf1Dread(cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 109
{int volatile ___ = 1;(void)surfObject;(void)x;(void)boundaryMode;
# 115
::exit(___);}
#if 0
# 109
{ 
# 115
} 
#endif
# 117 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 118
surf2Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 119
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)mode;
# 123
::exit(___);}
#if 0
# 119
{ 
# 123
} 
#endif
# 125 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 126
surf2Dread(cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 127
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)boundaryMode;
# 133
::exit(___);}
#if 0
# 127
{ 
# 133
} 
#endif
# 136 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 137
surf3Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 138
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 142
::exit(___);}
#if 0
# 138
{ 
# 142
} 
#endif
# 144 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 145
surf3Dread(cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 146
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)z;(void)boundaryMode;
# 152
::exit(___);}
#if 0
# 146
{ 
# 152
} 
#endif
# 154 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 155
surf1DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 156
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)mode;
# 160
::exit(___);}
#if 0
# 156
{ 
# 160
} 
#endif
# 162 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 163
surf1DLayeredread(cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 164
{int volatile ___ = 1;(void)surfObject;(void)x;(void)layer;(void)boundaryMode;
# 170
::exit(___);}
#if 0
# 164
{ 
# 170
} 
#endif
# 172 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 173
surf2DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 174
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 178
::exit(___);}
#if 0
# 174
{ 
# 178
} 
#endif
# 180 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 181
surf2DLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 182
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layer;(void)boundaryMode;
# 188
::exit(___);}
#if 0
# 182
{ 
# 188
} 
#endif
# 190 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 191
surfCubemapread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 192
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 196
::exit(___);}
#if 0
# 192
{ 
# 196
} 
#endif
# 198 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 199
surfCubemapread(cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 200
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)face;(void)boundaryMode;
# 206
::exit(___);}
#if 0
# 200
{ 
# 206
} 
#endif
# 208 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 209
surfCubemapLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 210
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 214
::exit(___);}
#if 0
# 210
{ 
# 214
} 
#endif
# 216 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 217
surfCubemapLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 218
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layerface;(void)boundaryMode;
# 224
::exit(___);}
#if 0
# 218
{ 
# 224
} 
#endif
# 226 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 227
surf1Dwrite(T val, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 228
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)mode;
# 232
::exit(___);}
#if 0
# 228
{ 
# 232
} 
#endif
# 234 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 235
surf2Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 236
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)mode;
# 240
::exit(___);}
#if 0
# 236
{ 
# 240
} 
#endif
# 242 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 243
surf3Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 244
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 248
::exit(___);}
#if 0
# 244
{ 
# 248
} 
#endif
# 250 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 251
surf1DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 252
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)layer;(void)mode;
# 256
::exit(___);}
#if 0
# 252
{ 
# 256
} 
#endif
# 258 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 259
surf2DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 260
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 264
::exit(___);}
#if 0
# 260
{ 
# 264
} 
#endif
# 266 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 267
surfCubemapwrite(T val, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 268
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 272
::exit(___);}
#if 0
# 268
{ 
# 272
} 
#endif
# 274 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 275
surfCubemapLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 276
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 280
::exit(___);}
#if 0
# 276
{ 
# 280
} 
#endif
# 3296 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/crt/device_functions.h"
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, CUstream_st * stream = 0); 
# 68 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/device_launch_parameters.h"
extern "C" {
# 71
extern const uint3 __device_builtin_variable_threadIdx; 
# 72
extern const uint3 __device_builtin_variable_blockIdx; 
# 73
extern const dim3 __device_builtin_variable_blockDim; 
# 74
extern const dim3 __device_builtin_variable_gridDim; 
# 75
extern const int __device_builtin_variable_warpSize; 
# 80
}
# 199 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 200
cudaLaunchKernel(const T *
# 201
func, dim3 
# 202
gridDim, dim3 
# 203
blockDim, void **
# 204
args, size_t 
# 205
sharedMem = 0, cudaStream_t 
# 206
stream = 0) 
# 208
{ 
# 209
return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 210
} 
# 261 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 262
cudaLaunchCooperativeKernel(const T *
# 263
func, dim3 
# 264
gridDim, dim3 
# 265
blockDim, void **
# 266
args, size_t 
# 267
sharedMem = 0, cudaStream_t 
# 268
stream = 0) 
# 270
{ 
# 271
return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 272
} 
# 305 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
static inline cudaError_t cudaEventCreate(cudaEvent_t *
# 306
event, unsigned 
# 307
flags) 
# 309
{ 
# 310
return ::cudaEventCreateWithFlags(event, flags); 
# 311
} 
# 370 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
static inline cudaError_t cudaMallocHost(void **
# 371
ptr, size_t 
# 372
size, unsigned 
# 373
flags) 
# 375
{ 
# 376
return ::cudaHostAlloc(ptr, size, flags); 
# 377
} 
# 379
template< class T> static inline cudaError_t 
# 380
cudaHostAlloc(T **
# 381
ptr, size_t 
# 382
size, unsigned 
# 383
flags) 
# 385
{ 
# 386
return ::cudaHostAlloc((void **)((void *)ptr), size, flags); 
# 387
} 
# 389
template< class T> static inline cudaError_t 
# 390
cudaHostGetDevicePointer(T **
# 391
pDevice, void *
# 392
pHost, unsigned 
# 393
flags) 
# 395
{ 
# 396
return ::cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags); 
# 397
} 
# 499 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 500
cudaMallocManaged(T **
# 501
devPtr, size_t 
# 502
size, unsigned 
# 503
flags = 1) 
# 505
{ 
# 506
return ::cudaMallocManaged((void **)((void *)devPtr), size, flags); 
# 507
} 
# 589 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 590
cudaStreamAttachMemAsync(cudaStream_t 
# 591
stream, T *
# 592
devPtr, size_t 
# 593
length = 0, unsigned 
# 594
flags = 4) 
# 596
{ 
# 597
return ::cudaStreamAttachMemAsync(stream, (void *)devPtr, length, flags); 
# 598
} 
# 600
template< class T> inline cudaError_t 
# 601
cudaMalloc(T **
# 602
devPtr, size_t 
# 603
size) 
# 605
{ 
# 606
return ::cudaMalloc((void **)((void *)devPtr), size); 
# 607
} 
# 609
template< class T> static inline cudaError_t 
# 610
cudaMallocHost(T **
# 611
ptr, size_t 
# 612
size, unsigned 
# 613
flags = 0) 
# 615
{ 
# 616
return cudaMallocHost((void **)((void *)ptr), size, flags); 
# 617
} 
# 619
template< class T> static inline cudaError_t 
# 620
cudaMallocPitch(T **
# 621
devPtr, size_t *
# 622
pitch, size_t 
# 623
width, size_t 
# 624
height) 
# 626
{ 
# 627
return ::cudaMallocPitch((void **)((void *)devPtr), pitch, width, height); 
# 628
} 
# 667 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 668
cudaMemcpyToSymbol(const T &
# 669
symbol, const void *
# 670
src, size_t 
# 671
count, size_t 
# 672
offset = 0, cudaMemcpyKind 
# 673
kind = cudaMemcpyHostToDevice) 
# 675
{ 
# 676
return ::cudaMemcpyToSymbol((const void *)(&symbol), src, count, offset, kind); 
# 677
} 
# 721 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 722
cudaMemcpyToSymbolAsync(const T &
# 723
symbol, const void *
# 724
src, size_t 
# 725
count, size_t 
# 726
offset = 0, cudaMemcpyKind 
# 727
kind = cudaMemcpyHostToDevice, cudaStream_t 
# 728
stream = 0) 
# 730
{ 
# 731
return ::cudaMemcpyToSymbolAsync((const void *)(&symbol), src, count, offset, kind, stream); 
# 732
} 
# 769 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 770
cudaMemcpyFromSymbol(void *
# 771
dst, const T &
# 772
symbol, size_t 
# 773
count, size_t 
# 774
offset = 0, cudaMemcpyKind 
# 775
kind = cudaMemcpyDeviceToHost) 
# 777
{ 
# 778
return ::cudaMemcpyFromSymbol(dst, (const void *)(&symbol), count, offset, kind); 
# 779
} 
# 823 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 824
cudaMemcpyFromSymbolAsync(void *
# 825
dst, const T &
# 826
symbol, size_t 
# 827
count, size_t 
# 828
offset = 0, cudaMemcpyKind 
# 829
kind = cudaMemcpyDeviceToHost, cudaStream_t 
# 830
stream = 0) 
# 832
{ 
# 833
return ::cudaMemcpyFromSymbolAsync(dst, (const void *)(&symbol), count, offset, kind, stream); 
# 834
} 
# 859 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 860
cudaGetSymbolAddress(void **
# 861
devPtr, const T &
# 862
symbol) 
# 864
{ 
# 865
return ::cudaGetSymbolAddress(devPtr, (const void *)(&symbol)); 
# 866
} 
# 891 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 892
cudaGetSymbolSize(size_t *
# 893
size, const T &
# 894
symbol) 
# 896
{ 
# 897
return ::cudaGetSymbolSize(size, (const void *)(&symbol)); 
# 898
} 
# 935 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 936
cudaBindTexture(size_t *
# 937
offset, const texture< T, dim, readMode>  &
# 938
tex, const void *
# 939
devPtr, const cudaChannelFormatDesc &
# 940
desc, size_t 
# 941
size = ((2147483647) * 2U) + 1U) 
# 943
{ 
# 944
return ::cudaBindTexture(offset, &tex, devPtr, &desc, size); 
# 945
} 
# 981 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 982
cudaBindTexture(size_t *
# 983
offset, const texture< T, dim, readMode>  &
# 984
tex, const void *
# 985
devPtr, size_t 
# 986
size = ((2147483647) * 2U) + 1U) 
# 988
{ 
# 989
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size); 
# 990
} 
# 1038 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1039
cudaBindTexture2D(size_t *
# 1040
offset, const texture< T, dim, readMode>  &
# 1041
tex, const void *
# 1042
devPtr, const cudaChannelFormatDesc &
# 1043
desc, size_t 
# 1044
width, size_t 
# 1045
height, size_t 
# 1046
pitch) 
# 1048
{ 
# 1049
return ::cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch); 
# 1050
} 
# 1097 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1098
cudaBindTexture2D(size_t *
# 1099
offset, const texture< T, dim, readMode>  &
# 1100
tex, const void *
# 1101
devPtr, size_t 
# 1102
width, size_t 
# 1103
height, size_t 
# 1104
pitch) 
# 1106
{ 
# 1107
return ::cudaBindTexture2D(offset, &tex, devPtr, &(tex.channelDesc), width, height, pitch); 
# 1108
} 
# 1140 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1141
cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 1142
tex, cudaArray_const_t 
# 1143
array, const cudaChannelFormatDesc &
# 1144
desc) 
# 1146
{ 
# 1147
return ::cudaBindTextureToArray(&tex, array, &desc); 
# 1148
} 
# 1179 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1180
cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 1181
tex, cudaArray_const_t 
# 1182
array) 
# 1184
{ 
# 1185
cudaChannelFormatDesc desc; 
# 1186
cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
# 1188
return (err == (cudaSuccess)) ? cudaBindTextureToArray(tex, array, desc) : err; 
# 1189
} 
# 1221 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1222
cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
# 1223
tex, cudaMipmappedArray_const_t 
# 1224
mipmappedArray, const cudaChannelFormatDesc &
# 1225
desc) 
# 1227
{ 
# 1228
return ::cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc); 
# 1229
} 
# 1260 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1261
cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
# 1262
tex, cudaMipmappedArray_const_t 
# 1263
mipmappedArray) 
# 1265
{ 
# 1266
cudaChannelFormatDesc desc; 
# 1267
cudaArray_t levelArray; 
# 1268
cudaError_t err = ::cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0); 
# 1270
if (err != (cudaSuccess)) { 
# 1271
return err; 
# 1272
}  
# 1273
err = ::cudaGetChannelDesc(&desc, levelArray); 
# 1275
return (err == (cudaSuccess)) ? cudaBindTextureToMipmappedArray(tex, mipmappedArray, desc) : err; 
# 1276
} 
# 1303 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1304
cudaUnbindTexture(const texture< T, dim, readMode>  &
# 1305
tex) 
# 1307
{ 
# 1308
return ::cudaUnbindTexture(&tex); 
# 1309
} 
# 1339 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1340
cudaGetTextureAlignmentOffset(size_t *
# 1341
offset, const texture< T, dim, readMode>  &
# 1342
tex) 
# 1344
{ 
# 1345
return ::cudaGetTextureAlignmentOffset(offset, &tex); 
# 1346
} 
# 1391 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1392
cudaFuncSetCacheConfig(T *
# 1393
func, cudaFuncCache 
# 1394
cacheConfig) 
# 1396
{ 
# 1397
return ::cudaFuncSetCacheConfig((const void *)func, cacheConfig); 
# 1398
} 
# 1400
template< class T> static inline cudaError_t 
# 1401
cudaFuncSetSharedMemConfig(T *
# 1402
func, cudaSharedMemConfig 
# 1403
config) 
# 1405
{ 
# 1406
return ::cudaFuncSetSharedMemConfig((const void *)func, config); 
# 1407
} 
# 1436 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 1437
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *
# 1438
numBlocks, T 
# 1439
func, int 
# 1440
blockSize, size_t 
# 1441
dynamicSMemSize) 
# 1442
{ 
# 1443
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, 0); 
# 1444
} 
# 1487 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 1488
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *
# 1489
numBlocks, T 
# 1490
func, int 
# 1491
blockSize, size_t 
# 1492
dynamicSMemSize, unsigned 
# 1493
flags) 
# 1494
{ 
# 1495
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, flags); 
# 1496
} 
# 1501
class __cudaOccupancyB2DHelper { 
# 1502
size_t n; 
# 1504
public: __cudaOccupancyB2DHelper(size_t n_) : n(n_) { } 
# 1505
size_t operator()(int) 
# 1506
{ 
# 1507
return n; 
# 1508
} 
# 1509
}; 
# 1556 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class UnaryFunction, class T> static inline cudaError_t 
# 1557
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *
# 1558
minGridSize, int *
# 1559
blockSize, T 
# 1560
func, UnaryFunction 
# 1561
blockSizeToDynamicSMemSize, int 
# 1562
blockSizeLimit = 0, unsigned 
# 1563
flags = 0) 
# 1564
{ 
# 1565
cudaError_t status; 
# 1568
int device; 
# 1569
cudaFuncAttributes attr; 
# 1572
int maxThreadsPerMultiProcessor; 
# 1573
int warpSize; 
# 1574
int devMaxThreadsPerBlock; 
# 1575
int multiProcessorCount; 
# 1576
int funcMaxThreadsPerBlock; 
# 1577
int occupancyLimit; 
# 1578
int granularity; 
# 1581
int maxBlockSize = 0; 
# 1582
int numBlocks = 0; 
# 1583
int maxOccupancy = 0; 
# 1586
int blockSizeToTryAligned; 
# 1587
int blockSizeToTry; 
# 1588
int blockSizeLimitAligned; 
# 1589
int occupancyInBlocks; 
# 1590
int occupancyInThreads; 
# 1591
size_t dynamicSMemSize; 
# 1597
if (((!minGridSize) || (!blockSize)) || (!func)) { 
# 1598
return cudaErrorInvalidValue; 
# 1599
}  
# 1605
status = ::cudaGetDevice(&device); 
# 1606
if (status != (cudaSuccess)) { 
# 1607
return status; 
# 1608
}  
# 1610
status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device); 
# 1614
if (status != (cudaSuccess)) { 
# 1615
return status; 
# 1616
}  
# 1618
status = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device); 
# 1622
if (status != (cudaSuccess)) { 
# 1623
return status; 
# 1624
}  
# 1626
status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device); 
# 1630
if (status != (cudaSuccess)) { 
# 1631
return status; 
# 1632
}  
# 1634
status = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device); 
# 1638
if (status != (cudaSuccess)) { 
# 1639
return status; 
# 1640
}  
# 1642
status = cudaFuncGetAttributes(&attr, func); 
# 1643
if (status != (cudaSuccess)) { 
# 1644
return status; 
# 1645
}  
# 1647
funcMaxThreadsPerBlock = (attr.maxThreadsPerBlock); 
# 1653
occupancyLimit = maxThreadsPerMultiProcessor; 
# 1654
granularity = warpSize; 
# 1656
if (blockSizeLimit == 0) { 
# 1657
blockSizeLimit = devMaxThreadsPerBlock; 
# 1658
}  
# 1660
if (devMaxThreadsPerBlock < blockSizeLimit) { 
# 1661
blockSizeLimit = devMaxThreadsPerBlock; 
# 1662
}  
# 1664
if (funcMaxThreadsPerBlock < blockSizeLimit) { 
# 1665
blockSizeLimit = funcMaxThreadsPerBlock; 
# 1666
}  
# 1668
blockSizeLimitAligned = (((blockSizeLimit + (granularity - 1)) / granularity) * granularity); 
# 1670
for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) { 
# 1674
if (blockSizeLimit < blockSizeToTryAligned) { 
# 1675
blockSizeToTry = blockSizeLimit; 
# 1676
} else { 
# 1677
blockSizeToTry = blockSizeToTryAligned; 
# 1678
}  
# 1680
dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry); 
# 1682
status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancyInBlocks, func, blockSizeToTry, dynamicSMemSize, flags); 
# 1689
if (status != (cudaSuccess)) { 
# 1690
return status; 
# 1691
}  
# 1693
occupancyInThreads = (blockSizeToTry * occupancyInBlocks); 
# 1695
if (occupancyInThreads > maxOccupancy) { 
# 1696
maxBlockSize = blockSizeToTry; 
# 1697
numBlocks = occupancyInBlocks; 
# 1698
maxOccupancy = occupancyInThreads; 
# 1699
}  
# 1703
if (occupancyLimit == maxOccupancy) { 
# 1704
break; 
# 1705
}  
# 1706
}  
# 1714
(*minGridSize) = (numBlocks * multiProcessorCount); 
# 1715
(*blockSize) = maxBlockSize; 
# 1717
return status; 
# 1718
} 
# 1751 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class UnaryFunction, class T> static inline cudaError_t 
# 1752
cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *
# 1753
minGridSize, int *
# 1754
blockSize, T 
# 1755
func, UnaryFunction 
# 1756
blockSizeToDynamicSMemSize, int 
# 1757
blockSizeLimit = 0) 
# 1758
{ 
# 1759
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0); 
# 1760
} 
# 1796 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1797
cudaOccupancyMaxPotentialBlockSize(int *
# 1798
minGridSize, int *
# 1799
blockSize, T 
# 1800
func, size_t 
# 1801
dynamicSMemSize = 0, int 
# 1802
blockSizeLimit = 0) 
# 1803
{ 
# 1804
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, 0); 
# 1805
} 
# 1855 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1856
cudaOccupancyMaxPotentialBlockSizeWithFlags(int *
# 1857
minGridSize, int *
# 1858
blockSize, T 
# 1859
func, size_t 
# 1860
dynamicSMemSize = 0, int 
# 1861
blockSizeLimit = 0, unsigned 
# 1862
flags = 0) 
# 1863
{ 
# 1864
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, flags); 
# 1865
} 
# 1896 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 1897
cudaFuncGetAttributes(cudaFuncAttributes *
# 1898
attr, T *
# 1899
entry) 
# 1901
{ 
# 1902
return ::cudaFuncGetAttributes(attr, (const void *)entry); 
# 1903
} 
# 1941 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1942
cudaFuncSetAttribute(T *
# 1943
entry, cudaFuncAttribute 
# 1944
attr, int 
# 1945
value) 
# 1947
{ 
# 1948
return ::cudaFuncSetAttribute((const void *)entry, attr, value); 
# 1949
} 
# 1973 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim> 
# 1974
__attribute((deprecated)) static inline cudaError_t cudaBindSurfaceToArray(const surface< T, dim>  &
# 1975
surf, cudaArray_const_t 
# 1976
array, const cudaChannelFormatDesc &
# 1977
desc) 
# 1979
{ 
# 1980
return ::cudaBindSurfaceToArray(&surf, array, &desc); 
# 1981
} 
# 2004 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim> 
# 2005
__attribute((deprecated)) static inline cudaError_t cudaBindSurfaceToArray(const surface< T, dim>  &
# 2006
surf, cudaArray_const_t 
# 2007
array) 
# 2009
{ 
# 2010
cudaChannelFormatDesc desc; 
# 2011
cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
# 2013
return (err == (cudaSuccess)) ? cudaBindSurfaceToArray(surf, array, desc) : err; 
# 2014
} 
# 2025 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda_runtime.h"
#pragma GCC diagnostic pop
# 29 "/usr/include/stdio.h" 3
extern "C" {
# 44 "/usr/include/stdio.h" 3
struct _IO_FILE; 
# 48
typedef _IO_FILE FILE; 
# 64 "/usr/include/stdio.h" 3
typedef _IO_FILE __FILE; 
# 94 "/usr/include/wchar.h" 3
typedef 
# 83
struct { 
# 84
int __count; 
# 86
union { 
# 88
unsigned __wch; 
# 92
char __wchb[4]; 
# 93
} __value; 
# 94
} __mbstate_t; 
# 25 "/usr/include/_G_config.h" 3
typedef 
# 22
struct { 
# 23
__off_t __pos; 
# 24
__mbstate_t __state; 
# 25
} _G_fpos_t; 
# 30
typedef 
# 27
struct { 
# 28
__off64_t __pos; 
# 29
__mbstate_t __state; 
# 30
} _G_fpos64_t; 
# 40 "/usr/lib/gcc/x86_64-redhat-linux/4.8.5/include/stdarg.h" 3
typedef __builtin_va_list __gnuc_va_list; 
# 145 "/usr/include/libio.h" 3
struct _IO_jump_t; struct _IO_FILE; 
# 155 "/usr/include/libio.h" 3
typedef void _IO_lock_t; 
# 161
struct _IO_marker { 
# 162
_IO_marker *_next; 
# 163
_IO_FILE *_sbuf; 
# 167
int _pos; 
# 178 "/usr/include/libio.h" 3
}; 
# 181
enum __codecvt_result { 
# 183
__codecvt_ok, 
# 184
__codecvt_partial, 
# 185
__codecvt_error, 
# 186
__codecvt_noconv
# 187
}; 
# 246 "/usr/include/libio.h" 3
struct _IO_FILE { 
# 247
int _flags; 
# 252
char *_IO_read_ptr; 
# 253
char *_IO_read_end; 
# 254
char *_IO_read_base; 
# 255
char *_IO_write_base; 
# 256
char *_IO_write_ptr; 
# 257
char *_IO_write_end; 
# 258
char *_IO_buf_base; 
# 259
char *_IO_buf_end; 
# 261
char *_IO_save_base; 
# 262
char *_IO_backup_base; 
# 263
char *_IO_save_end; 
# 265
_IO_marker *_markers; 
# 267
_IO_FILE *_chain; 
# 269
int _fileno; 
# 273
int _flags2; 
# 275
__off_t _old_offset; 
# 279
unsigned short _cur_column; 
# 280
signed char _vtable_offset; 
# 281
char _shortbuf[1]; 
# 285
_IO_lock_t *_lock; 
# 294 "/usr/include/libio.h" 3
__off64_t _offset; 
# 303 "/usr/include/libio.h" 3
void *__pad1; 
# 304
void *__pad2; 
# 305
void *__pad3; 
# 306
void *__pad4; 
# 307
size_t __pad5; 
# 309
int _mode; 
# 311
char _unused2[(((15) * sizeof(int)) - ((4) * sizeof(void *))) - sizeof(size_t)]; 
# 313
}; 
# 319
struct _IO_FILE_plus; 
# 321
extern _IO_FILE_plus _IO_2_1_stdin_; 
# 322
extern _IO_FILE_plus _IO_2_1_stdout_; 
# 323
extern _IO_FILE_plus _IO_2_1_stderr_; 
# 339 "/usr/include/libio.h" 3
typedef __ssize_t __io_read_fn(void * __cookie, char * __buf, size_t __nbytes); 
# 347
typedef __ssize_t __io_write_fn(void * __cookie, const char * __buf, size_t __n); 
# 356
typedef int __io_seek_fn(void * __cookie, __off64_t * __pos, int __w); 
# 359
typedef int __io_close_fn(void * __cookie); 
# 364
typedef __io_read_fn cookie_read_function_t; 
# 365
typedef __io_write_fn cookie_write_function_t; 
# 366
typedef __io_seek_fn cookie_seek_function_t; 
# 367
typedef __io_close_fn cookie_close_function_t; 
# 376
typedef 
# 371
struct { 
# 372
__io_read_fn *read; 
# 373
__io_write_fn *write; 
# 374
__io_seek_fn *seek; 
# 375
__io_close_fn *close; 
# 376
} _IO_cookie_io_functions_t; 
# 377
typedef _IO_cookie_io_functions_t cookie_io_functions_t; 
# 379
struct _IO_cookie_file; 
# 382
extern void _IO_cookie_init(_IO_cookie_file * __cfile, int __read_write, void * __cookie, _IO_cookie_io_functions_t __fns); 
# 388
extern "C" {
# 391
extern int __underflow(_IO_FILE *); 
# 392
extern int __uflow(_IO_FILE *); 
# 393
extern int __overflow(_IO_FILE *, int); 
# 435 "/usr/include/libio.h" 3
extern int _IO_getc(_IO_FILE * __fp); 
# 436
extern int _IO_putc(int __c, _IO_FILE * __fp); 
# 437
extern int _IO_feof(_IO_FILE * __fp) throw(); 
# 438
extern int _IO_ferror(_IO_FILE * __fp) throw(); 
# 440
extern int _IO_peekc_locked(_IO_FILE * __fp); 
# 446
extern void _IO_flockfile(_IO_FILE *) throw(); 
# 447
extern void _IO_funlockfile(_IO_FILE *) throw(); 
# 448
extern int _IO_ftrylockfile(_IO_FILE *) throw(); 
# 465 "/usr/include/libio.h" 3
extern int _IO_vfscanf(_IO_FILE *__restrict__, const char *__restrict__, __gnuc_va_list, int *__restrict__); 
# 467
extern int _IO_vfprintf(_IO_FILE *__restrict__, const char *__restrict__, __gnuc_va_list); 
# 469
extern __ssize_t _IO_padn(_IO_FILE *, int, __ssize_t); 
# 470
extern size_t _IO_sgetn(_IO_FILE *, void *, size_t); 
# 472
extern __off64_t _IO_seekoff(_IO_FILE *, __off64_t, int, int); 
# 473
extern __off64_t _IO_seekpos(_IO_FILE *, __off64_t, int); 
# 475
extern void _IO_free_backup_area(_IO_FILE *) throw(); 
# 527 "/usr/include/libio.h" 3
}
# 79 "/usr/include/stdio.h" 3
typedef __gnuc_va_list va_list; 
# 110 "/usr/include/stdio.h" 3
typedef _G_fpos_t fpos_t; 
# 116
typedef _G_fpos64_t fpos64_t; 
# 168 "/usr/include/stdio.h" 3
extern _IO_FILE *stdin; 
# 169
extern _IO_FILE *stdout; 
# 170
extern _IO_FILE *stderr; 
# 178
extern int remove(const char * __filename) throw(); 
# 180
extern int rename(const char * __old, const char * __new) throw(); 
# 185
extern int renameat(int __oldfd, const char * __old, int __newfd, const char * __new) throw(); 
# 195
extern FILE *tmpfile(); 
# 205 "/usr/include/stdio.h" 3
extern FILE *tmpfile64(); 
# 209
extern char *tmpnam(char * __s) throw(); 
# 215
extern char *tmpnam_r(char * __s) throw(); 
# 227 "/usr/include/stdio.h" 3
extern char *tempnam(const char * __dir, const char * __pfx) throw()
# 228
 __attribute((__malloc__)); 
# 237
extern int fclose(FILE * __stream); 
# 242
extern int fflush(FILE * __stream); 
# 252 "/usr/include/stdio.h" 3
extern int fflush_unlocked(FILE * __stream); 
# 262 "/usr/include/stdio.h" 3
extern int fcloseall(); 
# 272
extern FILE *fopen(const char *__restrict__ __filename, const char *__restrict__ __modes); 
# 278
extern FILE *freopen(const char *__restrict__ __filename, const char *__restrict__ __modes, FILE *__restrict__ __stream); 
# 297 "/usr/include/stdio.h" 3
extern FILE *fopen64(const char *__restrict__ __filename, const char *__restrict__ __modes); 
# 299
extern FILE *freopen64(const char *__restrict__ __filename, const char *__restrict__ __modes, FILE *__restrict__ __stream); 
# 306
extern FILE *fdopen(int __fd, const char * __modes) throw(); 
# 312
extern FILE *fopencookie(void *__restrict__ __magic_cookie, const char *__restrict__ __modes, _IO_cookie_io_functions_t __io_funcs) throw(); 
# 319
extern FILE *fmemopen(void * __s, size_t __len, const char * __modes) throw(); 
# 325
extern FILE *open_memstream(char ** __bufloc, size_t * __sizeloc) throw(); 
# 332
extern void setbuf(FILE *__restrict__ __stream, char *__restrict__ __buf) throw(); 
# 336
extern int setvbuf(FILE *__restrict__ __stream, char *__restrict__ __buf, int __modes, size_t __n) throw(); 
# 343
extern void setbuffer(FILE *__restrict__ __stream, char *__restrict__ __buf, size_t __size) throw(); 
# 347
extern void setlinebuf(FILE * __stream) throw(); 
# 356
extern int fprintf(FILE *__restrict__ __stream, const char *__restrict__ __format, ...); 
# 362
extern int printf(const char *__restrict__ __format, ...); 
# 364
extern int sprintf(char *__restrict__ __s, const char *__restrict__ __format, ...) throw(); 
# 371
extern int vfprintf(FILE *__restrict__ __s, const char *__restrict__ __format, __gnuc_va_list __arg); 
# 377
extern __attribute((gnu_inline)) inline int vprintf(const char *__restrict__ __format, __gnuc_va_list __arg); 
# 379
extern int vsprintf(char *__restrict__ __s, const char *__restrict__ __format, __gnuc_va_list __arg) throw(); 
# 386
extern int snprintf(char *__restrict__ __s, size_t __maxlen, const char *__restrict__ __format, ...) throw()
# 388
 __attribute((__format__(__printf__, 3, 4))); 
# 390
extern int vsnprintf(char *__restrict__ __s, size_t __maxlen, const char *__restrict__ __format, __gnuc_va_list __arg) throw()
# 392
 __attribute((__format__(__printf__, 3, 0))); 
# 399
extern int vasprintf(char **__restrict__ __ptr, const char *__restrict__ __f, __gnuc_va_list __arg) throw()
# 401
 __attribute((__format__(__printf__, 2, 0))); 
# 402
extern int __asprintf(char **__restrict__ __ptr, const char *__restrict__ __fmt, ...) throw()
# 404
 __attribute((__format__(__printf__, 2, 3))); 
# 405
extern int asprintf(char **__restrict__ __ptr, const char *__restrict__ __fmt, ...) throw()
# 407
 __attribute((__format__(__printf__, 2, 3))); 
# 412
extern int vdprintf(int __fd, const char *__restrict__ __fmt, __gnuc_va_list __arg)
# 414
 __attribute((__format__(__printf__, 2, 0))); 
# 415
extern int dprintf(int __fd, const char *__restrict__ __fmt, ...)
# 416
 __attribute((__format__(__printf__, 2, 3))); 
# 425
extern int fscanf(FILE *__restrict__ __stream, const char *__restrict__ __format, ...); 
# 431
extern int scanf(const char *__restrict__ __format, ...); 
# 433
extern int sscanf(const char *__restrict__ __s, const char *__restrict__ __format, ...) throw(); 
# 471 "/usr/include/stdio.h" 3
extern int vfscanf(FILE *__restrict__ __s, const char *__restrict__ __format, __gnuc_va_list __arg)
# 473
 __attribute((__format__(__scanf__, 2, 0))); 
# 479
extern int vscanf(const char *__restrict__ __format, __gnuc_va_list __arg)
# 480
 __attribute((__format__(__scanf__, 1, 0))); 
# 483
extern int vsscanf(const char *__restrict__ __s, const char *__restrict__ __format, __gnuc_va_list __arg) throw()
# 485
 __attribute((__format__(__scanf__, 2, 0))); 
# 531 "/usr/include/stdio.h" 3
extern int fgetc(FILE * __stream); 
# 532
extern int getc(FILE * __stream); 
# 538
extern __attribute((gnu_inline)) inline int getchar(); 
# 550 "/usr/include/stdio.h" 3
extern __attribute((gnu_inline)) inline int getc_unlocked(FILE * __stream); 
# 551
extern __attribute((gnu_inline)) inline int getchar_unlocked(); 
# 561 "/usr/include/stdio.h" 3
extern __attribute((gnu_inline)) inline int fgetc_unlocked(FILE * __stream); 
# 573
extern int fputc(int __c, FILE * __stream); 
# 574
extern int putc(int __c, FILE * __stream); 
# 580
extern __attribute((gnu_inline)) inline int putchar(int __c); 
# 594 "/usr/include/stdio.h" 3
extern __attribute((gnu_inline)) inline int fputc_unlocked(int __c, FILE * __stream); 
# 602
extern __attribute((gnu_inline)) inline int putc_unlocked(int __c, FILE * __stream); 
# 603
extern __attribute((gnu_inline)) inline int putchar_unlocked(int __c); 
# 610
extern int getw(FILE * __stream); 
# 613
extern int putw(int __w, FILE * __stream); 
# 622
extern char *fgets(char *__restrict__ __s, int __n, FILE *__restrict__ __stream); 
# 638 "/usr/include/stdio.h" 3
extern char *gets(char * __s) __attribute((__deprecated__)); 
# 649 "/usr/include/stdio.h" 3
extern char *fgets_unlocked(char *__restrict__ __s, int __n, FILE *__restrict__ __stream); 
# 665 "/usr/include/stdio.h" 3
extern __ssize_t __getdelim(char **__restrict__ __lineptr, size_t *__restrict__ __n, int __delimiter, FILE *__restrict__ __stream); 
# 668
extern __ssize_t getdelim(char **__restrict__ __lineptr, size_t *__restrict__ __n, int __delimiter, FILE *__restrict__ __stream); 
# 678
extern __attribute((gnu_inline)) inline __ssize_t getline(char **__restrict__ __lineptr, size_t *__restrict__ __n, FILE *__restrict__ __stream); 
# 689
extern int fputs(const char *__restrict__ __s, FILE *__restrict__ __stream); 
# 695
extern int puts(const char * __s); 
# 702
extern int ungetc(int __c, FILE * __stream); 
# 709
extern size_t fread(void *__restrict__ __ptr, size_t __size, size_t __n, FILE *__restrict__ __stream); 
# 715
extern size_t fwrite(const void *__restrict__ __ptr, size_t __size, size_t __n, FILE *__restrict__ __s); 
# 726 "/usr/include/stdio.h" 3
extern int fputs_unlocked(const char *__restrict__ __s, FILE *__restrict__ __stream); 
# 737 "/usr/include/stdio.h" 3
extern size_t fread_unlocked(void *__restrict__ __ptr, size_t __size, size_t __n, FILE *__restrict__ __stream); 
# 739
extern size_t fwrite_unlocked(const void *__restrict__ __ptr, size_t __size, size_t __n, FILE *__restrict__ __stream); 
# 749
extern int fseek(FILE * __stream, long __off, int __whence); 
# 754
extern long ftell(FILE * __stream); 
# 759
extern void rewind(FILE * __stream); 
# 773 "/usr/include/stdio.h" 3
extern int fseeko(FILE * __stream, __off_t __off, int __whence); 
# 778
extern __off_t ftello(FILE * __stream); 
# 798 "/usr/include/stdio.h" 3
extern int fgetpos(FILE *__restrict__ __stream, fpos_t *__restrict__ __pos); 
# 803
extern int fsetpos(FILE * __stream, const fpos_t * __pos); 
# 818 "/usr/include/stdio.h" 3
extern int fseeko64(FILE * __stream, __off64_t __off, int __whence); 
# 819
extern __off64_t ftello64(FILE * __stream); 
# 820
extern int fgetpos64(FILE *__restrict__ __stream, fpos64_t *__restrict__ __pos); 
# 821
extern int fsetpos64(FILE * __stream, const fpos64_t * __pos); 
# 826
extern void clearerr(FILE * __stream) throw(); 
# 828
extern int feof(FILE * __stream) throw(); 
# 830
extern int ferror(FILE * __stream) throw(); 
# 835
extern void clearerr_unlocked(FILE * __stream) throw(); 
# 836
extern __attribute((gnu_inline)) inline int feof_unlocked(FILE * __stream) throw(); 
# 837
extern __attribute((gnu_inline)) inline int ferror_unlocked(FILE * __stream) throw(); 
# 846
extern void perror(const char * __s); 
# 26 "/usr/include/bits/sys_errlist.h" 3
extern int sys_nerr; 
# 27
extern const char *const sys_errlist[]; 
# 30
extern int _sys_nerr; 
# 31
extern const char *const _sys_errlist[]; 
# 858 "/usr/include/stdio.h" 3
extern int fileno(FILE * __stream) throw(); 
# 863
extern int fileno_unlocked(FILE * __stream) throw(); 
# 873 "/usr/include/stdio.h" 3
extern FILE *popen(const char * __command, const char * __modes); 
# 879
extern int pclose(FILE * __stream); 
# 885
extern char *ctermid(char * __s) throw(); 
# 891
extern char *cuserid(char * __s); 
# 896
struct obstack; 
# 899
extern int obstack_printf(obstack *__restrict__ __obstack, const char *__restrict__ __format, ...) throw()
# 901
 __attribute((__format__(__printf__, 2, 3))); 
# 902
extern int obstack_vprintf(obstack *__restrict__ __obstack, const char *__restrict__ __format, __gnuc_va_list __args) throw()
# 905
 __attribute((__format__(__printf__, 2, 0))); 
# 913
extern void flockfile(FILE * __stream) throw(); 
# 917
extern int ftrylockfile(FILE * __stream) throw(); 
# 920
extern void funlockfile(FILE * __stream) throw(); 
# 35 "/usr/include/bits/stdio.h" 3
__attribute((__gnu_inline__)) extern inline int 
# 36
vprintf(const char *__restrict__ __fmt, __gnuc_va_list __arg) 
# 37
{ 
# 38
return vfprintf(stdout, __fmt, __arg); 
# 39
} 
# 43
__attribute((__gnu_inline__)) extern inline int 
# 44
getchar() 
# 45
{ 
# 46
return _IO_getc(stdin); 
# 47
} 
# 52
__attribute((__gnu_inline__)) extern inline int 
# 53
fgetc_unlocked(FILE *__fp) 
# 54
{ 
# 55
return (__builtin_expect((__fp->_IO_read_ptr) >= (__fp->_IO_read_end), 0)) ? __uflow(__fp) : (*((unsigned char *)((__fp->_IO_read_ptr)++))); 
# 56
} 
# 62
__attribute((__gnu_inline__)) extern inline int 
# 63
getc_unlocked(FILE *__fp) 
# 64
{ 
# 65
return (__builtin_expect((__fp->_IO_read_ptr) >= (__fp->_IO_read_end), 0)) ? __uflow(__fp) : (*((unsigned char *)((__fp->_IO_read_ptr)++))); 
# 66
} 
# 69
__attribute((__gnu_inline__)) extern inline int 
# 70
getchar_unlocked() 
# 71
{ 
# 72
return (__builtin_expect((stdin->_IO_read_ptr) >= (stdin->_IO_read_end), 0)) ? __uflow(stdin) : (*((unsigned char *)((stdin->_IO_read_ptr)++))); 
# 73
} 
# 78
__attribute((__gnu_inline__)) extern inline int 
# 79
putchar(int __c) 
# 80
{ 
# 81
return _IO_putc(__c, stdout); 
# 82
} 
# 87
__attribute((__gnu_inline__)) extern inline int 
# 88
fputc_unlocked(int __c, FILE *__stream) 
# 89
{ 
# 90
return (__builtin_expect((__stream->_IO_write_ptr) >= (__stream->_IO_write_end), 0)) ? __overflow(__stream, (unsigned char)__c) : ((unsigned char)((*((__stream->_IO_write_ptr)++)) = __c)); 
# 91
} 
# 97
__attribute((__gnu_inline__)) extern inline int 
# 98
putc_unlocked(int __c, FILE *__stream) 
# 99
{ 
# 100
return (__builtin_expect((__stream->_IO_write_ptr) >= (__stream->_IO_write_end), 0)) ? __overflow(__stream, (unsigned char)__c) : ((unsigned char)((*((__stream->_IO_write_ptr)++)) = __c)); 
# 101
} 
# 104
__attribute((__gnu_inline__)) extern inline int 
# 105
putchar_unlocked(int __c) 
# 106
{ 
# 107
return (__builtin_expect((stdout->_IO_write_ptr) >= (stdout->_IO_write_end), 0)) ? __overflow(stdout, (unsigned char)__c) : ((unsigned char)((*((stdout->_IO_write_ptr)++)) = __c)); 
# 108
} 
# 114
__attribute((__gnu_inline__)) extern inline __ssize_t 
# 115
getline(char **__lineptr, size_t *__n, FILE *__stream) 
# 116
{ 
# 117
return __getdelim(__lineptr, __n, '\n', __stream); 
# 118
} 
# 124
__attribute((__gnu_inline__)) extern inline int
# 125
 __attribute((__leaf__)) feof_unlocked(FILE *__stream) throw() 
# 126
{ 
# 127
return ((__stream->_flags) & 16) != 0; 
# 128
} 
# 131
__attribute((__gnu_inline__)) extern inline int
# 132
 __attribute((__leaf__)) ferror_unlocked(FILE *__stream) throw() 
# 133
{ 
# 134
return ((__stream->_flags) & 32) != 0; 
# 135
} 
# 943 "/usr/include/stdio.h" 3
}
# 27 "/usr/include/unistd.h" 3
extern "C" {
# 267 "/usr/include/unistd.h" 3
typedef __intptr_t intptr_t; 
# 274
typedef __socklen_t socklen_t; 
# 287 "/usr/include/unistd.h" 3
extern int access(const char * __name, int __type) throw() __attribute((__nonnull__(1))); 
# 292
extern int euidaccess(const char * __name, int __type) throw()
# 293
 __attribute((__nonnull__(1))); 
# 296
extern int eaccess(const char * __name, int __type) throw()
# 297
 __attribute((__nonnull__(1))); 
# 304
extern int faccessat(int __fd, const char * __file, int __type, int __flag) throw()
# 305
 __attribute((__nonnull__(2))); 
# 334 "/usr/include/unistd.h" 3
extern __off_t lseek(int __fd, __off_t __offset, int __whence) throw(); 
# 345 "/usr/include/unistd.h" 3
extern __off64_t lseek64(int __fd, __off64_t __offset, int __whence) throw(); 
# 353
extern int close(int __fd); 
# 360
extern ssize_t read(int __fd, void * __buf, size_t __nbytes); 
# 366
extern ssize_t write(int __fd, const void * __buf, size_t __n); 
# 376 "/usr/include/unistd.h" 3
extern ssize_t pread(int __fd, void * __buf, size_t __nbytes, __off_t __offset); 
# 384
extern ssize_t pwrite(int __fd, const void * __buf, size_t __n, __off_t __offset); 
# 404 "/usr/include/unistd.h" 3
extern ssize_t pread64(int __fd, void * __buf, size_t __nbytes, __off64_t __offset); 
# 408
extern ssize_t pwrite64(int __fd, const void * __buf, size_t __n, __off64_t __offset); 
# 417
extern int pipe(int  __pipedes[2]) throw(); 
# 422
extern int pipe2(int  __pipedes[2], int __flags) throw(); 
# 432 "/usr/include/unistd.h" 3
extern unsigned alarm(unsigned __seconds) throw(); 
# 444 "/usr/include/unistd.h" 3
extern unsigned sleep(unsigned __seconds); 
# 452
extern __useconds_t ualarm(__useconds_t __value, __useconds_t __interval) throw(); 
# 460
extern int usleep(__useconds_t __useconds); 
# 469 "/usr/include/unistd.h" 3
extern int pause(); 
# 473
extern int chown(const char * __file, __uid_t __owner, __gid_t __group) throw()
# 474
 __attribute((__nonnull__(1))); 
# 478
extern int fchown(int __fd, __uid_t __owner, __gid_t __group) throw(); 
# 483
extern int lchown(const char * __file, __uid_t __owner, __gid_t __group) throw()
# 484
 __attribute((__nonnull__(1))); 
# 491
extern int fchownat(int __fd, const char * __file, __uid_t __owner, __gid_t __group, int __flag) throw()
# 493
 __attribute((__nonnull__(2))); 
# 497
extern int chdir(const char * __path) throw() __attribute((__nonnull__(1))); 
# 501
extern int fchdir(int __fd) throw(); 
# 511 "/usr/include/unistd.h" 3
extern char *getcwd(char * __buf, size_t __size) throw(); 
# 517
extern char *get_current_dir_name() throw(); 
# 525
extern char *getwd(char * __buf) throw()
# 526
 __attribute((__nonnull__(1))) __attribute((__deprecated__)); 
# 531
extern int dup(int __fd) throw(); 
# 534
extern int dup2(int __fd, int __fd2) throw(); 
# 539
extern int dup3(int __fd, int __fd2, int __flags) throw(); 
# 543
extern char **__environ; 
# 545
extern char **environ; 
# 551
extern int execve(const char * __path, char *const  __argv[], char *const  __envp[]) throw()
# 552
 __attribute((__nonnull__(1, 2))); 
# 557
extern int fexecve(int __fd, char *const  __argv[], char *const  __envp[]) throw()
# 558
 __attribute((__nonnull__(2))); 
# 563
extern int execv(const char * __path, char *const  __argv[]) throw()
# 564
 __attribute((__nonnull__(1, 2))); 
# 568
extern int execle(const char * __path, const char * __arg, ...) throw()
# 569
 __attribute((__nonnull__(1, 2))); 
# 573
extern int execl(const char * __path, const char * __arg, ...) throw()
# 574
 __attribute((__nonnull__(1, 2))); 
# 578
extern int execvp(const char * __file, char *const  __argv[]) throw()
# 579
 __attribute((__nonnull__(1, 2))); 
# 584
extern int execlp(const char * __file, const char * __arg, ...) throw()
# 585
 __attribute((__nonnull__(1, 2))); 
# 590
extern int execvpe(const char * __file, char *const  __argv[], char *const  __envp[]) throw()
# 592
 __attribute((__nonnull__(1, 2))); 
# 598
extern int nice(int __inc) throw(); 
# 603
extern void _exit(int __status) __attribute((__noreturn__)); 
# 26 "/usr/include/bits/confname.h" 3
enum { 
# 27
_PC_LINK_MAX, 
# 29
_PC_MAX_CANON, 
# 31
_PC_MAX_INPUT, 
# 33
_PC_NAME_MAX, 
# 35
_PC_PATH_MAX, 
# 37
_PC_PIPE_BUF, 
# 39
_PC_CHOWN_RESTRICTED, 
# 41
_PC_NO_TRUNC, 
# 43
_PC_VDISABLE, 
# 45
_PC_SYNC_IO, 
# 47
_PC_ASYNC_IO, 
# 49
_PC_PRIO_IO, 
# 51
_PC_SOCK_MAXBUF, 
# 53
_PC_FILESIZEBITS, 
# 55
_PC_REC_INCR_XFER_SIZE, 
# 57
_PC_REC_MAX_XFER_SIZE, 
# 59
_PC_REC_MIN_XFER_SIZE, 
# 61
_PC_REC_XFER_ALIGN, 
# 63
_PC_ALLOC_SIZE_MIN, 
# 65
_PC_SYMLINK_MAX, 
# 67
_PC_2_SYMLINKS
# 69
}; 
# 73
enum { 
# 74
_SC_ARG_MAX, 
# 76
_SC_CHILD_MAX, 
# 78
_SC_CLK_TCK, 
# 80
_SC_NGROUPS_MAX, 
# 82
_SC_OPEN_MAX, 
# 84
_SC_STREAM_MAX, 
# 86
_SC_TZNAME_MAX, 
# 88
_SC_JOB_CONTROL, 
# 90
_SC_SAVED_IDS, 
# 92
_SC_REALTIME_SIGNALS, 
# 94
_SC_PRIORITY_SCHEDULING, 
# 96
_SC_TIMERS, 
# 98
_SC_ASYNCHRONOUS_IO, 
# 100
_SC_PRIORITIZED_IO, 
# 102
_SC_SYNCHRONIZED_IO, 
# 104
_SC_FSYNC, 
# 106
_SC_MAPPED_FILES, 
# 108
_SC_MEMLOCK, 
# 110
_SC_MEMLOCK_RANGE, 
# 112
_SC_MEMORY_PROTECTION, 
# 114
_SC_MESSAGE_PASSING, 
# 116
_SC_SEMAPHORES, 
# 118
_SC_SHARED_MEMORY_OBJECTS, 
# 120
_SC_AIO_LISTIO_MAX, 
# 122
_SC_AIO_MAX, 
# 124
_SC_AIO_PRIO_DELTA_MAX, 
# 126
_SC_DELAYTIMER_MAX, 
# 128
_SC_MQ_OPEN_MAX, 
# 130
_SC_MQ_PRIO_MAX, 
# 132
_SC_VERSION, 
# 134
_SC_PAGESIZE, 
# 137
_SC_RTSIG_MAX, 
# 139
_SC_SEM_NSEMS_MAX, 
# 141
_SC_SEM_VALUE_MAX, 
# 143
_SC_SIGQUEUE_MAX, 
# 145
_SC_TIMER_MAX, 
# 150
_SC_BC_BASE_MAX, 
# 152
_SC_BC_DIM_MAX, 
# 154
_SC_BC_SCALE_MAX, 
# 156
_SC_BC_STRING_MAX, 
# 158
_SC_COLL_WEIGHTS_MAX, 
# 160
_SC_EQUIV_CLASS_MAX, 
# 162
_SC_EXPR_NEST_MAX, 
# 164
_SC_LINE_MAX, 
# 166
_SC_RE_DUP_MAX, 
# 168
_SC_CHARCLASS_NAME_MAX, 
# 171
_SC_2_VERSION, 
# 173
_SC_2_C_BIND, 
# 175
_SC_2_C_DEV, 
# 177
_SC_2_FORT_DEV, 
# 179
_SC_2_FORT_RUN, 
# 181
_SC_2_SW_DEV, 
# 183
_SC_2_LOCALEDEF, 
# 186
_SC_PII, 
# 188
_SC_PII_XTI, 
# 190
_SC_PII_SOCKET, 
# 192
_SC_PII_INTERNET, 
# 194
_SC_PII_OSI, 
# 196
_SC_POLL, 
# 198
_SC_SELECT, 
# 200
_SC_UIO_MAXIOV, 
# 202
_SC_IOV_MAX = 60, 
# 204
_SC_PII_INTERNET_STREAM, 
# 206
_SC_PII_INTERNET_DGRAM, 
# 208
_SC_PII_OSI_COTS, 
# 210
_SC_PII_OSI_CLTS, 
# 212
_SC_PII_OSI_M, 
# 214
_SC_T_IOV_MAX, 
# 218
_SC_THREADS, 
# 220
_SC_THREAD_SAFE_FUNCTIONS, 
# 222
_SC_GETGR_R_SIZE_MAX, 
# 224
_SC_GETPW_R_SIZE_MAX, 
# 226
_SC_LOGIN_NAME_MAX, 
# 228
_SC_TTY_NAME_MAX, 
# 230
_SC_THREAD_DESTRUCTOR_ITERATIONS, 
# 232
_SC_THREAD_KEYS_MAX, 
# 234
_SC_THREAD_STACK_MIN, 
# 236
_SC_THREAD_THREADS_MAX, 
# 238
_SC_THREAD_ATTR_STACKADDR, 
# 240
_SC_THREAD_ATTR_STACKSIZE, 
# 242
_SC_THREAD_PRIORITY_SCHEDULING, 
# 244
_SC_THREAD_PRIO_INHERIT, 
# 246
_SC_THREAD_PRIO_PROTECT, 
# 248
_SC_THREAD_PROCESS_SHARED, 
# 251
_SC_NPROCESSORS_CONF, 
# 253
_SC_NPROCESSORS_ONLN, 
# 255
_SC_PHYS_PAGES, 
# 257
_SC_AVPHYS_PAGES, 
# 259
_SC_ATEXIT_MAX, 
# 261
_SC_PASS_MAX, 
# 264
_SC_XOPEN_VERSION, 
# 266
_SC_XOPEN_XCU_VERSION, 
# 268
_SC_XOPEN_UNIX, 
# 270
_SC_XOPEN_CRYPT, 
# 272
_SC_XOPEN_ENH_I18N, 
# 274
_SC_XOPEN_SHM, 
# 277
_SC_2_CHAR_TERM, 
# 279
_SC_2_C_VERSION, 
# 281
_SC_2_UPE, 
# 284
_SC_XOPEN_XPG2, 
# 286
_SC_XOPEN_XPG3, 
# 288
_SC_XOPEN_XPG4, 
# 291
_SC_CHAR_BIT, 
# 293
_SC_CHAR_MAX, 
# 295
_SC_CHAR_MIN, 
# 297
_SC_INT_MAX, 
# 299
_SC_INT_MIN, 
# 301
_SC_LONG_BIT, 
# 303
_SC_WORD_BIT, 
# 305
_SC_MB_LEN_MAX, 
# 307
_SC_NZERO, 
# 309
_SC_SSIZE_MAX, 
# 311
_SC_SCHAR_MAX, 
# 313
_SC_SCHAR_MIN, 
# 315
_SC_SHRT_MAX, 
# 317
_SC_SHRT_MIN, 
# 319
_SC_UCHAR_MAX, 
# 321
_SC_UINT_MAX, 
# 323
_SC_ULONG_MAX, 
# 325
_SC_USHRT_MAX, 
# 328
_SC_NL_ARGMAX, 
# 330
_SC_NL_LANGMAX, 
# 332
_SC_NL_MSGMAX, 
# 334
_SC_NL_NMAX, 
# 336
_SC_NL_SETMAX, 
# 338
_SC_NL_TEXTMAX, 
# 341
_SC_XBS5_ILP32_OFF32, 
# 343
_SC_XBS5_ILP32_OFFBIG, 
# 345
_SC_XBS5_LP64_OFF64, 
# 347
_SC_XBS5_LPBIG_OFFBIG, 
# 350
_SC_XOPEN_LEGACY, 
# 352
_SC_XOPEN_REALTIME, 
# 354
_SC_XOPEN_REALTIME_THREADS, 
# 357
_SC_ADVISORY_INFO, 
# 359
_SC_BARRIERS, 
# 361
_SC_BASE, 
# 363
_SC_C_LANG_SUPPORT, 
# 365
_SC_C_LANG_SUPPORT_R, 
# 367
_SC_CLOCK_SELECTION, 
# 369
_SC_CPUTIME, 
# 371
_SC_THREAD_CPUTIME, 
# 373
_SC_DEVICE_IO, 
# 375
_SC_DEVICE_SPECIFIC, 
# 377
_SC_DEVICE_SPECIFIC_R, 
# 379
_SC_FD_MGMT, 
# 381
_SC_FIFO, 
# 383
_SC_PIPE, 
# 385
_SC_FILE_ATTRIBUTES, 
# 387
_SC_FILE_LOCKING, 
# 389
_SC_FILE_SYSTEM, 
# 391
_SC_MONOTONIC_CLOCK, 
# 393
_SC_MULTI_PROCESS, 
# 395
_SC_SINGLE_PROCESS, 
# 397
_SC_NETWORKING, 
# 399
_SC_READER_WRITER_LOCKS, 
# 401
_SC_SPIN_LOCKS, 
# 403
_SC_REGEXP, 
# 405
_SC_REGEX_VERSION, 
# 407
_SC_SHELL, 
# 409
_SC_SIGNALS, 
# 411
_SC_SPAWN, 
# 413
_SC_SPORADIC_SERVER, 
# 415
_SC_THREAD_SPORADIC_SERVER, 
# 417
_SC_SYSTEM_DATABASE, 
# 419
_SC_SYSTEM_DATABASE_R, 
# 421
_SC_TIMEOUTS, 
# 423
_SC_TYPED_MEMORY_OBJECTS, 
# 425
_SC_USER_GROUPS, 
# 427
_SC_USER_GROUPS_R, 
# 429
_SC_2_PBS, 
# 431
_SC_2_PBS_ACCOUNTING, 
# 433
_SC_2_PBS_LOCATE, 
# 435
_SC_2_PBS_MESSAGE, 
# 437
_SC_2_PBS_TRACK, 
# 439
_SC_SYMLOOP_MAX, 
# 441
_SC_STREAMS, 
# 443
_SC_2_PBS_CHECKPOINT, 
# 446
_SC_V6_ILP32_OFF32, 
# 448
_SC_V6_ILP32_OFFBIG, 
# 450
_SC_V6_LP64_OFF64, 
# 452
_SC_V6_LPBIG_OFFBIG, 
# 455
_SC_HOST_NAME_MAX, 
# 457
_SC_TRACE, 
# 459
_SC_TRACE_EVENT_FILTER, 
# 461
_SC_TRACE_INHERIT, 
# 463
_SC_TRACE_LOG, 
# 466
_SC_LEVEL1_ICACHE_SIZE, 
# 468
_SC_LEVEL1_ICACHE_ASSOC, 
# 470
_SC_LEVEL1_ICACHE_LINESIZE, 
# 472
_SC_LEVEL1_DCACHE_SIZE, 
# 474
_SC_LEVEL1_DCACHE_ASSOC, 
# 476
_SC_LEVEL1_DCACHE_LINESIZE, 
# 478
_SC_LEVEL2_CACHE_SIZE, 
# 480
_SC_LEVEL2_CACHE_ASSOC, 
# 482
_SC_LEVEL2_CACHE_LINESIZE, 
# 484
_SC_LEVEL3_CACHE_SIZE, 
# 486
_SC_LEVEL3_CACHE_ASSOC, 
# 488
_SC_LEVEL3_CACHE_LINESIZE, 
# 490
_SC_LEVEL4_CACHE_SIZE, 
# 492
_SC_LEVEL4_CACHE_ASSOC, 
# 494
_SC_LEVEL4_CACHE_LINESIZE, 
# 498
_SC_IPV6 = 235, 
# 500
_SC_RAW_SOCKETS, 
# 503
_SC_V7_ILP32_OFF32, 
# 505
_SC_V7_ILP32_OFFBIG, 
# 507
_SC_V7_LP64_OFF64, 
# 509
_SC_V7_LPBIG_OFFBIG, 
# 512
_SC_SS_REPL_MAX, 
# 515
_SC_TRACE_EVENT_NAME_MAX, 
# 517
_SC_TRACE_NAME_MAX, 
# 519
_SC_TRACE_SYS_MAX, 
# 521
_SC_TRACE_USER_EVENT_MAX, 
# 524
_SC_XOPEN_STREAMS, 
# 527
_SC_THREAD_ROBUST_PRIO_INHERIT, 
# 529
_SC_THREAD_ROBUST_PRIO_PROTECT
# 531
}; 
# 535
enum { 
# 536
_CS_PATH, 
# 539
_CS_V6_WIDTH_RESTRICTED_ENVS, 
# 543
_CS_GNU_LIBC_VERSION, 
# 545
_CS_GNU_LIBPTHREAD_VERSION, 
# 548
_CS_V5_WIDTH_RESTRICTED_ENVS, 
# 552
_CS_V7_WIDTH_RESTRICTED_ENVS, 
# 556
_CS_LFS_CFLAGS = 1000, 
# 558
_CS_LFS_LDFLAGS, 
# 560
_CS_LFS_LIBS, 
# 562
_CS_LFS_LINTFLAGS, 
# 564
_CS_LFS64_CFLAGS, 
# 566
_CS_LFS64_LDFLAGS, 
# 568
_CS_LFS64_LIBS, 
# 570
_CS_LFS64_LINTFLAGS, 
# 573
_CS_XBS5_ILP32_OFF32_CFLAGS = 1100, 
# 575
_CS_XBS5_ILP32_OFF32_LDFLAGS, 
# 577
_CS_XBS5_ILP32_OFF32_LIBS, 
# 579
_CS_XBS5_ILP32_OFF32_LINTFLAGS, 
# 581
_CS_XBS5_ILP32_OFFBIG_CFLAGS, 
# 583
_CS_XBS5_ILP32_OFFBIG_LDFLAGS, 
# 585
_CS_XBS5_ILP32_OFFBIG_LIBS, 
# 587
_CS_XBS5_ILP32_OFFBIG_LINTFLAGS, 
# 589
_CS_XBS5_LP64_OFF64_CFLAGS, 
# 591
_CS_XBS5_LP64_OFF64_LDFLAGS, 
# 593
_CS_XBS5_LP64_OFF64_LIBS, 
# 595
_CS_XBS5_LP64_OFF64_LINTFLAGS, 
# 597
_CS_XBS5_LPBIG_OFFBIG_CFLAGS, 
# 599
_CS_XBS5_LPBIG_OFFBIG_LDFLAGS, 
# 601
_CS_XBS5_LPBIG_OFFBIG_LIBS, 
# 603
_CS_XBS5_LPBIG_OFFBIG_LINTFLAGS, 
# 606
_CS_POSIX_V6_ILP32_OFF32_CFLAGS, 
# 608
_CS_POSIX_V6_ILP32_OFF32_LDFLAGS, 
# 610
_CS_POSIX_V6_ILP32_OFF32_LIBS, 
# 612
_CS_POSIX_V6_ILP32_OFF32_LINTFLAGS, 
# 614
_CS_POSIX_V6_ILP32_OFFBIG_CFLAGS, 
# 616
_CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS, 
# 618
_CS_POSIX_V6_ILP32_OFFBIG_LIBS, 
# 620
_CS_POSIX_V6_ILP32_OFFBIG_LINTFLAGS, 
# 622
_CS_POSIX_V6_LP64_OFF64_CFLAGS, 
# 624
_CS_POSIX_V6_LP64_OFF64_LDFLAGS, 
# 626
_CS_POSIX_V6_LP64_OFF64_LIBS, 
# 628
_CS_POSIX_V6_LP64_OFF64_LINTFLAGS, 
# 630
_CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS, 
# 632
_CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS, 
# 634
_CS_POSIX_V6_LPBIG_OFFBIG_LIBS, 
# 636
_CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS, 
# 639
_CS_POSIX_V7_ILP32_OFF32_CFLAGS, 
# 641
_CS_POSIX_V7_ILP32_OFF32_LDFLAGS, 
# 643
_CS_POSIX_V7_ILP32_OFF32_LIBS, 
# 645
_CS_POSIX_V7_ILP32_OFF32_LINTFLAGS, 
# 647
_CS_POSIX_V7_ILP32_OFFBIG_CFLAGS, 
# 649
_CS_POSIX_V7_ILP32_OFFBIG_LDFLAGS, 
# 651
_CS_POSIX_V7_ILP32_OFFBIG_LIBS, 
# 653
_CS_POSIX_V7_ILP32_OFFBIG_LINTFLAGS, 
# 655
_CS_POSIX_V7_LP64_OFF64_CFLAGS, 
# 657
_CS_POSIX_V7_LP64_OFF64_LDFLAGS, 
# 659
_CS_POSIX_V7_LP64_OFF64_LIBS, 
# 661
_CS_POSIX_V7_LP64_OFF64_LINTFLAGS, 
# 663
_CS_POSIX_V7_LPBIG_OFFBIG_CFLAGS, 
# 665
_CS_POSIX_V7_LPBIG_OFFBIG_LDFLAGS, 
# 667
_CS_POSIX_V7_LPBIG_OFFBIG_LIBS, 
# 669
_CS_POSIX_V7_LPBIG_OFFBIG_LINTFLAGS, 
# 672
_CS_V6_ENV, 
# 674
_CS_V7_ENV
# 676
}; 
# 612 "/usr/include/unistd.h" 3
extern long pathconf(const char * __path, int __name) throw()
# 613
 __attribute((__nonnull__(1))); 
# 616
extern long fpathconf(int __fd, int __name) throw(); 
# 619
extern long sysconf(int __name) throw(); 
# 623
extern size_t confstr(int __name, char * __buf, size_t __len) throw(); 
# 628
extern __pid_t getpid() throw(); 
# 631
extern __pid_t getppid() throw(); 
# 636
extern __pid_t getpgrp() throw(); 
# 646 "/usr/include/unistd.h" 3
extern __pid_t __getpgid(__pid_t __pid) throw(); 
# 648
extern __pid_t getpgid(__pid_t __pid) throw(); 
# 655
extern int setpgid(__pid_t __pid, __pid_t __pgid) throw(); 
# 672 "/usr/include/unistd.h" 3
extern int setpgrp() throw(); 
# 689 "/usr/include/unistd.h" 3
extern __pid_t setsid() throw(); 
# 693
extern __pid_t getsid(__pid_t __pid) throw(); 
# 697
extern __uid_t getuid() throw(); 
# 700
extern __uid_t geteuid() throw(); 
# 703
extern __gid_t getgid() throw(); 
# 706
extern __gid_t getegid() throw(); 
# 711
extern int getgroups(int __size, __gid_t  __list[]) throw(); 
# 715
extern int group_member(__gid_t __gid) throw(); 
# 722
extern int setuid(__uid_t __uid) throw(); 
# 727
extern int setreuid(__uid_t __ruid, __uid_t __euid) throw(); 
# 732
extern int seteuid(__uid_t __uid) throw(); 
# 739
extern int setgid(__gid_t __gid) throw(); 
# 744
extern int setregid(__gid_t __rgid, __gid_t __egid) throw(); 
# 749
extern int setegid(__gid_t __gid) throw(); 
# 755
extern int getresuid(__uid_t * __ruid, __uid_t * __euid, __uid_t * __suid) throw(); 
# 760
extern int getresgid(__gid_t * __rgid, __gid_t * __egid, __gid_t * __sgid) throw(); 
# 765
extern int setresuid(__uid_t __ruid, __uid_t __euid, __uid_t __suid) throw(); 
# 770
extern int setresgid(__gid_t __rgid, __gid_t __egid, __gid_t __sgid) throw(); 
# 778
extern __pid_t fork() throw(); 
# 786
extern __pid_t vfork() throw(); 
# 792
extern char *ttyname(int __fd) throw(); 
# 796
extern int ttyname_r(int __fd, char * __buf, size_t __buflen) throw()
# 797
 __attribute((__nonnull__(2))); 
# 801
extern int isatty(int __fd) throw(); 
# 807
extern int ttyslot() throw(); 
# 812
extern int link(const char * __from, const char * __to) throw()
# 813
 __attribute((__nonnull__(1, 2))); 
# 818
extern int linkat(int __fromfd, const char * __from, int __tofd, const char * __to, int __flags) throw()
# 820
 __attribute((__nonnull__(2, 4))); 
# 825
extern int symlink(const char * __from, const char * __to) throw()
# 826
 __attribute((__nonnull__(1, 2))); 
# 831
extern ssize_t readlink(const char *__restrict__ __path, char *__restrict__ __buf, size_t __len) throw()
# 833
 __attribute((__nonnull__(1, 2))); 
# 838
extern int symlinkat(const char * __from, int __tofd, const char * __to) throw()
# 839
 __attribute((__nonnull__(1, 3))); 
# 842
extern ssize_t readlinkat(int __fd, const char *__restrict__ __path, char *__restrict__ __buf, size_t __len) throw()
# 844
 __attribute((__nonnull__(2, 3))); 
# 848
extern int unlink(const char * __name) throw() __attribute((__nonnull__(1))); 
# 852
extern int unlinkat(int __fd, const char * __name, int __flag) throw()
# 853
 __attribute((__nonnull__(2))); 
# 857
extern int rmdir(const char * __path) throw() __attribute((__nonnull__(1))); 
# 861
extern __pid_t tcgetpgrp(int __fd) throw(); 
# 864
extern int tcsetpgrp(int __fd, __pid_t __pgrp_id) throw(); 
# 871
extern char *getlogin(); 
# 879
extern int getlogin_r(char * __name, size_t __name_len) __attribute((__nonnull__(1))); 
# 884
extern int setlogin(const char * __name) throw() __attribute((__nonnull__(1))); 
# 49 "/usr/include/getopt.h" 3
extern "C" {
# 58 "/usr/include/getopt.h" 3
extern char *optarg; 
# 72 "/usr/include/getopt.h" 3
extern int optind; 
# 77
extern int opterr; 
# 81
extern int optopt; 
# 151 "/usr/include/getopt.h" 3
extern int getopt(int ___argc, char *const * ___argv, const char * __shortopts) throw(); 
# 186 "/usr/include/getopt.h" 3
}
# 901 "/usr/include/unistd.h" 3
extern int gethostname(char * __name, size_t __len) throw() __attribute((__nonnull__(1))); 
# 908
extern int sethostname(const char * __name, size_t __len) throw()
# 909
 __attribute((__nonnull__(1))); 
# 913
extern int sethostid(long __id) throw(); 
# 919
extern int getdomainname(char * __name, size_t __len) throw()
# 920
 __attribute((__nonnull__(1))); 
# 921
extern int setdomainname(const char * __name, size_t __len) throw()
# 922
 __attribute((__nonnull__(1))); 
# 928
extern int vhangup() throw(); 
# 931
extern int revoke(const char * __file) throw() __attribute((__nonnull__(1))); 
# 939
extern int profil(unsigned short * __sample_buffer, size_t __size, size_t __offset, unsigned __scale) throw()
# 941
 __attribute((__nonnull__(1))); 
# 947
extern int acct(const char * __name) throw(); 
# 951
extern char *getusershell() throw(); 
# 952
extern void endusershell() throw(); 
# 953
extern void setusershell() throw(); 
# 959
extern int daemon(int __nochdir, int __noclose) throw(); 
# 966
extern int chroot(const char * __path) throw() __attribute((__nonnull__(1))); 
# 970
extern char *getpass(const char * __prompt) __attribute((__nonnull__(1))); 
# 978
extern int fsync(int __fd); 
# 984
extern int syncfs(int __fd) throw(); 
# 991
extern long gethostid(); 
# 994
extern void sync() throw(); 
# 1000
extern int getpagesize() throw() __attribute((const)); 
# 1005
extern int getdtablesize() throw(); 
# 1015 "/usr/include/unistd.h" 3
extern int truncate(const char * __file, __off_t __length) throw()
# 1016
 __attribute((__nonnull__(1))); 
# 1027 "/usr/include/unistd.h" 3
extern int truncate64(const char * __file, __off64_t __length) throw()
# 1028
 __attribute((__nonnull__(1))); 
# 1038 "/usr/include/unistd.h" 3
extern int ftruncate(int __fd, __off_t __length) throw(); 
# 1048 "/usr/include/unistd.h" 3
extern int ftruncate64(int __fd, __off64_t __length) throw(); 
# 1059 "/usr/include/unistd.h" 3
extern int brk(void * __addr) throw(); 
# 1065
extern void *sbrk(intptr_t __delta) throw(); 
# 1080 "/usr/include/unistd.h" 3
extern long syscall(long __sysno, ...) throw(); 
# 1103 "/usr/include/unistd.h" 3
extern int lockf(int __fd, int __cmd, __off_t __len); 
# 1113 "/usr/include/unistd.h" 3
extern int lockf64(int __fd, int __cmd, __off64_t __len); 
# 1134 "/usr/include/unistd.h" 3
extern int fdatasync(int __fildes); 
# 1142
extern char *crypt(const char * __key, const char * __salt) throw()
# 1143
 __attribute((__nonnull__(1, 2))); 
# 1147
extern void encrypt(char * __block, int __edflag) throw() __attribute((__nonnull__(1))); 
# 1154
extern void swab(const void *__restrict__ __from, void *__restrict__ __to, ssize_t __n) throw()
# 1155
 __attribute((__nonnull__(1, 2))); 
# 1172 "/usr/include/unistd.h" 3
}
# 48 "/usr/include/stdint.h" 3
typedef unsigned char uint8_t; 
# 49
typedef unsigned short uint16_t; 
# 51
typedef unsigned uint32_t; 
# 55
typedef unsigned long uint64_t; 
# 65 "/usr/include/stdint.h" 3
typedef signed char int_least8_t; 
# 66
typedef short int_least16_t; 
# 67
typedef int int_least32_t; 
# 69
typedef long int_least64_t; 
# 76
typedef unsigned char uint_least8_t; 
# 77
typedef unsigned short uint_least16_t; 
# 78
typedef unsigned uint_least32_t; 
# 80
typedef unsigned long uint_least64_t; 
# 90 "/usr/include/stdint.h" 3
typedef signed char int_fast8_t; 
# 92
typedef long int_fast16_t; 
# 93
typedef long int_fast32_t; 
# 94
typedef long int_fast64_t; 
# 103 "/usr/include/stdint.h" 3
typedef unsigned char uint_fast8_t; 
# 105
typedef unsigned long uint_fast16_t; 
# 106
typedef unsigned long uint_fast32_t; 
# 107
typedef unsigned long uint_fast64_t; 
# 122 "/usr/include/stdint.h" 3
typedef unsigned long uintptr_t; 
# 134 "/usr/include/stdint.h" 3
typedef long intmax_t; 
# 135
typedef unsigned long uintmax_t; 
# 59 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef uint32_t cuuint32_t; 
# 60
typedef uint64_t cuuint64_t; 
# 240 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
extern "C" {
# 250 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef unsigned long long CUdeviceptr; 
# 257
typedef int CUdevice; 
# 258
typedef struct CUctx_st *CUcontext; 
# 259
typedef struct CUmod_st *CUmodule; 
# 260
typedef struct CUfunc_st *CUfunction; 
# 261
typedef struct CUarray_st *CUarray; 
# 262
typedef struct CUmipmappedArray_st *CUmipmappedArray; 
# 263
typedef struct CUtexref_st *CUtexref; 
# 264
typedef struct CUsurfref_st *CUsurfref; 
# 265
typedef CUevent_st *CUevent; 
# 266
typedef CUstream_st *CUstream; 
# 267
typedef struct CUgraphicsResource_st *CUgraphicsResource; 
# 268
typedef unsigned long long CUtexObject; 
# 269
typedef unsigned long long CUsurfObject; 
# 270
typedef struct CUextMemory_st *CUexternalMemory; 
# 271
typedef struct CUextSemaphore_st *CUexternalSemaphore; 
# 272
typedef CUgraph_st *CUgraph; 
# 273
typedef CUgraphNode_st *CUgraphNode; 
# 274
typedef CUgraphExec_st *CUgraphExec; 
# 295 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef 
# 293
struct CUipcEventHandle_st { 
# 294
char reserved[64]; 
# 295
} CUipcEventHandle; 
# 302
typedef 
# 300
struct CUipcMemHandle_st { 
# 301
char reserved[64]; 
# 302
} CUipcMemHandle; 
# 309
typedef 
# 307
enum CUipcMem_flags_enum { 
# 308
CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1
# 309
} CUipcMem_flags; 
# 320
typedef 
# 316
enum CUmemAttach_flags_enum { 
# 317
CU_MEM_ATTACH_GLOBAL = 1, 
# 318
CU_MEM_ATTACH_HOST, 
# 319
CU_MEM_ATTACH_SINGLE = 4
# 320
} CUmemAttach_flags; 
# 337
typedef 
# 325
enum CUctx_flags_enum { 
# 326
CU_CTX_SCHED_AUTO, 
# 327
CU_CTX_SCHED_SPIN, 
# 328
CU_CTX_SCHED_YIELD, 
# 329
CU_CTX_SCHED_BLOCKING_SYNC = 4, 
# 330
CU_CTX_BLOCKING_SYNC = 4, 
# 333
CU_CTX_SCHED_MASK = 7, 
# 334
CU_CTX_MAP_HOST, 
# 335
CU_CTX_LMEM_RESIZE_TO_MAX = 16, 
# 336
CU_CTX_FLAGS_MASK = 31
# 337
} CUctx_flags; 
# 345
typedef 
# 342
enum CUstream_flags_enum { 
# 343
CU_STREAM_DEFAULT, 
# 344
CU_STREAM_NON_BLOCKING
# 345
} CUstream_flags; 
# 375 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef 
# 370
enum CUevent_flags_enum { 
# 371
CU_EVENT_DEFAULT, 
# 372
CU_EVENT_BLOCKING_SYNC, 
# 373
CU_EVENT_DISABLE_TIMING, 
# 374
CU_EVENT_INTERPROCESS = 4
# 375
} CUevent_flags; 
# 399 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef 
# 381 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
enum CUstreamWaitValue_flags_enum { 
# 382
CU_STREAM_WAIT_VALUE_GEQ, 
# 385
CU_STREAM_WAIT_VALUE_EQ, 
# 386
CU_STREAM_WAIT_VALUE_AND, 
# 387
CU_STREAM_WAIT_VALUE_NOR, 
# 390
CU_STREAM_WAIT_VALUE_FLUSH = 1073741824
# 399 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
} CUstreamWaitValue_flags; 
# 412
typedef 
# 404
enum CUstreamWriteValue_flags_enum { 
# 405
CU_STREAM_WRITE_VALUE_DEFAULT, 
# 406
CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER
# 412
} CUstreamWriteValue_flags; 
# 424
typedef 
# 417
enum CUstreamBatchMemOpType_enum { 
# 418
CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1, 
# 419
CU_STREAM_MEM_OP_WRITE_VALUE_32, 
# 420
CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4, 
# 421
CU_STREAM_MEM_OP_WRITE_VALUE_64, 
# 422
CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3
# 424
} CUstreamBatchMemOpType; 
# 456
typedef 
# 429
union CUstreamBatchMemOpParams_union { 
# 430
CUstreamBatchMemOpType operation; 
# 431
struct CUstreamMemOpWaitValueParams_st { 
# 432
CUstreamBatchMemOpType operation; 
# 433
CUdeviceptr address; 
# 434
union { 
# 435
cuuint32_t value; 
# 436
cuuint64_t value64; 
# 437
}; 
# 438
unsigned flags; 
# 439
CUdeviceptr alias; 
# 440
} waitValue; 
# 441
struct CUstreamMemOpWriteValueParams_st { 
# 442
CUstreamBatchMemOpType operation; 
# 443
CUdeviceptr address; 
# 444
union { 
# 445
cuuint32_t value; 
# 446
cuuint64_t value64; 
# 447
}; 
# 448
unsigned flags; 
# 449
CUdeviceptr alias; 
# 450
} writeValue; 
# 451
struct CUstreamMemOpFlushRemoteWritesParams_st { 
# 452
CUstreamBatchMemOpType operation; 
# 453
unsigned flags; 
# 454
} flushRemoteWrites; 
# 455
cuuint64_t pad[6]; 
# 456
} CUstreamBatchMemOpParams; 
# 465
typedef 
# 462
enum CUoccupancy_flags_enum { 
# 463
CU_OCCUPANCY_DEFAULT, 
# 464
CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE
# 465
} CUoccupancy_flags; 
# 479
typedef 
# 470
enum CUarray_format_enum { 
# 471
CU_AD_FORMAT_UNSIGNED_INT8 = 1, 
# 472
CU_AD_FORMAT_UNSIGNED_INT16, 
# 473
CU_AD_FORMAT_UNSIGNED_INT32, 
# 474
CU_AD_FORMAT_SIGNED_INT8 = 8, 
# 475
CU_AD_FORMAT_SIGNED_INT16, 
# 476
CU_AD_FORMAT_SIGNED_INT32, 
# 477
CU_AD_FORMAT_HALF = 16, 
# 478
CU_AD_FORMAT_FLOAT = 32
# 479
} CUarray_format; 
# 489
typedef 
# 484
enum CUaddress_mode_enum { 
# 485
CU_TR_ADDRESS_MODE_WRAP, 
# 486
CU_TR_ADDRESS_MODE_CLAMP, 
# 487
CU_TR_ADDRESS_MODE_MIRROR, 
# 488
CU_TR_ADDRESS_MODE_BORDER
# 489
} CUaddress_mode; 
# 497
typedef 
# 494
enum CUfilter_mode_enum { 
# 495
CU_TR_FILTER_MODE_POINT, 
# 496
CU_TR_FILTER_MODE_LINEAR
# 497
} CUfilter_mode; 
# 610
typedef 
# 502
enum CUdevice_attribute_enum { 
# 503
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1, 
# 504
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 
# 505
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, 
# 506
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, 
# 507
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, 
# 508
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, 
# 509
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, 
# 510
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, 
# 511
CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8, 
# 512
CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, 
# 513
CU_DEVICE_ATTRIBUTE_WARP_SIZE, 
# 514
CU_DEVICE_ATTRIBUTE_MAX_PITCH, 
# 515
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, 
# 516
CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12, 
# 517
CU_DEVICE_ATTRIBUTE_CLOCK_RATE, 
# 518
CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, 
# 519
CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, 
# 520
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 
# 521
CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, 
# 522
CU_DEVICE_ATTRIBUTE_INTEGRATED, 
# 523
CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, 
# 524
CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, 
# 525
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, 
# 526
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, 
# 527
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, 
# 528
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, 
# 529
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, 
# 530
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, 
# 531
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, 
# 532
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, 
# 533
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, 
# 534
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27, 
# 535
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT, 
# 536
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES, 
# 537
CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, 
# 538
CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, 
# 539
CU_DEVICE_ATTRIBUTE_ECC_ENABLED, 
# 540
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, 
# 541
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, 
# 542
CU_DEVICE_ATTRIBUTE_TCC_DRIVER, 
# 543
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, 
# 544
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, 
# 545
CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 
# 546
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, 
# 547
CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, 
# 548
CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, 
# 549
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, 
# 550
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, 
# 551
CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER, 
# 552
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH, 
# 553
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT, 
# 554
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE, 
# 555
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE, 
# 556
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE, 
# 557
CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, 
# 558
CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, 
# 559
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH, 
# 560
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH, 
# 561
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS, 
# 562
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH, 
# 563
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH, 
# 564
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT, 
# 565
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH, 
# 566
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT, 
# 567
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH, 
# 568
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH, 
# 569
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS, 
# 570
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH, 
# 571
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT, 
# 572
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS, 
# 573
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH, 
# 574
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH, 
# 575
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS, 
# 576
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH, 
# 577
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH, 
# 578
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, 
# 579
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH, 
# 580
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH, 
# 581
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT, 
# 582
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 
# 583
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 
# 584
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH, 
# 585
CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED, 
# 586
CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, 
# 587
CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED, 
# 588
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, 
# 589
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, 
# 590
CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, 
# 591
CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, 
# 592
CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, 
# 593
CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, 
# 594
CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, 
# 595
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, 
# 596
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, 
# 597
CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, 
# 598
CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, 
# 599
CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, 
# 600
CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, 
# 601
CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR, 
# 602
CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, 
# 603
CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH, 
# 604
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, 
# 605
CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, 
# 606
CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED, 
# 607
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, 
# 608
CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, 
# 609
CU_DEVICE_ATTRIBUTE_MAX
# 610
} CUdevice_attribute; 
# 626
typedef 
# 615
struct CUdevprop_st { 
# 616
int maxThreadsPerBlock; 
# 617
int maxThreadsDim[3]; 
# 618
int maxGridSize[3]; 
# 619
int sharedMemPerBlock; 
# 620
int totalConstantMemory; 
# 621
int SIMDWidth; 
# 622
int memPitch; 
# 623
int regsPerBlock; 
# 624
int clockRate; 
# 625
int textureAlign; 
# 626
} CUdevprop; 
# 641
typedef 
# 631
enum CUpointer_attribute_enum { 
# 632
CU_POINTER_ATTRIBUTE_CONTEXT = 1, 
# 633
CU_POINTER_ATTRIBUTE_MEMORY_TYPE, 
# 634
CU_POINTER_ATTRIBUTE_DEVICE_POINTER, 
# 635
CU_POINTER_ATTRIBUTE_HOST_POINTER, 
# 636
CU_POINTER_ATTRIBUTE_P2P_TOKENS, 
# 637
CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, 
# 638
CU_POINTER_ATTRIBUTE_BUFFER_ID, 
# 639
CU_POINTER_ATTRIBUTE_IS_MANAGED, 
# 640
CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
# 641
} CUpointer_attribute; 
# 719 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef 
# 646 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
enum CUfunction_attribute_enum { 
# 652
CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 
# 659
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, 
# 665
CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, 
# 670
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, 
# 675
CU_FUNC_ATTRIBUTE_NUM_REGS, 
# 684 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CU_FUNC_ATTRIBUTE_PTX_VERSION, 
# 693 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CU_FUNC_ATTRIBUTE_BINARY_VERSION, 
# 699
CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, 
# 707
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 
# 716 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 
# 718
CU_FUNC_ATTRIBUTE_MAX
# 719
} CUfunction_attribute; 
# 729
typedef 
# 724
enum CUfunc_cache_enum { 
# 725
CU_FUNC_CACHE_PREFER_NONE, 
# 726
CU_FUNC_CACHE_PREFER_SHARED, 
# 727
CU_FUNC_CACHE_PREFER_L1, 
# 728
CU_FUNC_CACHE_PREFER_EQUAL
# 729
} CUfunc_cache; 
# 738
typedef 
# 734
enum CUsharedconfig_enum { 
# 735
CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE, 
# 736
CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE, 
# 737
CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE
# 738
} CUsharedconfig; 
# 747
typedef 
# 743
enum CUshared_carveout_enum { 
# 744
CU_SHAREDMEM_CARVEOUT_DEFAULT = (-1), 
# 745
CU_SHAREDMEM_CARVEOUT_MAX_SHARED = 100, 
# 746
CU_SHAREDMEM_CARVEOUT_MAX_L1 = 0
# 747
} CUshared_carveout; 
# 757
typedef 
# 752
enum CUmemorytype_enum { 
# 753
CU_MEMORYTYPE_HOST = 1, 
# 754
CU_MEMORYTYPE_DEVICE, 
# 755
CU_MEMORYTYPE_ARRAY, 
# 756
CU_MEMORYTYPE_UNIFIED
# 757
} CUmemorytype; 
# 766
typedef 
# 762
enum CUcomputemode_enum { 
# 763
CU_COMPUTEMODE_DEFAULT, 
# 764
CU_COMPUTEMODE_PROHIBITED = 2, 
# 765
CU_COMPUTEMODE_EXCLUSIVE_PROCESS
# 766
} CUcomputemode; 
# 778
typedef 
# 771
enum CUmem_advise_enum { 
# 772
CU_MEM_ADVISE_SET_READ_MOSTLY = 1, 
# 773
CU_MEM_ADVISE_UNSET_READ_MOSTLY, 
# 774
CU_MEM_ADVISE_SET_PREFERRED_LOCATION, 
# 775
CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, 
# 776
CU_MEM_ADVISE_SET_ACCESSED_BY, 
# 777
CU_MEM_ADVISE_UNSET_ACCESSED_BY
# 778
} CUmem_advise; 
# 785
typedef 
# 780
enum CUmem_range_attribute_enum { 
# 781
CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1, 
# 782
CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION, 
# 783
CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY, 
# 784
CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION
# 785
} CUmem_range_attribute; 
# 960 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef 
# 790 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
enum CUjit_option_enum { 
# 797
CU_JIT_MAX_REGISTERS, 
# 812 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CU_JIT_THREADS_PER_BLOCK, 
# 820
CU_JIT_WALL_TIME, 
# 829 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CU_JIT_INFO_LOG_BUFFER, 
# 838 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, 
# 847 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CU_JIT_ERROR_LOG_BUFFER, 
# 856 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, 
# 864
CU_JIT_OPTIMIZATION_LEVEL, 
# 872
CU_JIT_TARGET_FROM_CUCONTEXT, 
# 880
CU_JIT_TARGET, 
# 889 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CU_JIT_FALLBACK_STRATEGY, 
# 897
CU_JIT_GENERATE_DEBUG_INFO, 
# 904
CU_JIT_LOG_VERBOSE, 
# 911
CU_JIT_GENERATE_LINE_INFO, 
# 919
CU_JIT_CACHE_MODE, 
# 924
CU_JIT_NEW_SM3X_OPT, 
# 925
CU_JIT_FAST_COMPILE, 
# 939 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CU_JIT_GLOBAL_SYMBOL_NAMES, 
# 948 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CU_JIT_GLOBAL_SYMBOL_ADDRESSES, 
# 956
CU_JIT_GLOBAL_SYMBOL_COUNT, 
# 958
CU_JIT_NUM_OPTIONS
# 960
} CUjit_option; 
# 982
typedef 
# 965
enum CUjit_target_enum { 
# 967
CU_TARGET_COMPUTE_20 = 20, 
# 968
CU_TARGET_COMPUTE_21, 
# 969
CU_TARGET_COMPUTE_30 = 30, 
# 970
CU_TARGET_COMPUTE_32 = 32, 
# 971
CU_TARGET_COMPUTE_35 = 35, 
# 972
CU_TARGET_COMPUTE_37 = 37, 
# 973
CU_TARGET_COMPUTE_50 = 50, 
# 974
CU_TARGET_COMPUTE_52 = 52, 
# 975
CU_TARGET_COMPUTE_53, 
# 976
CU_TARGET_COMPUTE_60 = 60, 
# 977
CU_TARGET_COMPUTE_61, 
# 978
CU_TARGET_COMPUTE_62, 
# 979
CU_TARGET_COMPUTE_70 = 70, 
# 980
CU_TARGET_COMPUTE_72 = 72, 
# 981
CU_TARGET_COMPUTE_75 = 75
# 982
} CUjit_target; 
# 993
typedef 
# 987
enum CUjit_fallback_enum { 
# 989
CU_PREFER_PTX, 
# 991
CU_PREFER_BINARY
# 993
} CUjit_fallback; 
# 1003
typedef 
# 998
enum CUjit_cacheMode_enum { 
# 1000
CU_JIT_CACHE_OPTION_NONE, 
# 1001
CU_JIT_CACHE_OPTION_CG, 
# 1002
CU_JIT_CACHE_OPTION_CA
# 1003
} CUjit_cacheMode; 
# 1041
typedef 
# 1008
enum CUjitInputType_enum { 
# 1014
CU_JIT_INPUT_CUBIN, 
# 1020
CU_JIT_INPUT_PTX, 
# 1026
CU_JIT_INPUT_FATBINARY, 
# 1032
CU_JIT_INPUT_OBJECT, 
# 1038
CU_JIT_INPUT_LIBRARY, 
# 1040
CU_JIT_NUM_INPUT_TYPES
# 1041
} CUjitInputType; 
# 1044
typedef struct CUlinkState_st *CUlinkState; 
# 1056
typedef 
# 1050
enum CUgraphicsRegisterFlags_enum { 
# 1051
CU_GRAPHICS_REGISTER_FLAGS_NONE, 
# 1052
CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY, 
# 1053
CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD, 
# 1054
CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4, 
# 1055
CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 8
# 1056
} CUgraphicsRegisterFlags; 
# 1065
typedef 
# 1061
enum CUgraphicsMapResourceFlags_enum { 
# 1062
CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE, 
# 1063
CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY, 
# 1064
CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD
# 1065
} CUgraphicsMapResourceFlags; 
# 1077
typedef 
# 1070
enum CUarray_cubemap_face_enum { 
# 1071
CU_CUBEMAP_FACE_POSITIVE_X, 
# 1072
CU_CUBEMAP_FACE_NEGATIVE_X, 
# 1073
CU_CUBEMAP_FACE_POSITIVE_Y, 
# 1074
CU_CUBEMAP_FACE_NEGATIVE_Y, 
# 1075
CU_CUBEMAP_FACE_POSITIVE_Z, 
# 1076
CU_CUBEMAP_FACE_NEGATIVE_Z
# 1077
} CUarray_cubemap_face; 
# 1090
typedef 
# 1082
enum CUlimit_enum { 
# 1083
CU_LIMIT_STACK_SIZE, 
# 1084
CU_LIMIT_PRINTF_FIFO_SIZE, 
# 1085
CU_LIMIT_MALLOC_HEAP_SIZE, 
# 1086
CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH, 
# 1087
CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT, 
# 1088
CU_LIMIT_MAX_L2_FETCH_GRANULARITY, 
# 1089
CU_LIMIT_MAX
# 1090
} CUlimit; 
# 1100
typedef 
# 1095
enum CUresourcetype_enum { 
# 1096
CU_RESOURCE_TYPE_ARRAY, 
# 1097
CU_RESOURCE_TYPE_MIPMAPPED_ARRAY, 
# 1098
CU_RESOURCE_TYPE_LINEAR, 
# 1099
CU_RESOURCE_TYPE_PITCH2D
# 1100
} CUresourcetype; 
# 1114 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef void (*CUhostFn)(void * userData); 
# 1130
typedef 
# 1119
struct CUDA_KERNEL_NODE_PARAMS_st { 
# 1120
CUfunction func; 
# 1121
unsigned gridDimX; 
# 1122
unsigned gridDimY; 
# 1123
unsigned gridDimZ; 
# 1124
unsigned blockDimX; 
# 1125
unsigned blockDimY; 
# 1126
unsigned blockDimZ; 
# 1127
unsigned sharedMemBytes; 
# 1128
void **kernelParams; 
# 1129
void **extra; 
# 1130
} CUDA_KERNEL_NODE_PARAMS; 
# 1142
typedef 
# 1135
struct CUDA_MEMSET_NODE_PARAMS_st { 
# 1136
CUdeviceptr dst; 
# 1137
size_t pitch; 
# 1138
unsigned value; 
# 1139
unsigned elementSize; 
# 1140
size_t width; 
# 1141
size_t height; 
# 1142
} CUDA_MEMSET_NODE_PARAMS; 
# 1150
typedef 
# 1147
struct CUDA_HOST_NODE_PARAMS_st { 
# 1148
CUhostFn fn; 
# 1149
void *userData; 
# 1150
} CUDA_HOST_NODE_PARAMS; 
# 1163
typedef 
# 1155
enum CUgraphNodeType_enum { 
# 1156
CU_GRAPH_NODE_TYPE_KERNEL, 
# 1157
CU_GRAPH_NODE_TYPE_MEMCPY, 
# 1158
CU_GRAPH_NODE_TYPE_MEMSET, 
# 1159
CU_GRAPH_NODE_TYPE_HOST, 
# 1160
CU_GRAPH_NODE_TYPE_GRAPH, 
# 1161
CU_GRAPH_NODE_TYPE_EMPTY, 
# 1162
CU_GRAPH_NODE_TYPE_COUNT
# 1163
} CUgraphNodeType; 
# 1173
typedef 
# 1168
enum CUstreamCaptureStatus_enum { 
# 1169
CU_STREAM_CAPTURE_STATUS_NONE, 
# 1170
CU_STREAM_CAPTURE_STATUS_ACTIVE, 
# 1171
CU_STREAM_CAPTURE_STATUS_INVALIDATED
# 1173
} CUstreamCaptureStatus; 
# 1187 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef 
# 1183
enum CUstreamCaptureMode_enum { 
# 1184
CU_STREAM_CAPTURE_MODE_GLOBAL, 
# 1185
CU_STREAM_CAPTURE_MODE_THREAD_LOCAL, 
# 1186
CU_STREAM_CAPTURE_MODE_RELAXED
# 1187
} CUstreamCaptureMode; 
# 1690 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef 
# 1194 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
enum cudaError_enum { 
# 1200
CUDA_SUCCESS, 
# 1206
CUDA_ERROR_INVALID_VALUE, 
# 1212
CUDA_ERROR_OUT_OF_MEMORY, 
# 1218
CUDA_ERROR_NOT_INITIALIZED, 
# 1223
CUDA_ERROR_DEINITIALIZED, 
# 1230
CUDA_ERROR_PROFILER_DISABLED, 
# 1238
CUDA_ERROR_PROFILER_NOT_INITIALIZED, 
# 1245
CUDA_ERROR_PROFILER_ALREADY_STARTED, 
# 1252
CUDA_ERROR_PROFILER_ALREADY_STOPPED, 
# 1258
CUDA_ERROR_NO_DEVICE = 100, 
# 1264
CUDA_ERROR_INVALID_DEVICE, 
# 1271
CUDA_ERROR_INVALID_IMAGE = 200, 
# 1281 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUDA_ERROR_INVALID_CONTEXT, 
# 1290 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUDA_ERROR_CONTEXT_ALREADY_CURRENT, 
# 1295
CUDA_ERROR_MAP_FAILED = 205, 
# 1300
CUDA_ERROR_UNMAP_FAILED, 
# 1306
CUDA_ERROR_ARRAY_IS_MAPPED, 
# 1311
CUDA_ERROR_ALREADY_MAPPED, 
# 1319
CUDA_ERROR_NO_BINARY_FOR_GPU, 
# 1324
CUDA_ERROR_ALREADY_ACQUIRED, 
# 1329
CUDA_ERROR_NOT_MAPPED, 
# 1335
CUDA_ERROR_NOT_MAPPED_AS_ARRAY, 
# 1341
CUDA_ERROR_NOT_MAPPED_AS_POINTER, 
# 1347
CUDA_ERROR_ECC_UNCORRECTABLE, 
# 1353
CUDA_ERROR_UNSUPPORTED_LIMIT, 
# 1360
CUDA_ERROR_CONTEXT_ALREADY_IN_USE, 
# 1366
CUDA_ERROR_PEER_ACCESS_UNSUPPORTED, 
# 1371
CUDA_ERROR_INVALID_PTX, 
# 1376
CUDA_ERROR_INVALID_GRAPHICS_CONTEXT, 
# 1382
CUDA_ERROR_NVLINK_UNCORRECTABLE, 
# 1387
CUDA_ERROR_JIT_COMPILER_NOT_FOUND, 
# 1392
CUDA_ERROR_INVALID_SOURCE = 300, 
# 1397
CUDA_ERROR_FILE_NOT_FOUND, 
# 1402
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND, 
# 1407
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED, 
# 1412
CUDA_ERROR_OPERATING_SYSTEM, 
# 1418
CUDA_ERROR_INVALID_HANDLE = 400, 
# 1424
CUDA_ERROR_ILLEGAL_STATE, 
# 1430
CUDA_ERROR_NOT_FOUND = 500, 
# 1438
CUDA_ERROR_NOT_READY = 600, 
# 1447 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUDA_ERROR_ILLEGAL_ADDRESS = 700, 
# 1458 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, 
# 1468 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUDA_ERROR_LAUNCH_TIMEOUT, 
# 1474
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, 
# 1481
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED, 
# 1488
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, 
# 1494
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708, 
# 1501
CUDA_ERROR_CONTEXT_IS_DESTROYED, 
# 1509
CUDA_ERROR_ASSERT, 
# 1516
CUDA_ERROR_TOO_MANY_PEERS, 
# 1522
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED, 
# 1528
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, 
# 1537 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUDA_ERROR_HARDWARE_STACK_ERROR, 
# 1545
CUDA_ERROR_ILLEGAL_INSTRUCTION, 
# 1554 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUDA_ERROR_MISALIGNED_ADDRESS, 
# 1565 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUDA_ERROR_INVALID_ADDRESS_SPACE, 
# 1573
CUDA_ERROR_INVALID_PC, 
# 1584 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUDA_ERROR_LAUNCH_FAILED, 
# 1593 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE, 
# 1598
CUDA_ERROR_NOT_PERMITTED = 800, 
# 1604
CUDA_ERROR_NOT_SUPPORTED, 
# 1613 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUDA_ERROR_SYSTEM_NOT_READY, 
# 1620
CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, 
# 1629 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE, 
# 1635
CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900, 
# 1641
CUDA_ERROR_STREAM_CAPTURE_INVALIDATED, 
# 1647
CUDA_ERROR_STREAM_CAPTURE_MERGE, 
# 1652
CUDA_ERROR_STREAM_CAPTURE_UNMATCHED, 
# 1658
CUDA_ERROR_STREAM_CAPTURE_UNJOINED, 
# 1665
CUDA_ERROR_STREAM_CAPTURE_ISOLATION, 
# 1671
CUDA_ERROR_STREAM_CAPTURE_IMPLICIT, 
# 1677
CUDA_ERROR_CAPTURED_EVENT, 
# 1684
CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD, 
# 1689
CUDA_ERROR_UNKNOWN = 999
# 1690
} CUresult; 
# 1701
typedef 
# 1695
enum CUdevice_P2PAttribute_enum { 
# 1696
CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 1, 
# 1697
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED, 
# 1698
CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED, 
# 1699
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED, 
# 1700
CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 4
# 1701
} CUdevice_P2PAttribute; 
# 1709
typedef void (*CUstreamCallback)(CUstream hStream, CUresult status, void * userData); 
# 1717
typedef size_t (*CUoccupancyB2DSize)(int blockSize); 
# 1793 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef 
# 1772
struct CUDA_MEMCPY2D_st { 
# 1773
size_t srcXInBytes; 
# 1774
size_t srcY; 
# 1776
CUmemorytype srcMemoryType; 
# 1777
const void *srcHost; 
# 1778
CUdeviceptr srcDevice; 
# 1779
CUarray srcArray; 
# 1780
size_t srcPitch; 
# 1782
size_t dstXInBytes; 
# 1783
size_t dstY; 
# 1785
CUmemorytype dstMemoryType; 
# 1786
void *dstHost; 
# 1787
CUdeviceptr dstDevice; 
# 1788
CUarray dstArray; 
# 1789
size_t dstPitch; 
# 1791
size_t WidthInBytes; 
# 1792
size_t Height; 
# 1793
} CUDA_MEMCPY2D; 
# 1826
typedef 
# 1798
struct CUDA_MEMCPY3D_st { 
# 1799
size_t srcXInBytes; 
# 1800
size_t srcY; 
# 1801
size_t srcZ; 
# 1802
size_t srcLOD; 
# 1803
CUmemorytype srcMemoryType; 
# 1804
const void *srcHost; 
# 1805
CUdeviceptr srcDevice; 
# 1806
CUarray srcArray; 
# 1807
void *reserved0; 
# 1808
size_t srcPitch; 
# 1809
size_t srcHeight; 
# 1811
size_t dstXInBytes; 
# 1812
size_t dstY; 
# 1813
size_t dstZ; 
# 1814
size_t dstLOD; 
# 1815
CUmemorytype dstMemoryType; 
# 1816
void *dstHost; 
# 1817
CUdeviceptr dstDevice; 
# 1818
CUarray dstArray; 
# 1819
void *reserved1; 
# 1820
size_t dstPitch; 
# 1821
size_t dstHeight; 
# 1823
size_t WidthInBytes; 
# 1824
size_t Height; 
# 1825
size_t Depth; 
# 1826
} CUDA_MEMCPY3D; 
# 1859
typedef 
# 1831
struct CUDA_MEMCPY3D_PEER_st { 
# 1832
size_t srcXInBytes; 
# 1833
size_t srcY; 
# 1834
size_t srcZ; 
# 1835
size_t srcLOD; 
# 1836
CUmemorytype srcMemoryType; 
# 1837
const void *srcHost; 
# 1838
CUdeviceptr srcDevice; 
# 1839
CUarray srcArray; 
# 1840
CUcontext srcContext; 
# 1841
size_t srcPitch; 
# 1842
size_t srcHeight; 
# 1844
size_t dstXInBytes; 
# 1845
size_t dstY; 
# 1846
size_t dstZ; 
# 1847
size_t dstLOD; 
# 1848
CUmemorytype dstMemoryType; 
# 1849
void *dstHost; 
# 1850
CUdeviceptr dstDevice; 
# 1851
CUarray dstArray; 
# 1852
CUcontext dstContext; 
# 1853
size_t dstPitch; 
# 1854
size_t dstHeight; 
# 1856
size_t WidthInBytes; 
# 1857
size_t Height; 
# 1858
size_t Depth; 
# 1859
} CUDA_MEMCPY3D_PEER; 
# 1871
typedef 
# 1864
struct CUDA_ARRAY_DESCRIPTOR_st { 
# 1866
size_t Width; 
# 1867
size_t Height; 
# 1869
CUarray_format Format; 
# 1870
unsigned NumChannels; 
# 1871
} CUDA_ARRAY_DESCRIPTOR; 
# 1885
typedef 
# 1876
struct CUDA_ARRAY3D_DESCRIPTOR_st { 
# 1878
size_t Width; 
# 1879
size_t Height; 
# 1880
size_t Depth; 
# 1882
CUarray_format Format; 
# 1883
unsigned NumChannels; 
# 1884
unsigned Flags; 
# 1885
} CUDA_ARRAY3D_DESCRIPTOR; 
# 1925 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef 
# 1894
struct CUDA_RESOURCE_DESC_st { 
# 1896
CUresourcetype resType; 
# 1898
union { 
# 1899
struct { 
# 1900
CUarray hArray; 
# 1901
} array; 
# 1902
struct { 
# 1903
CUmipmappedArray hMipmappedArray; 
# 1904
} mipmap; 
# 1905
struct { 
# 1906
CUdeviceptr devPtr; 
# 1907
CUarray_format format; 
# 1908
unsigned numChannels; 
# 1909
size_t sizeInBytes; 
# 1910
} linear; 
# 1911
struct { 
# 1912
CUdeviceptr devPtr; 
# 1913
CUarray_format format; 
# 1914
unsigned numChannels; 
# 1915
size_t width; 
# 1916
size_t height; 
# 1917
size_t pitchInBytes; 
# 1918
} pitch2D; 
# 1919
struct { 
# 1920
int reserved[32]; 
# 1921
} reserved; 
# 1922
} res; 
# 1924
unsigned flags; 
# 1925
} CUDA_RESOURCE_DESC; 
# 1941
typedef 
# 1930
struct CUDA_TEXTURE_DESC_st { 
# 1931
CUaddress_mode addressMode[3]; 
# 1932
CUfilter_mode filterMode; 
# 1933
unsigned flags; 
# 1934
unsigned maxAnisotropy; 
# 1935
CUfilter_mode mipmapFilterMode; 
# 1936
float mipmapLevelBias; 
# 1937
float minMipmapLevelClamp; 
# 1938
float maxMipmapLevelClamp; 
# 1939
float borderColor[4]; 
# 1940
int reserved[12]; 
# 1941
} CUDA_TEXTURE_DESC; 
# 1983
typedef 
# 1946
enum CUresourceViewFormat_enum { 
# 1948
CU_RES_VIEW_FORMAT_NONE, 
# 1949
CU_RES_VIEW_FORMAT_UINT_1X8, 
# 1950
CU_RES_VIEW_FORMAT_UINT_2X8, 
# 1951
CU_RES_VIEW_FORMAT_UINT_4X8, 
# 1952
CU_RES_VIEW_FORMAT_SINT_1X8, 
# 1953
CU_RES_VIEW_FORMAT_SINT_2X8, 
# 1954
CU_RES_VIEW_FORMAT_SINT_4X8, 
# 1955
CU_RES_VIEW_FORMAT_UINT_1X16, 
# 1956
CU_RES_VIEW_FORMAT_UINT_2X16, 
# 1957
CU_RES_VIEW_FORMAT_UINT_4X16, 
# 1958
CU_RES_VIEW_FORMAT_SINT_1X16, 
# 1959
CU_RES_VIEW_FORMAT_SINT_2X16, 
# 1960
CU_RES_VIEW_FORMAT_SINT_4X16, 
# 1961
CU_RES_VIEW_FORMAT_UINT_1X32, 
# 1962
CU_RES_VIEW_FORMAT_UINT_2X32, 
# 1963
CU_RES_VIEW_FORMAT_UINT_4X32, 
# 1964
CU_RES_VIEW_FORMAT_SINT_1X32, 
# 1965
CU_RES_VIEW_FORMAT_SINT_2X32, 
# 1966
CU_RES_VIEW_FORMAT_SINT_4X32, 
# 1967
CU_RES_VIEW_FORMAT_FLOAT_1X16, 
# 1968
CU_RES_VIEW_FORMAT_FLOAT_2X16, 
# 1969
CU_RES_VIEW_FORMAT_FLOAT_4X16, 
# 1970
CU_RES_VIEW_FORMAT_FLOAT_1X32, 
# 1971
CU_RES_VIEW_FORMAT_FLOAT_2X32, 
# 1972
CU_RES_VIEW_FORMAT_FLOAT_4X32, 
# 1973
CU_RES_VIEW_FORMAT_UNSIGNED_BC1, 
# 1974
CU_RES_VIEW_FORMAT_UNSIGNED_BC2, 
# 1975
CU_RES_VIEW_FORMAT_UNSIGNED_BC3, 
# 1976
CU_RES_VIEW_FORMAT_UNSIGNED_BC4, 
# 1977
CU_RES_VIEW_FORMAT_SIGNED_BC4, 
# 1978
CU_RES_VIEW_FORMAT_UNSIGNED_BC5, 
# 1979
CU_RES_VIEW_FORMAT_SIGNED_BC5, 
# 1980
CU_RES_VIEW_FORMAT_UNSIGNED_BC6H, 
# 1981
CU_RES_VIEW_FORMAT_SIGNED_BC6H, 
# 1982
CU_RES_VIEW_FORMAT_UNSIGNED_BC7
# 1983
} CUresourceViewFormat; 
# 1999
typedef 
# 1988
struct CUDA_RESOURCE_VIEW_DESC_st { 
# 1990
CUresourceViewFormat format; 
# 1991
size_t width; 
# 1992
size_t height; 
# 1993
size_t depth; 
# 1994
unsigned firstMipmapLevel; 
# 1995
unsigned lastMipmapLevel; 
# 1996
unsigned firstLayer; 
# 1997
unsigned lastLayer; 
# 1998
unsigned reserved[16]; 
# 1999
} CUDA_RESOURCE_VIEW_DESC; 
# 2007
typedef 
# 2004
struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st { 
# 2005
unsigned long long p2pToken; 
# 2006
unsigned vaSpaceToken; 
# 2007
} CUDA_POINTER_ATTRIBUTE_P2P_TOKENS; 
# 2027 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef 
# 2016
struct CUDA_LAUNCH_PARAMS_st { 
# 2017
CUfunction function; 
# 2018
unsigned gridDimX; 
# 2019
unsigned gridDimY; 
# 2020
unsigned gridDimZ; 
# 2021
unsigned blockDimX; 
# 2022
unsigned blockDimY; 
# 2023
unsigned blockDimZ; 
# 2024
unsigned sharedMemBytes; 
# 2025
CUstream hStream; 
# 2026
void **kernelParams; 
# 2027
} CUDA_LAUNCH_PARAMS; 
# 2057 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef 
# 2036
enum CUexternalMemoryHandleType_enum { 
# 2040
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1, 
# 2044
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32, 
# 2048
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT, 
# 2052
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP, 
# 2056
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
# 2057
} CUexternalMemoryHandleType; 
# 2112 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef 
# 2067 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st { 
# 2071
CUexternalMemoryHandleType type; 
# 2072
union { 
# 2078
int fd; 
# 2091 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
struct { 
# 2095
void *handle; 
# 2100
const void *name; 
# 2101
} win32; 
# 2102
} handle; 
# 2106
unsigned long long size; 
# 2110
unsigned flags; 
# 2111
unsigned reserved[16]; 
# 2112
} CUDA_EXTERNAL_MEMORY_HANDLE_DESC; 
# 2131
typedef 
# 2117
struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st { 
# 2121
unsigned long long offset; 
# 2125
unsigned long long size; 
# 2129
unsigned flags; 
# 2130
unsigned reserved[16]; 
# 2131
} CUDA_EXTERNAL_MEMORY_BUFFER_DESC; 
# 2151
typedef 
# 2136
struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st { 
# 2141
unsigned long long offset; 
# 2145
CUDA_ARRAY3D_DESCRIPTOR arrayDesc; 
# 2149
unsigned numLevels; 
# 2150
unsigned reserved[16]; 
# 2151
} CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC; 
# 2173
typedef 
# 2156
enum CUexternalSemaphoreHandleType_enum { 
# 2160
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1, 
# 2164
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32, 
# 2168
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT, 
# 2172
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
# 2173
} CUexternalSemaphoreHandleType; 
# 2218 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
typedef 
# 2178 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st { 
# 2182
CUexternalSemaphoreHandleType type; 
# 2183
union { 
# 2189
int fd; 
# 2201 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
struct { 
# 2205
void *handle; 
# 2210
const void *name; 
# 2211
} win32; 
# 2212
} handle; 
# 2216
unsigned flags; 
# 2217
unsigned reserved[16]; 
# 2218
} CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC; 
# 2241
typedef 
# 2223
struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st { 
# 2224
struct { 
# 2228
struct { 
# 2232
unsigned long long value; 
# 2233
} fence; 
# 2234
unsigned reserved[16]; 
# 2235
} params; 
# 2239
unsigned flags; 
# 2240
unsigned reserved[16]; 
# 2241
} CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS; 
# 2264
typedef 
# 2246
struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st { 
# 2247
struct { 
# 2251
struct { 
# 2255
unsigned long long value; 
# 2256
} fence; 
# 2257
unsigned reserved[16]; 
# 2258
} params; 
# 2262
unsigned flags; 
# 2263
unsigned reserved[16]; 
# 2264
} CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS; 
# 2434 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGetErrorString(CUresult error, const char ** pStr); 
# 2455 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGetErrorName(CUresult error, const char ** pStr); 
# 2489 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuInit(unsigned Flags); 
# 2527 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDriverGetVersion(int * driverVersion); 
# 2569 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDeviceGet(CUdevice * device, int ordinal); 
# 2597 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDeviceGetCount(int * count); 
# 2628 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDeviceGetName(char * name, int len, CUdevice dev); 
# 2657 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDeviceGetUuid(CUuuid * uuid, CUdevice dev); 
# 2717 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDeviceTotalMem_v2(size_t * bytes, CUdevice dev); 
# 2922 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDeviceGetAttribute(int * pi, CUdevice_attribute attrib, CUdevice dev); 
# 3000 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
__attribute((deprecated)) CUresult cuDeviceGetProperties(CUdevprop * prop, CUdevice dev); 
# 3034 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
__attribute((deprecated)) CUresult cuDeviceComputeCapability(int * major, int * minor, CUdevice dev); 
# 3102 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDevicePrimaryCtxRetain(CUcontext * pctx, CUdevice dev); 
# 3136 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDevicePrimaryCtxRelease(CUdevice dev); 
# 3201 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned flags); 
# 3227 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned * flags, int * active); 
# 3265 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDevicePrimaryCtxReset(CUdevice dev); 
# 3377 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxCreate_v2(CUcontext * pctx, unsigned flags, CUdevice dev); 
# 3417 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxDestroy_v2(CUcontext ctx); 
# 3453 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxPushCurrent_v2(CUcontext ctx); 
# 3487 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxPopCurrent_v2(CUcontext * pctx); 
# 3517 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxSetCurrent(CUcontext ctx); 
# 3540 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxGetCurrent(CUcontext * pctx); 
# 3571 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxGetDevice(CUdevice * device); 
# 3600 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxGetFlags(unsigned * flags); 
# 3631 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxSynchronize(); 
# 3723 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxSetLimit(CUlimit limit, size_t value); 
# 3764 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxGetLimit(size_t * pvalue, CUlimit limit); 
# 3808 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxGetCacheConfig(CUfunc_cache * pconfig); 
# 3859 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxSetCacheConfig(CUfunc_cache config); 
# 3902 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxGetSharedMemConfig(CUsharedconfig * pConfig); 
# 3955 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxSetSharedMemConfig(CUsharedconfig config); 
# 3994 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned * version); 
# 4034 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
# 4089 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
__attribute((deprecated)) CUresult cuCtxAttach(CUcontext * pctx, unsigned flags); 
# 4125 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
__attribute((deprecated)) CUresult cuCtxDetach(CUcontext ctx); 
# 4180 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuModuleLoad(CUmodule * module, const char * fname); 
# 4217 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuModuleLoadData(CUmodule * module, const void * image); 
# 4260 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuModuleLoadDataEx(CUmodule * module, const void * image, unsigned numOptions, CUjit_option * options, void ** optionValues); 
# 4302 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuModuleLoadFatBinary(CUmodule * module, const void * fatCubin); 
# 4327 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuModuleUnload(CUmodule hmod); 
# 4357 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuModuleGetFunction(CUfunction * hfunc, CUmodule hmod, const char * name); 
# 4393 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuModuleGetGlobal_v2(CUdeviceptr * dptr, size_t * bytes, CUmodule hmod, const char * name); 
# 4428 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuModuleGetTexRef(CUtexref * pTexRef, CUmodule hmod, const char * name); 
# 4460 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuModuleGetSurfRef(CUsurfref * pSurfRef, CUmodule hmod, const char * name); 
# 4503 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuLinkCreate_v2(unsigned numOptions, CUjit_option * options, void ** optionValues, CUlinkState * stateOut); 
# 4540 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void * data, size_t size, const char * name, unsigned numOptions, CUjit_option * options, void ** optionValues); 
# 4579 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char * path, unsigned numOptions, CUjit_option * options, void ** optionValues); 
# 4606 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuLinkComplete(CUlinkState state, void ** cubinOut, size_t * sizeOut); 
# 4620 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuLinkDestroy(CUlinkState state); 
# 4669 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemGetInfo_v2(size_t * free, size_t * total); 
# 4703 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemAlloc_v2(CUdeviceptr * dptr, size_t bytesize); 
# 4765 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemAllocPitch_v2(CUdeviceptr * dptr, size_t * pPitch, size_t WidthInBytes, size_t Height, unsigned ElementSizeBytes); 
# 4795 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemFree_v2(CUdeviceptr dptr); 
# 4829 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemGetAddressRange_v2(CUdeviceptr * pbase, size_t * psize, CUdeviceptr dptr); 
# 4876 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemAllocHost_v2(void ** pp, size_t bytesize); 
# 4907 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemFreeHost(void * p); 
# 4989 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemHostAlloc(void ** pp, size_t bytesize, unsigned Flags); 
# 5043 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr * pdptr, void * p, unsigned Flags); 
# 5071 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemHostGetFlags(unsigned * pFlags, void * p); 
# 5183 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemAllocManaged(CUdeviceptr * dptr, size_t bytesize, unsigned flags); 
# 5216 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDeviceGetByPCIBusId(CUdevice * dev, const char * pciBusId); 
# 5248 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDeviceGetPCIBusId(char * pciBusId, int len, CUdevice dev); 
# 5293 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuIpcGetEventHandle(CUipcEventHandle * pHandle, CUevent event); 
# 5333 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuIpcOpenEventHandle(CUevent * phEvent, CUipcEventHandle handle); 
# 5373 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuIpcGetMemHandle(CUipcMemHandle * pHandle, CUdeviceptr dptr); 
# 5430 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuIpcOpenMemHandle(CUdeviceptr * pdptr, CUipcMemHandle handle, unsigned Flags); 
# 5463 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuIpcCloseMemHandle(CUdeviceptr dptr); 
# 5549 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemHostRegister_v2(void * p, size_t bytesize, unsigned Flags); 
# 5575 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemHostUnregister(void * p); 
# 5614 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount); 
# 5644 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount); 
# 5682 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void * srcHost, size_t ByteCount); 
# 5717 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyDtoH_v2(void * dstHost, CUdeviceptr srcDevice, size_t ByteCount); 
# 5753 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount); 
# 5789 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount); 
# 5827 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount); 
# 5863 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void * srcHost, size_t ByteCount); 
# 5899 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyAtoH_v2(void * dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount); 
# 5939 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount); 
# 6103 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D * pCopy); 
# 6265 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D * pCopy); 
# 6434 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D * pCopy); 
# 6460 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER * pCopy); 
# 6504 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream); 
# 6537 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream); 
# 6579 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void * srcHost, size_t ByteCount, CUstream hStream); 
# 6619 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyDtoHAsync_v2(void * dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream); 
# 6660 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream); 
# 6701 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void * srcHost, size_t ByteCount, CUstream hStream); 
# 6742 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpyAtoHAsync_v2(void * dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream); 
# 6911 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D * pCopy, CUstream hStream); 
# 7085 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D * pCopy, CUstream hStream); 
# 7113 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER * pCopy, CUstream hStream); 
# 7150 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N); 
# 7185 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N); 
# 7220 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned ui, size_t N); 
# 7260 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height); 
# 7301 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height); 
# 7342 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned ui, size_t Width, size_t Height); 
# 7379 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream); 
# 7416 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream); 
# 7452 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned ui, size_t N, CUstream hStream); 
# 7494 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream); 
# 7537 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream); 
# 7580 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned ui, size_t Width, size_t Height, CUstream hStream); 
# 7684 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuArrayCreate_v2(CUarray * pHandle, const CUDA_ARRAY_DESCRIPTOR * pAllocateArray); 
# 7718 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR * pArrayDescriptor, CUarray hArray); 
# 7751 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuArrayDestroy(CUarray hArray); 
# 7932 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuArray3DCreate_v2(CUarray * pHandle, const CUDA_ARRAY3D_DESCRIPTOR * pAllocateArray); 
# 7970 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR * pArrayDescriptor, CUarray hArray); 
# 8115 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMipmappedArrayCreate(CUmipmappedArray * pHandle, const CUDA_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc, unsigned numMipmapLevels); 
# 8145 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMipmappedArrayGetLevel(CUarray * pLevelArray, CUmipmappedArray hMipmappedArray, unsigned level); 
# 8170 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray); 
# 8422 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuPointerGetAttribute(void * data, CUpointer_attribute attribute, CUdeviceptr ptr); 
# 8492 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream); 
# 8606 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device); 
# 8664 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemRangeGetAttribute(void * data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count); 
# 8704 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuMemRangeGetAttributes(void ** data, size_t * dataSizes, CUmem_range_attribute * attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count); 
# 8748 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuPointerSetAttribute(const void * value, CUpointer_attribute attribute, CUdeviceptr ptr); 
# 8793 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuPointerGetAttributes(unsigned numAttributes, CUpointer_attribute * attributes, void ** data, CUdeviceptr ptr); 
# 8843 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamCreate(CUstream * phStream, unsigned Flags); 
# 8892 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamCreateWithPriority(CUstream * phStream, unsigned flags, int priority); 
# 8923 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamGetPriority(CUstream hStream, int * priority); 
# 8951 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamGetFlags(CUstream hStream, unsigned * flags); 
# 8997 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamGetCtx(CUstream hStream, CUcontext * pctx); 
# 9030 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned Flags); 
# 9105 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void * userData, unsigned flags); 
# 9145 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode); 
# 9201 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode * mode); 
# 9234 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamEndCapture(CUstream hStream, CUgraph * phGraph); 
# 9274 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus * captureStatus); 
# 9302 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus * captureStatus, cuuint64_t * id); 
# 9394 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned flags); 
# 9426 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamQuery(CUstream hStream); 
# 9455 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamSynchronize(CUstream hStream); 
# 9486 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamDestroy_v2(CUstream hStream); 
# 9543 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuEventCreate(CUevent * phEvent, unsigned Flags); 
# 9584 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuEventRecord(CUevent hEvent, CUstream hStream); 
# 9616 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuEventQuery(CUevent hEvent); 
# 9647 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuEventSynchronize(CUevent hEvent); 
# 9677 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuEventDestroy_v2(CUevent hEvent); 
# 9722 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuEventElapsedTime(float * pMilliseconds, CUevent hStart, CUevent hEnd); 
# 9862 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuImportExternalMemory(CUexternalMemory * extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC * memHandleDesc); 
# 9915 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr * devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC * bufferDesc); 
# 9964 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray * mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC * mipmapDesc); 
# 9986 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDestroyExternalMemory(CUexternalMemory extMem); 
# 10083 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuImportExternalSemaphore(CUexternalSemaphore * extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC * semHandleDesc); 
# 10121 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore * extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS * paramsArray, unsigned numExtSems, CUstream stream); 
# 10163 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore * extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS * paramsArray, unsigned numExtSems, CUstream stream); 
# 10184 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem); 
# 10271 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned flags); 
# 10306 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned flags); 
# 10341 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned flags); 
# 10375 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned flags); 
# 10410 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuStreamBatchMemOp(CUstream stream, unsigned count, CUstreamBatchMemOpParams * paramArray, unsigned flags); 
# 10484 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuFuncGetAttribute(int * pi, CUfunction_attribute attrib, CUfunction hfunc); 
# 10532 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value); 
# 10577 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config); 
# 10630 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config); 
# 10745 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuLaunchKernel(CUfunction f, unsigned gridDimX, unsigned gridDimY, unsigned gridDimZ, unsigned blockDimX, unsigned blockDimY, unsigned blockDimZ, unsigned sharedMemBytes, CUstream hStream, void ** kernelParams, void ** extra); 
# 10834 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned gridDimX, unsigned gridDimY, unsigned gridDimZ, unsigned blockDimX, unsigned blockDimY, unsigned blockDimZ, unsigned sharedMemBytes, CUstream hStream, void ** kernelParams); 
# 10978 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS * launchParamsList, unsigned numDevices, unsigned flags); 
# 11047 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void * userData); 
# 11099 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
__attribute((deprecated)) CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z); 
# 11133 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
__attribute((deprecated)) CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned bytes); 
# 11165 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
__attribute((deprecated)) CUresult cuParamSetSize(CUfunction hfunc, unsigned numbytes); 
# 11198 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
__attribute((deprecated)) CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned value); 
# 11231 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
__attribute((deprecated)) CUresult cuParamSetf(CUfunction hfunc, int offset, float value); 
# 11266 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
__attribute((deprecated)) CUresult cuParamSetv(CUfunction hfunc, int offset, void * ptr, unsigned numbytes); 
# 11303 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
__attribute((deprecated)) CUresult cuLaunch(CUfunction f); 
# 11342 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
__attribute((deprecated)) CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height); 
# 11389 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
__attribute((deprecated)) CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream); 
# 11414 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
__attribute((deprecated)) CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef); 
# 11461 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphCreate(CUgraph * phGraph, unsigned flags); 
# 11560 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphAddKernelNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS * nodeParams); 
# 11592 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS * nodeParams); 
# 11615 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams); 
# 11663 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphAddMemcpyNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, const CUDA_MEMCPY3D * copyParams, CUcontext ctx); 
# 11686 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D * nodeParams); 
# 11709 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D * nodeParams); 
# 11751 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphAddMemsetNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS * memsetParams, CUcontext ctx); 
# 11774 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS * nodeParams); 
# 11797 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS * nodeParams); 
# 11838 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphAddHostNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS * nodeParams); 
# 11861 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS * nodeParams); 
# 11884 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS * nodeParams); 
# 11922 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphAddChildGraphNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies, CUgraph childGraph); 
# 11946 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph * phGraph); 
# 11984 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphAddEmptyNode(CUgraphNode * phGraphNode, CUgraph hGraph, const CUgraphNode * dependencies, size_t numDependencies); 
# 12009 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphClone(CUgraph * phGraphClone, CUgraph originalGraph); 
# 12035 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphNodeFindInClone(CUgraphNode * phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph); 
# 12066 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType * type); 
# 12097 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode * nodes, size_t * numNodes); 
# 12128 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode * rootNodes, size_t * numRootNodes); 
# 12162 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode * from, CUgraphNode * to, size_t * numEdges); 
# 12193 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode * dependencies, size_t * numDependencies); 
# 12225 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode * dependentNodes, size_t * numDependentNodes); 
# 12254 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode * from, const CUgraphNode * to, size_t numDependencies); 
# 12283 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode * from, const CUgraphNode * to, size_t numDependencies); 
# 12307 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphDestroyNode(CUgraphNode hNode); 
# 12343 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphInstantiate(CUgraphExec * phGraphExec, CUgraph hGraph, CUgraphNode * phErrorNode, char * logBuffer, size_t bufferSize); 
# 12377 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams); 
# 12404 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream); 
# 12428 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphExecDestroy(CUgraphExec hGraphExec); 
# 12448 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphDestroy(CUgraph hGraph); 
# 12488 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize); 
# 12530 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
# 12582 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuOccupancyMaxPotentialBlockSize(int * minGridSize, int * blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit); 
# 12628 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int * minGridSize, int * blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned flags); 
# 12674 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned Flags); 
# 12704 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned Flags); 
# 12751 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefSetAddress_v2(size_t * ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes); 
# 12806 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR * desc, CUdeviceptr dptr, size_t Pitch); 
# 12842 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents); 
# 12888 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am); 
# 12924 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm); 
# 12960 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm); 
# 12989 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias); 
# 13020 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp); 
# 13050 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned maxAniso); 
# 13086 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float * pBorderColor); 
# 13127 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned Flags); 
# 13155 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefGetAddress_v2(CUdeviceptr * pdptr, CUtexref hTexRef); 
# 13183 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefGetArray(CUarray * phArray, CUtexref hTexRef); 
# 13210 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefGetMipmappedArray(CUmipmappedArray * phMipmappedArray, CUtexref hTexRef); 
# 13238 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefGetAddressMode(CUaddress_mode * pam, CUtexref hTexRef, int dim); 
# 13264 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefGetFilterMode(CUfilter_mode * pfm, CUtexref hTexRef); 
# 13292 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefGetFormat(CUarray_format * pFormat, int * pNumChannels, CUtexref hTexRef); 
# 13318 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode * pfm, CUtexref hTexRef); 
# 13344 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefGetMipmapLevelBias(float * pbias, CUtexref hTexRef); 
# 13371 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefGetMipmapLevelClamp(float * pminMipmapLevelClamp, float * pmaxMipmapLevelClamp, CUtexref hTexRef); 
# 13397 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefGetMaxAnisotropy(int * pmaxAniso, CUtexref hTexRef); 
# 13426 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefGetBorderColor(float * pBorderColor, CUtexref hTexRef); 
# 13451 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefGetFlags(unsigned * pFlags, CUtexref hTexRef); 
# 13476 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefCreate(CUtexref * pTexRef); 
# 13496 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexRefDestroy(CUtexref hTexRef); 
# 13540 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned Flags); 
# 13563 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuSurfRefGetArray(CUarray * phArray, CUsurfref hSurfRef); 
# 13787 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexObjectCreate(CUtexObject * pTexObject, const CUDA_RESOURCE_DESC * pResDesc, const CUDA_TEXTURE_DESC * pTexDesc, const CUDA_RESOURCE_VIEW_DESC * pResViewDesc); 
# 13807 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexObjectDestroy(CUtexObject texObject); 
# 13828 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC * pResDesc, CUtexObject texObject); 
# 13849 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC * pTexDesc, CUtexObject texObject); 
# 13871 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC * pResViewDesc, CUtexObject texObject); 
# 13914 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuSurfObjectCreate(CUsurfObject * pSurfObject, const CUDA_RESOURCE_DESC * pResDesc); 
# 13934 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuSurfObjectDestroy(CUsurfObject surfObject); 
# 13955 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC * pResDesc, CUsurfObject surfObject); 
# 14000 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDeviceCanAccessPeer(int * canAccessPeer, CUdevice dev, CUdevice peerDev); 
# 14051 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned Flags); 
# 14078 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuCtxDisablePeerAccess(CUcontext peerContext); 
# 14122 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuDeviceGetP2PAttribute(int * value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice); 
# 14168 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource); 
# 14208 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphicsSubResourceGetMappedArray(CUarray * pArray, CUgraphicsResource resource, unsigned arrayIndex, unsigned mipLevel); 
# 14241 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray * pMipmappedArray, CUgraphicsResource resource); 
# 14278 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr * pDevPtr, size_t * pSize, CUgraphicsResource resource); 
# 14320 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned flags); 
# 14360 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphicsMapResources(unsigned count, CUgraphicsResource * resources, CUstream hStream); 
# 14397 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
CUresult cuGraphicsUnmapResources(unsigned count, CUgraphicsResource * resources, CUstream hStream); 
# 14401
CUresult cuGetExportTable(const void ** ppExportTable, const CUuuid * pExportTableId); 
# 14746 "/nvme/h/buildsets/eb_cyclone/software/CUDA/10.1.243/bin/../targets/x86_64-linux/include/cuda.h"
}
# 876 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) inline void prefetch_gl1(const void *p) {int volatile ___ = 1;(void)p;
# 880
::exit(___);}
#if 0
# 876
{ 
# 878
__asm__("prefetch.global.L1 [%0];" : : "l" (p)); 
# 880
} 
#endif
# 881 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) inline void prefetch_ll1(const void *p) {int volatile ___ = 1;(void)p;
# 885
::exit(___);}
#if 0
# 881
{ 
# 883
__asm__("prefetch.local.L1 [%0];" : : "l" (p)); 
# 885
} 
#endif
# 888 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) inline void prefetch_gl2(const void *p) {int volatile ___ = 1;(void)p;
# 892
::exit(___);}
#if 0
# 888
{ 
# 890
__asm__("prefetch.global.L2 [%0];" : : "l" (p)); 
# 892
} 
#endif
# 893 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) inline void prefetch_ll2(const void *p) {int volatile ___ = 1;(void)p;
# 897
::exit(___);}
#if 0
# 893
{ 
# 895
__asm__("prefetch.local.L2 [%0];" : : "l" (p)); 
# 897
} 
#endif
# 901 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) void update_rconst(const double *__restrict__ var, const double *__restrict__ khet_st, const double *__restrict__ khet_tr, const double *__restrict__ jx, double *__restrict__ rconst, const double *__restrict__ temp_gpu, const double *__restrict__ press_gpu, const double *__restrict__ cair_gpu, const int VL_GLO); 
# 910
double machine_eps_flt() 
# 911
{ 
# 912
double machEps = ((1.0F)); 
# 914
do 
# 915
{ 
# 916
machEps /= ((2.0F)); 
# 919
} 
# 920
while (((double)((1.0) + (machEps / (2.0)))) != (1.0)); 
# 922
return machEps; 
# 923
} 
# 926
__attribute__((unused)) double machine_eps_flt_cuda() 
# 927
{int volatile ___ = 1;
# 939
::exit(___);}
#if 0
# 927
{ 
# 932
typedef 
# 929
union { 
# 930
long i64; 
# 931
double f64; 
# 932
} flt_64; 
# 934
flt_64 s; 
# 936
(s.f64) = (1.0); 
# 937
(s.i64)++; 
# 938
return (s.f64) - (1.0); 
# 939
} 
#endif
# 941 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) static double alpha_AN(const int n, const int ro2type, const double temp, const double cair) {int volatile ___ = 1;(void)n;(void)ro2type;(void)temp;(void)cair;
# 957
::exit(___);}
#if 0
# 941
{ 
# 942
double alpha = (2.000000000000000097e-22), beta = (1.0), Yinf_298K = (0.4299999999999999933), F = (0.4099999999999999756), m0 = (0.0), minf = (8.0); 
# 943
double Y0_298K, Y0_298K_tp, Yinf_298K_t, zeta, k_ratio, alpha_a; 
# 949
double m = (1.0); 
# 950
Y0_298K = (alpha * exp(beta * n)); 
# 951
Y0_298K_tp = ((Y0_298K * cair) * pow(temp / (298.0), -m0)); 
# 952
Yinf_298K_t = (Yinf_298K * pow(temp / (298.0), -minf)); 
# 953
zeta = ((1) / ((1) + pow(log10(Y0_298K_tp / Yinf_298K_t), 2))); 
# 954
k_ratio = ((Y0_298K_tp / ((1) + (Y0_298K_tp / Yinf_298K_t))) * pow(F, zeta)); 
# 955
alpha_a = ((k_ratio / ((1) + k_ratio)) * m); 
# 956
return alpha_a; 
# 957
} 
#endif
# 958 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) static double alpha_AN(const int n, const int ro2type, const int bcarb, const int gcarb, const int abic, const double temp, const double cair) {int volatile ___ = 1;(void)n;(void)ro2type;(void)bcarb;(void)gcarb;(void)abic;(void)temp;(void)cair;
# 976
::exit(___);}
#if 0
# 958
{ 
# 959
double alpha = (2.000000000000000097e-22), beta = (1.0), Yinf_298K = (0.4299999999999999933), F = (0.4099999999999999756), m0 = (0.0), minf = (8.0); 
# 960
double Y0_298K, Y0_298K_tp, Yinf_298K_t, zeta, k_ratio, alpha_a; 
# 961
double bcf = (1.0), gcf = (1.0), abf = (1.0); 
# 962
double m = (1.0); 
# 964
if (bcarb == 1) { bcf = (0.1900000000000000022); }  
# 965
if (gcarb == 1) { gcf = (0.4400000000000000022); }  
# 966
if (abic == 1) { abf = (0.2399999999999999911); }  
# 969
Y0_298K = (alpha * exp(beta * n)); 
# 970
Y0_298K_tp = ((Y0_298K * cair) * pow(temp / (298.0), -m0)); 
# 971
Yinf_298K_t = (Yinf_298K * pow(temp / (298.0), -minf)); 
# 972
zeta = ((1) / ((1) + pow(log10(Y0_298K_tp / Yinf_298K_t), 2))); 
# 973
k_ratio = ((Y0_298K_tp / ((1) + (Y0_298K_tp / Yinf_298K_t))) * pow(F, zeta)); 
# 974
alpha_a = (((((k_ratio / ((1) + k_ratio)) * m) * bcf) * gcf) * abf); 
# 975
return alpha_a; 
# 976
} 
#endif
# 977 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) static double k_RO2_HO2(const double temp, const int nC) {int volatile ___ = 1;(void)temp;(void)nC;
# 979
::exit(___);}
#if 0
# 977
{ 
# 978
return ((2.910000000000000222e-13) * exp((1300.0) / temp)) * ((1.0) - exp((-(0.2449999999999999956)) * nC)); 
# 979
} 
#endif
# 980 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) double ros_ErrorNorm(double *__restrict__ var, double *__restrict__ varNew, double *__restrict__ varErr, const double *__restrict__ 
# 981
absTol, const double *__restrict__ relTol, const int 
# 982
vectorTol) 
# 983
{int volatile ___ = 1;(void)var;(void)varNew;(void)varErr;(void)absTol;(void)relTol;(void)vectorTol;
# 1026
::exit(___);}
#if 0
# 983
{ 
# 984
double err, scale, varMax; 
# 987
err = (0.0); 
# 989
if (vectorTol) { 
# 990
for (int i = 0; i < (73 - 16); i += 16) { 
# 991
prefetch_ll1(&(varErr[i])); 
# 992
prefetch_ll1(&(absTol[i])); 
# 993
prefetch_ll1(&(relTol[i])); 
# 994
prefetch_ll1(&(var[i])); 
# 995
prefetch_ll1(&(varNew[i])); 
# 996
}  
# 998
for (int i = 0; i < 73; i++) 
# 999
{ 
# 1000
varMax = fmax(fabs(var[i]), fabs(varNew[i])); 
# 1001
scale = ((absTol[i]) + ((relTol[i]) * varMax)); 
# 1003
err += pow(((double)(varErr[i])) / scale, (2.0)); 
# 1004
}  
# 1005
err = sqrt(((double)err) / (73)); 
# 1006
} else { 
# 1007
for (int i = 0; i < (73 - 16); i += 16) { 
# 1008
prefetch_ll1(&(varErr[i])); 
# 1009
prefetch_ll1(&(var[i])); 
# 1010
prefetch_ll1(&(varNew[i])); 
# 1011
}  
# 1013
for (int i = 0; i < 73; i++) 
# 1014
{ 
# 1015
varMax = fmax(fabs(var[i]), fabs(varNew[i])); 
# 1017
scale = ((absTol[0]) + ((relTol[0]) * varMax)); 
# 1018
err += pow(((double)(varErr[i])) / scale, (2.0)); 
# 1019
}  
# 1020
err = sqrt(((double)err) / (73)); 
# 1021
}  
# 1023
return err; 
# 1026
} 
#endif
# 1028 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) void kppSolve(const double *__restrict__ Ghimj, double *__restrict__ K, const int 
# 1029
istage, const int ros_S) {int volatile ___ = 1;(void)Ghimj;(void)K;(void)istage;(void)ros_S;
# 1129
::exit(___);}
#if 0
# 1029
{ 
# 1030
int index = ((__device_builtin_variable_blockIdx.x) * (__device_builtin_variable_blockDim.x)) + (__device_builtin_variable_threadIdx.x); 
# 1032
K = (&(K[istage * 73])); 
# 1034
(K[9]) = ((K[9]) - ((Ghimj[32]) * (K[0]))); 
# 1035
(K[10]) = ((K[10]) - ((Ghimj[34]) * (K[0]))); 
# 1036
(K[17]) = ((K[17]) - ((Ghimj[54]) * (K[0]))); 
# 1037
(K[47]) = (((((((((((K[47]) - ((Ghimj[155]) * (K[29]))) - ((Ghimj[156]) * (K[32]))) - ((Ghimj[157]) * (K[33]))) - ((Ghimj[158]) * (K[34]))) - ((Ghimj[159]) * (K[35]))) - ((Ghimj[160]) * (K[36]))) - ((Ghimj[161]) * (K[37]))) - ((Ghimj[162]) * (K[39]))) - ((Ghimj[163]) * (K[40]))) - ((Ghimj[164]) * (K[41]))); 
# 1038
(K[55]) = ((K[55]) - ((Ghimj[200]) * (K[48]))); 
# 1039
(K[56]) = ((K[56]) - ((Ghimj[206]) * (K[54]))); 
# 1040
(K[57]) = (((((K[57]) - ((Ghimj[214]) * (K[45]))) - ((Ghimj[215]) * (K[46]))) - ((Ghimj[216]) * (K[49]))) - ((Ghimj[217]) * (K[51]))); 
# 1041
(K[58]) = ((((((((K[58]) - ((Ghimj[225]) * (K[42]))) - ((Ghimj[226]) * (K[43]))) - ((Ghimj[227]) * (K[45]))) - ((Ghimj[228]) * (K[46]))) - ((Ghimj[229]) * (K[49]))) - ((Ghimj[230]) * (K[51]))) - ((Ghimj[231]) * (K[52]))); 
# 1042
(K[59]) = ((K[59]) - ((Ghimj[239]) * (K[56]))); 
# 1043
(K[60]) = ((K[60]) - ((Ghimj[248]) * (K[53]))); 
# 1044
(K[61]) = ((K[61]) - ((Ghimj[257]) * (K[55]))); 
# 1045
(K[62]) = (((((K[62]) - ((Ghimj[263]) * (K[50]))) - ((Ghimj[264]) * (K[51]))) - ((Ghimj[265]) * (K[53]))) - ((Ghimj[266]) * (K[61]))); 
# 1046
(K[63]) = (((K[63]) - ((Ghimj[276]) * (K[55]))) - ((Ghimj[277]) * (K[61]))); 
# 1047
(K[64]) = ((((K[64]) - ((Ghimj[283]) * (K[53]))) - ((Ghimj[284]) * (K[61]))) - ((Ghimj[285]) * (K[63]))); 
# 1048
(K[65]) = (((((((((((((((((((((((K[65]) - ((Ghimj[292]) * (K[32]))) - ((Ghimj[293]) * (K[33]))) - ((Ghimj[294]) * (K[34]))) - ((Ghimj[295]) * (K[35]))) - ((Ghimj[296]) * (K[36]))) - ((Ghimj[297]) * (K[37]))) - ((Ghimj[298]) * (K[39]))) - ((Ghimj[299]) * (K[40]))) - ((Ghimj[300]) * (K[41]))) - ((Ghimj[301]) * (K[44]))) - ((Ghimj[302]) * (K[47]))) - ((Ghimj[303]) * (K[50]))) - ((Ghimj[304]) * (K[51]))) - ((Ghimj[305]) * (K[52]))) - ((Ghimj[306]) * (K[57]))) - ((Ghimj[307]) * (K[58]))) - ((Ghimj[308]) * (K[59]))) - ((Ghimj[309]) * (K[60]))) - ((Ghimj[310]) * (K[61]))) - ((Ghimj[311]) * (K[62]))) - ((Ghimj[312]) * (K[63]))) - ((Ghimj[313]) * (K[64]))); 
# 1049
(K[66]) = (((((((((((((K[66]) - ((Ghimj[322]) * (K[28]))) - ((Ghimj[323]) * (K[41]))) - ((Ghimj[324]) * (K[47]))) - ((Ghimj[325]) * (K[52]))) - ((Ghimj[326]) * (K[57]))) - ((Ghimj[327]) * (K[58]))) - ((Ghimj[328]) * (K[60]))) - ((Ghimj[329]) * (K[61]))) - ((Ghimj[330]) * (K[62]))) - ((Ghimj[331]) * (K[63]))) - ((Ghimj[332]) * (K[64]))) - ((Ghimj[333]) * (K[65]))); 
# 1050
(K[67]) = ((((((((((((((((((((K[67]) - ((Ghimj[341]) * (K[39]))) - ((Ghimj[342]) * (K[42]))) - ((Ghimj[343]) * (K[43]))) - ((Ghimj[344]) * (K[45]))) - ((Ghimj[345]) * (K[46]))) - ((Ghimj[346]) * (K[48]))) - ((Ghimj[347]) * (K[49]))) - ((Ghimj[348]) * (K[51]))) - ((Ghimj[349]) * (K[52]))) - ((Ghimj[350]) * (K[54]))) - ((Ghimj[351]) * (K[56]))) - ((Ghimj[352]) * (K[58]))) - ((Ghimj[353]) * (K[59]))) - ((Ghimj[354]) * (K[61]))) - ((Ghimj[355]) * (K[62]))) - ((Ghimj[356]) * (K[63]))) - ((Ghimj[357]) * (K[64]))) - ((Ghimj[358]) * (K[65]))) - ((Ghimj[359]) * (K[66]))); 
# 1051
(K[68]) = (((((((((((((((((((((((K[68]) - ((Ghimj[366]) * (K[32]))) - ((Ghimj[367]) * (K[33]))) - ((Ghimj[368]) * (K[34]))) - ((Ghimj[369]) * (K[35]))) - ((Ghimj[370]) * (K[36]))) - ((Ghimj[371]) * (K[37]))) - ((Ghimj[372]) * (K[45]))) - ((Ghimj[373]) * (K[46]))) - ((Ghimj[374]) * (K[48]))) - ((Ghimj[375]) * (K[49]))) - ((Ghimj[376]) * (K[51]))) - ((Ghimj[377]) * (K[54]))) - ((Ghimj[378]) * (K[56]))) - ((Ghimj[379]) * (K[57]))) - ((Ghimj[380]) * (K[59]))) - ((Ghimj[381]) * (K[61]))) - ((Ghimj[382]) * (K[62]))) - ((Ghimj[383]) * (K[63]))) - ((Ghimj[384]) * (K[64]))) - ((Ghimj[385]) * (K[65]))) - ((Ghimj[386]) * (K[66]))) - ((Ghimj[387]) * (K[67]))); 
# 1052
(K[69]) = (((((((((K[69]) - ((Ghimj[393]) * (K[30]))) - ((Ghimj[394]) * (K[48]))) - ((Ghimj[395]) * (K[55]))) - ((Ghimj[396]) * (K[61]))) - ((Ghimj[397]) * (K[63]))) - ((Ghimj[398]) * (K[66]))) - ((Ghimj[399]) * (K[67]))) - ((Ghimj[400]) * (K[68]))); 
# 1053
(K[70]) = (((((((((((((((K[70]) - ((Ghimj[405]) * (K[42]))) - ((Ghimj[406]) * (K[43]))) - ((Ghimj[407]) * (K[49]))) - ((Ghimj[408]) * (K[50]))) - ((Ghimj[409]) * (K[51]))) - ((Ghimj[410]) * (K[60]))) - ((Ghimj[411]) * (K[61]))) - ((Ghimj[412]) * (K[63]))) - ((Ghimj[413]) * (K[64]))) - ((Ghimj[414]) * (K[65]))) - ((Ghimj[415]) * (K[66]))) - ((Ghimj[416]) * (K[67]))) - ((Ghimj[417]) * (K[68]))) - ((Ghimj[418]) * (K[69]))); 
# 1054
(K[71]) = ((((((((((((((((((K[71]) - ((Ghimj[422]) * (K[31]))) - ((Ghimj[423]) * (K[38]))) - ((Ghimj[424]) * (K[40]))) - ((Ghimj[425]) * (K[44]))) - ((Ghimj[426]) * (K[50]))) - ((Ghimj[427]) * (K[52]))) - ((Ghimj[428]) * (K[60]))) - ((Ghimj[429]) * (K[61]))) - ((Ghimj[430]) * (K[62]))) - ((Ghimj[431]) * (K[63]))) - ((Ghimj[432]) * (K[64]))) - ((Ghimj[433]) * (K[65]))) - ((Ghimj[434]) * (K[66]))) - ((Ghimj[435]) * (K[67]))) - ((Ghimj[436]) * (K[68]))) - ((Ghimj[437]) * (K[69]))) - ((Ghimj[438]) * (K[70]))); 
# 1055
(K[72]) = (((((((((((((K[72]) - ((Ghimj[441]) * (K[48]))) - ((Ghimj[442]) * (K[55]))) - ((Ghimj[443]) * (K[61]))) - ((Ghimj[444]) * (K[63]))) - ((Ghimj[445]) * (K[64]))) - ((Ghimj[446]) * (K[65]))) - ((Ghimj[447]) * (K[66]))) - ((Ghimj[448]) * (K[67]))) - ((Ghimj[449]) * (K[68]))) - ((Ghimj[450]) * (K[69]))) - ((Ghimj[451]) * (K[70]))) - ((Ghimj[452]) * (K[71]))); 
# 1056
(K[72]) = ((K[72]) / (Ghimj[453])); 
# 1057
(K[71]) = (((K[71]) - ((Ghimj[440]) * (K[72]))) / (Ghimj[439])); 
# 1058
(K[70]) = ((((K[70]) - ((Ghimj[420]) * (K[71]))) - ((Ghimj[421]) * (K[72]))) / (Ghimj[419])); 
# 1059
(K[69]) = (((((K[69]) - ((Ghimj[402]) * (K[70]))) - ((Ghimj[403]) * (K[71]))) - ((Ghimj[404]) * (K[72]))) / (Ghimj[401])); 
# 1060
(K[68]) = ((((((K[68]) - ((Ghimj[389]) * (K[69]))) - ((Ghimj[390]) * (K[70]))) - ((Ghimj[391]) * (K[71]))) - ((Ghimj[392]) * (K[72]))) / (Ghimj[388])); 
# 1061
(K[67]) = (((((((K[67]) - ((Ghimj[361]) * (K[68]))) - ((Ghimj[362]) * (K[69]))) - ((Ghimj[363]) * (K[70]))) - ((Ghimj[364]) * (K[71]))) - ((Ghimj[365]) * (K[72]))) / (Ghimj[360])); 
# 1062
(K[66]) = ((((((((K[66]) - ((Ghimj[335]) * (K[67]))) - ((Ghimj[336]) * (K[68]))) - ((Ghimj[337]) * (K[69]))) - ((Ghimj[338]) * (K[70]))) - ((Ghimj[339]) * (K[71]))) - ((Ghimj[340]) * (K[72]))) / (Ghimj[334])); 
# 1063
(K[65]) = (((((((((K[65]) - ((Ghimj[315]) * (K[66]))) - ((Ghimj[316]) * (K[67]))) - ((Ghimj[317]) * (K[68]))) - ((Ghimj[318]) * (K[69]))) - ((Ghimj[319]) * (K[70]))) - ((Ghimj[320]) * (K[71]))) - ((Ghimj[321]) * (K[72]))) / (Ghimj[314])); 
# 1064
(K[64]) = (((((((K[64]) - ((Ghimj[287]) * (K[65]))) - ((Ghimj[288]) * (K[67]))) - ((Ghimj[289]) * (K[68]))) - ((Ghimj[290]) * (K[69]))) - ((Ghimj[291]) * (K[72]))) / (Ghimj[286])); 
# 1065
(K[63]) = ((((((K[63]) - ((Ghimj[279]) * (K[67]))) - ((Ghimj[280]) * (K[68]))) - ((Ghimj[281]) * (K[69]))) - ((Ghimj[282]) * (K[72]))) / (Ghimj[278])); 
# 1066
(K[62]) = ((((((((((K[62]) - ((Ghimj[268]) * (K[63]))) - ((Ghimj[269]) * (K[64]))) - ((Ghimj[270]) * (K[65]))) - ((Ghimj[271]) * (K[67]))) - ((Ghimj[272]) * (K[68]))) - ((Ghimj[273]) * (K[69]))) - ((Ghimj[274]) * (K[70]))) - ((Ghimj[275]) * (K[72]))) / (Ghimj[267])); 
# 1067
(K[61]) = ((((((K[61]) - ((Ghimj[259]) * (K[63]))) - ((Ghimj[260]) * (K[68]))) - ((Ghimj[261]) * (K[69]))) - ((Ghimj[262]) * (K[72]))) / (Ghimj[258])); 
# 1068
(K[60]) = (((((((((K[60]) - ((Ghimj[250]) * (K[61]))) - ((Ghimj[251]) * (K[63]))) - ((Ghimj[252]) * (K[64]))) - ((Ghimj[253]) * (K[65]))) - ((Ghimj[254]) * (K[66]))) - ((Ghimj[255]) * (K[67]))) - ((Ghimj[256]) * (K[68]))) / (Ghimj[249])); 
# 1069
(K[59]) = (((((((((K[59]) - ((Ghimj[241]) * (K[61]))) - ((Ghimj[242]) * (K[63]))) - ((Ghimj[243]) * (K[65]))) - ((Ghimj[244]) * (K[67]))) - ((Ghimj[245]) * (K[68]))) - ((Ghimj[246]) * (K[69]))) - ((Ghimj[247]) * (K[72]))) / (Ghimj[240])); 
# 1070
(K[58]) = ((((((((K[58]) - ((Ghimj[233]) * (K[62]))) - ((Ghimj[234]) * (K[64]))) - ((Ghimj[235]) * (K[65]))) - ((Ghimj[236]) * (K[67]))) - ((Ghimj[237]) * (K[68]))) - ((Ghimj[238]) * (K[71]))) / (Ghimj[232])); 
# 1071
(K[57]) = ((((((((K[57]) - ((Ghimj[219]) * (K[62]))) - ((Ghimj[220]) * (K[64]))) - ((Ghimj[221]) * (K[65]))) - ((Ghimj[222]) * (K[67]))) - ((Ghimj[223]) * (K[68]))) - ((Ghimj[224]) * (K[71]))) / (Ghimj[218])); 
# 1072
(K[56]) = ((((((((K[56]) - ((Ghimj[208]) * (K[59]))) - ((Ghimj[209]) * (K[63]))) - ((Ghimj[210]) * (K[67]))) - ((Ghimj[211]) * (K[68]))) - ((Ghimj[212]) * (K[69]))) - ((Ghimj[213]) * (K[72]))) / (Ghimj[207])); 
# 1073
(K[55]) = ((((((K[55]) - ((Ghimj[202]) * (K[61]))) - ((Ghimj[203]) * (K[63]))) - ((Ghimj[204]) * (K[69]))) - ((Ghimj[205]) * (K[72]))) / (Ghimj[201])); 
# 1074
(K[54]) = (((((((K[54]) - ((Ghimj[195]) * (K[56]))) - ((Ghimj[196]) * (K[63]))) - ((Ghimj[197]) * (K[67]))) - ((Ghimj[198]) * (K[68]))) - ((Ghimj[199]) * (K[69]))) / (Ghimj[194])); 
# 1075
(K[53]) = (((((((K[53]) - ((Ghimj[189]) * (K[61]))) - ((Ghimj[190]) * (K[63]))) - ((Ghimj[191]) * (K[64]))) - ((Ghimj[192]) * (K[67]))) - ((Ghimj[193]) * (K[68]))) / (Ghimj[188])); 
# 1076
(K[52]) = (((((K[52]) - ((Ghimj[185]) * (K[65]))) - ((Ghimj[186]) * (K[67]))) - ((Ghimj[187]) * (K[71]))) / (Ghimj[184])); 
# 1077
(K[51]) = ((((K[51]) - ((Ghimj[182]) * (K[67]))) - ((Ghimj[183]) * (K[68]))) / (Ghimj[181])); 
# 1078
(K[50]) = (((((K[50]) - ((Ghimj[178]) * (K[64]))) - ((Ghimj[179]) * (K[65]))) - ((Ghimj[180]) * (K[72]))) / (Ghimj[177])); 
# 1079
(K[49]) = ((((K[49]) - ((Ghimj[175]) * (K[67]))) - ((Ghimj[176]) * (K[68]))) / (Ghimj[174])); 
# 1080
(K[48]) = ((((K[48]) - ((Ghimj[172]) * (K[69]))) - ((Ghimj[173]) * (K[72]))) / (Ghimj[171])); 
# 1081
(K[47]) = (((((((K[47]) - ((Ghimj[166]) * (K[52]))) - ((Ghimj[167]) * (K[57]))) - ((Ghimj[168]) * (K[58]))) - ((Ghimj[169]) * (K[65]))) - ((Ghimj[170]) * (K[71]))) / (Ghimj[165])); 
# 1082
(K[46]) = ((((K[46]) - ((Ghimj[153]) * (K[67]))) - ((Ghimj[154]) * (K[68]))) / (Ghimj[152])); 
# 1083
(K[45]) = ((((K[45]) - ((Ghimj[150]) * (K[67]))) - ((Ghimj[151]) * (K[68]))) / (Ghimj[149])); 
# 1084
(K[44]) = (((((K[44]) - ((Ghimj[146]) * (K[50]))) - ((Ghimj[147]) * (K[60]))) - ((Ghimj[148]) * (K[65]))) / (Ghimj[145])); 
# 1085
(K[43]) = (((K[43]) - ((Ghimj[144]) * (K[67]))) / (Ghimj[143])); 
# 1086
(K[42]) = (((K[42]) - ((Ghimj[142]) * (K[67]))) / (Ghimj[141])); 
# 1087
(K[41]) = (((K[41]) - ((Ghimj[140]) * (K[47]))) / (Ghimj[139])); 
# 1088
(K[40]) = (((K[40]) - ((Ghimj[138]) * (K[65]))) / (Ghimj[137])); 
# 1089
(K[39]) = (((K[39]) - ((Ghimj[136]) * (K[65]))) / (Ghimj[135])); 
# 1090
(K[38]) = (((((K[38]) - ((Ghimj[132]) * (K[60]))) - ((Ghimj[133]) * (K[66]))) - ((Ghimj[134]) * (K[71]))) / (Ghimj[131])); 
# 1091
(K[37]) = (((K[37]) - ((Ghimj[130]) * (K[65]))) / (Ghimj[129])); 
# 1092
(K[36]) = (((K[36]) - ((Ghimj[128]) * (K[65]))) / (Ghimj[127])); 
# 1093
(K[35]) = (((K[35]) - ((Ghimj[126]) * (K[65]))) / (Ghimj[125])); 
# 1094
(K[34]) = (((K[34]) - ((Ghimj[124]) * (K[65]))) / (Ghimj[123])); 
# 1095
(K[33]) = (((K[33]) - ((Ghimj[122]) * (K[65]))) / (Ghimj[121])); 
# 1096
(K[32]) = (((K[32]) - ((Ghimj[120]) * (K[65]))) / (Ghimj[119])); 
# 1097
(K[31]) = ((((K[31]) - ((Ghimj[117]) * (K[40]))) - ((Ghimj[118]) * (K[65]))) / (Ghimj[116])); 
# 1098
(K[30]) = (((K[30]) - ((Ghimj[115]) * (K[69]))) / (Ghimj[114])); 
# 1099
(K[29]) = (((K[29]) - ((Ghimj[113]) * (K[47]))) / (Ghimj[112])); 
# 1100
(K[28]) = (((K[28]) - ((Ghimj[111]) * (K[41]))) / (Ghimj[110])); 
# 1101
(K[27]) = ((((((((K[27]) - ((Ghimj[104]) * (K[33]))) - ((Ghimj[105]) * (K[34]))) - ((Ghimj[106]) * (K[35]))) - ((Ghimj[107]) * (K[36]))) - ((Ghimj[108]) * (K[37]))) - ((Ghimj[109]) * (K[65]))) / (Ghimj[103])); 
# 1102
(K[26]) = ((((((K[26]) - ((Ghimj[99]) * (K[50]))) - ((Ghimj[100]) * (K[59]))) - ((Ghimj[101]) * (K[60]))) - ((Ghimj[102]) * (K[65]))) / (Ghimj[98])); 
# 1103
(K[25]) = ((((K[25]) - ((Ghimj[96]) * (K[39]))) - ((Ghimj[97]) * (K[65]))) / (Ghimj[95])); 
# 1104
(K[24]) = ((((K[24]) - ((Ghimj[93]) * (K[32]))) - ((Ghimj[94]) * (K[65]))) / (Ghimj[92])); 
# 1105
(K[23]) = ((((K[23]) - ((Ghimj[90]) * (K[65]))) - ((Ghimj[91]) * (K[66]))) / (Ghimj[89])); 
# 1106
(K[22]) = ((((K[22]) - ((Ghimj[87]) * (K[41]))) - ((Ghimj[88]) * (K[47]))) / (Ghimj[86])); 
# 1107
(K[21]) = (((((K[21]) - ((Ghimj[83]) * (K[48]))) - ((Ghimj[84]) * (K[69]))) - ((Ghimj[85]) * (K[72]))) / (Ghimj[82])); 
# 1108
(K[20]) = (((((K[20]) - ((Ghimj[79]) * (K[65]))) - ((Ghimj[80]) * (K[66]))) - ((Ghimj[81]) * (K[71]))) / (Ghimj[78])); 
# 1109
(K[19]) = ((((((K[19]) - ((Ghimj[74]) * (K[48]))) - ((Ghimj[75]) * (K[69]))) - ((Ghimj[76]) * (K[70]))) - ((Ghimj[77]) * (K[72]))) / (Ghimj[73])); 
# 1110
(K[18]) = ((((((K[18]) - ((Ghimj[69]) * (K[48]))) - ((Ghimj[70]) * (K[69]))) - ((Ghimj[71]) * (K[70]))) - ((Ghimj[72]) * (K[72]))) / (Ghimj[68])); 
# 1111
(K[17]) = ((((((((((((((K[17]) - ((Ghimj[56]) * (K[41]))) - ((Ghimj[57]) * (K[44]))) - ((Ghimj[58]) * (K[47]))) - ((Ghimj[59]) * (K[53]))) - ((Ghimj[60]) * (K[60]))) - ((Ghimj[61]) * (K[64]))) - ((Ghimj[62]) * (K[65]))) - ((Ghimj[63]) * (K[66]))) - ((Ghimj[64]) * (K[69]))) - ((Ghimj[65]) * (K[70]))) - ((Ghimj[66]) * (K[71]))) - ((Ghimj[67]) * (K[72]))) / (Ghimj[55])); 
# 1112
(K[16]) = ((((K[16]) - ((Ghimj[52]) * (K[41]))) - ((Ghimj[53]) * (K[47]))) / (Ghimj[51])); 
# 1113
(K[15]) = ((((K[15]) - ((Ghimj[49]) * (K[66]))) - ((Ghimj[50]) * (K[71]))) / (Ghimj[48])); 
# 1114
(K[14]) = (((K[14]) - ((Ghimj[47]) * (K[71]))) / (Ghimj[46])); 
# 1115
(K[13]) = ((((K[13]) - ((Ghimj[44]) * (K[66]))) - ((Ghimj[45]) * (K[71]))) / (Ghimj[43])); 
# 1116
(K[12]) = ((((K[12]) - ((Ghimj[41]) * (K[65]))) - ((Ghimj[42]) * (K[66]))) / (Ghimj[40])); 
# 1117
(K[11]) = (((((K[11]) - ((Ghimj[37]) * (K[45]))) - ((Ghimj[38]) * (K[67]))) - ((Ghimj[39]) * (K[68]))) / (Ghimj[36])); 
# 1118
(K[10]) = ((K[10]) / (Ghimj[35])); 
# 1119
(K[9]) = ((K[9]) / (Ghimj[33])); 
# 1120
(K[8]) = ((((((K[8]) - ((Ghimj[28]) * (K[38]))) - ((Ghimj[29]) * (K[50]))) - ((Ghimj[30]) * (K[65]))) - ((Ghimj[31]) * (K[71]))) / (Ghimj[27])); 
# 1121
(K[7]) = ((((K[7]) - ((Ghimj[25]) * (K[44]))) - ((Ghimj[26]) * (K[65]))) / (Ghimj[24])); 
# 1122
(K[6]) = (((((((K[6]) - ((Ghimj[19]) * (K[59]))) - ((Ghimj[20]) * (K[65]))) - ((Ghimj[21]) * (K[70]))) - ((Ghimj[22]) * (K[71]))) - ((Ghimj[23]) * (K[72]))) / (Ghimj[18])); 
# 1123
(K[5]) = ((((K[5]) - ((Ghimj[16]) * (K[69]))) - ((Ghimj[17]) * (K[72]))) / (Ghimj[15])); 
# 1124
(K[4]) = ((((K[4]) - ((Ghimj[13]) * (K[69]))) - ((Ghimj[14]) * (K[71]))) / (Ghimj[12])); 
# 1125
(K[3]) = (((((K[3]) - ((Ghimj[9]) * (K[46]))) - ((Ghimj[10]) * (K[67]))) - ((Ghimj[11]) * (K[68]))) / (Ghimj[8])); 
# 1126
(K[2]) = (((((K[2]) - ((Ghimj[5]) * (K[62]))) - ((Ghimj[6]) * (K[67]))) - ((Ghimj[7]) * (K[68]))) / (Ghimj[4])); 
# 1127
(K[1]) = ((((K[1]) - ((Ghimj[2]) * (K[53]))) - ((Ghimj[3]) * (K[64]))) / (Ghimj[1])); 
# 1128
(K[0]) = ((K[0]) / (Ghimj[0])); 
# 1129
} 
#endif
# 1131 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) void ros_Solve(double *__restrict__ Ghimj, double *__restrict__ K, int &Nsol, const int istage, const int ros_S) 
# 1132
{int volatile ___ = 1;(void)Ghimj;(void)K;(void)Nsol;(void)istage;(void)ros_S;
# 1141
::exit(___);}
#if 0
# 1132
{ 
# 1135
#pragma unroll 4
for (
# 1135
int i = 0; i < (455 - 16); i += 16) { 
# 1136
prefetch_ll1(&(Ghimj[i])); 
# 1137
}  
# 1139
kppSolve(Ghimj, K, istage, ros_S); 
# 1140
Nsol++; 
# 1141
} 
#endif
# 1143 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) void kppDecomp(double *Ghimj, int VL_GLO) 
# 1144
{int volatile ___ = 1;(void)Ghimj;(void)VL_GLO;
# 2624
::exit(___);}
#if 0
# 1144
{ 
# 1145
double a = (0.0); 
# 1147
double dummy, W_0, W_1, W_2, W_3, W_4, W_5, W_6, W_7, W_8, W_9, W_10, W_11, W_12, W_13, W_14, W_15, W_16, W_17, W_18, W_19, W_20, W_21, W_22, W_23, W_24, W_25, W_26, W_27, W_28, W_29, W_30, W_31, W_32, W_33, W_34, W_35, W_36, W_37, W_38, W_39, W_40, W_41, W_42, W_43, W_44, W_45, W_46, W_47, W_48, W_49, W_50, W_51, W_52, W_53, W_54, W_55, W_56, W_57, W_58, W_59, W_60, W_61, W_62, W_63, W_64, W_65, W_66, W_67, W_68, W_69, W_70, W_71, W_72, W_73; 
# 1149
W_0 = (Ghimj[32]); 
# 1150
W_9 = (Ghimj[33]); 
# 1151
a = ((-W_0) / (Ghimj[0])); 
# 1152
W_0 = (-a); 
# 1153
(Ghimj[32]) = W_0; 
# 1154
(Ghimj[33]) = W_9; 
# 1155
W_0 = (Ghimj[34]); 
# 1156
W_10 = (Ghimj[35]); 
# 1157
a = ((-W_0) / (Ghimj[0])); 
# 1158
W_0 = (-a); 
# 1159
(Ghimj[34]) = W_0; 
# 1160
(Ghimj[35]) = W_10; 
# 1161
W_0 = (Ghimj[54]); 
# 1162
W_17 = (Ghimj[55]); 
# 1163
W_41 = (Ghimj[56]); 
# 1164
W_44 = (Ghimj[57]); 
# 1165
W_47 = (Ghimj[58]); 
# 1166
W_53 = (Ghimj[59]); 
# 1167
W_60 = (Ghimj[60]); 
# 1168
W_64 = (Ghimj[61]); 
# 1169
W_65 = (Ghimj[62]); 
# 1170
W_66 = (Ghimj[63]); 
# 1171
W_69 = (Ghimj[64]); 
# 1172
W_70 = (Ghimj[65]); 
# 1173
W_71 = (Ghimj[66]); 
# 1174
W_72 = (Ghimj[67]); 
# 1175
a = ((-W_0) / (Ghimj[0])); 
# 1176
W_0 = (-a); 
# 1177
(Ghimj[54]) = W_0; 
# 1178
(Ghimj[55]) = W_17; 
# 1179
(Ghimj[56]) = W_41; 
# 1180
(Ghimj[57]) = W_44; 
# 1181
(Ghimj[58]) = W_47; 
# 1182
(Ghimj[59]) = W_53; 
# 1183
(Ghimj[60]) = W_60; 
# 1184
(Ghimj[61]) = W_64; 
# 1185
(Ghimj[62]) = W_65; 
# 1186
(Ghimj[63]) = W_66; 
# 1187
(Ghimj[64]) = W_69; 
# 1188
(Ghimj[65]) = W_70; 
# 1189
(Ghimj[66]) = W_71; 
# 1190
(Ghimj[67]) = W_72; 
# 1191
W_29 = (Ghimj[155]); 
# 1192
W_32 = (Ghimj[156]); 
# 1193
W_33 = (Ghimj[157]); 
# 1194
W_34 = (Ghimj[158]); 
# 1195
W_35 = (Ghimj[159]); 
# 1196
W_36 = (Ghimj[160]); 
# 1197
W_37 = (Ghimj[161]); 
# 1198
W_39 = (Ghimj[162]); 
# 1199
W_40 = (Ghimj[163]); 
# 1200
W_41 = (Ghimj[164]); 
# 1201
W_47 = (Ghimj[165]); 
# 1202
W_52 = (Ghimj[166]); 
# 1203
W_57 = (Ghimj[167]); 
# 1204
W_58 = (Ghimj[168]); 
# 1205
W_65 = (Ghimj[169]); 
# 1206
W_71 = (Ghimj[170]); 
# 1207
a = ((-W_29) / (Ghimj[112])); 
# 1208
W_29 = (-a); 
# 1209
W_47 = (W_47 + (a * (Ghimj[113]))); 
# 1210
a = ((-W_32) / (Ghimj[119])); 
# 1211
W_32 = (-a); 
# 1212
W_65 = (W_65 + (a * (Ghimj[120]))); 
# 1213
a = ((-W_33) / (Ghimj[121])); 
# 1214
W_33 = (-a); 
# 1215
W_65 = (W_65 + (a * (Ghimj[122]))); 
# 1216
a = ((-W_34) / (Ghimj[123])); 
# 1217
W_34 = (-a); 
# 1218
W_65 = (W_65 + (a * (Ghimj[124]))); 
# 1219
a = ((-W_35) / (Ghimj[125])); 
# 1220
W_35 = (-a); 
# 1221
W_65 = (W_65 + (a * (Ghimj[126]))); 
# 1222
a = ((-W_36) / (Ghimj[127])); 
# 1223
W_36 = (-a); 
# 1224
W_65 = (W_65 + (a * (Ghimj[128]))); 
# 1225
a = ((-W_37) / (Ghimj[129])); 
# 1226
W_37 = (-a); 
# 1227
W_65 = (W_65 + (a * (Ghimj[130]))); 
# 1228
a = ((-W_39) / (Ghimj[135])); 
# 1229
W_39 = (-a); 
# 1230
W_65 = (W_65 + (a * (Ghimj[136]))); 
# 1231
a = ((-W_40) / (Ghimj[137])); 
# 1232
W_40 = (-a); 
# 1233
W_65 = (W_65 + (a * (Ghimj[138]))); 
# 1234
a = ((-W_41) / (Ghimj[139])); 
# 1235
W_41 = (-a); 
# 1236
W_47 = (W_47 + (a * (Ghimj[140]))); 
# 1237
(Ghimj[155]) = W_29; 
# 1238
(Ghimj[156]) = W_32; 
# 1239
(Ghimj[157]) = W_33; 
# 1240
(Ghimj[158]) = W_34; 
# 1241
(Ghimj[159]) = W_35; 
# 1242
(Ghimj[160]) = W_36; 
# 1243
(Ghimj[161]) = W_37; 
# 1244
(Ghimj[162]) = W_39; 
# 1245
(Ghimj[163]) = W_40; 
# 1246
(Ghimj[164]) = W_41; 
# 1247
(Ghimj[165]) = W_47; 
# 1248
(Ghimj[166]) = W_52; 
# 1249
(Ghimj[167]) = W_57; 
# 1250
(Ghimj[168]) = W_58; 
# 1251
(Ghimj[169]) = W_65; 
# 1252
(Ghimj[170]) = W_71; 
# 1253
W_48 = (Ghimj[200]); 
# 1254
W_55 = (Ghimj[201]); 
# 1255
W_61 = (Ghimj[202]); 
# 1256
W_63 = (Ghimj[203]); 
# 1257
W_69 = (Ghimj[204]); 
# 1258
W_72 = (Ghimj[205]); 
# 1259
a = ((-W_48) / (Ghimj[171])); 
# 1260
W_48 = (-a); 
# 1261
W_69 = (W_69 + (a * (Ghimj[172]))); 
# 1262
W_72 = (W_72 + (a * (Ghimj[173]))); 
# 1263
(Ghimj[200]) = W_48; 
# 1264
(Ghimj[201]) = W_55; 
# 1265
(Ghimj[202]) = W_61; 
# 1266
(Ghimj[203]) = W_63; 
# 1267
(Ghimj[204]) = W_69; 
# 1268
(Ghimj[205]) = W_72; 
# 1269
W_54 = (Ghimj[206]); 
# 1270
W_56 = (Ghimj[207]); 
# 1271
W_59 = (Ghimj[208]); 
# 1272
W_63 = (Ghimj[209]); 
# 1273
W_67 = (Ghimj[210]); 
# 1274
W_68 = (Ghimj[211]); 
# 1275
W_69 = (Ghimj[212]); 
# 1276
W_72 = (Ghimj[213]); 
# 1277
a = ((-W_54) / (Ghimj[194])); 
# 1278
W_54 = (-a); 
# 1279
W_56 = (W_56 + (a * (Ghimj[195]))); 
# 1280
W_63 = (W_63 + (a * (Ghimj[196]))); 
# 1281
W_67 = (W_67 + (a * (Ghimj[197]))); 
# 1282
W_68 = (W_68 + (a * (Ghimj[198]))); 
# 1283
W_69 = (W_69 + (a * (Ghimj[199]))); 
# 1284
(Ghimj[206]) = W_54; 
# 1285
(Ghimj[207]) = W_56; 
# 1286
(Ghimj[208]) = W_59; 
# 1287
(Ghimj[209]) = W_63; 
# 1288
(Ghimj[210]) = W_67; 
# 1289
(Ghimj[211]) = W_68; 
# 1290
(Ghimj[212]) = W_69; 
# 1291
(Ghimj[213]) = W_72; 
# 1292
W_45 = (Ghimj[214]); 
# 1293
W_46 = (Ghimj[215]); 
# 1294
W_49 = (Ghimj[216]); 
# 1295
W_51 = (Ghimj[217]); 
# 1296
W_57 = (Ghimj[218]); 
# 1297
W_62 = (Ghimj[219]); 
# 1298
W_64 = (Ghimj[220]); 
# 1299
W_65 = (Ghimj[221]); 
# 1300
W_67 = (Ghimj[222]); 
# 1301
W_68 = (Ghimj[223]); 
# 1302
W_71 = (Ghimj[224]); 
# 1303
a = ((-W_45) / (Ghimj[149])); 
# 1304
W_45 = (-a); 
# 1305
W_67 = (W_67 + (a * (Ghimj[150]))); 
# 1306
W_68 = (W_68 + (a * (Ghimj[151]))); 
# 1307
a = ((-W_46) / (Ghimj[152])); 
# 1308
W_46 = (-a); 
# 1309
W_67 = (W_67 + (a * (Ghimj[153]))); 
# 1310
W_68 = (W_68 + (a * (Ghimj[154]))); 
# 1311
a = ((-W_49) / (Ghimj[174])); 
# 1312
W_49 = (-a); 
# 1313
W_67 = (W_67 + (a * (Ghimj[175]))); 
# 1314
W_68 = (W_68 + (a * (Ghimj[176]))); 
# 1315
a = ((-W_51) / (Ghimj[181])); 
# 1316
W_51 = (-a); 
# 1317
W_67 = (W_67 + (a * (Ghimj[182]))); 
# 1318
W_68 = (W_68 + (a * (Ghimj[183]))); 
# 1319
(Ghimj[214]) = W_45; 
# 1320
(Ghimj[215]) = W_46; 
# 1321
(Ghimj[216]) = W_49; 
# 1322
(Ghimj[217]) = W_51; 
# 1323
(Ghimj[218]) = W_57; 
# 1324
(Ghimj[219]) = W_62; 
# 1325
(Ghimj[220]) = W_64; 
# 1326
(Ghimj[221]) = W_65; 
# 1327
(Ghimj[222]) = W_67; 
# 1328
(Ghimj[223]) = W_68; 
# 1329
(Ghimj[224]) = W_71; 
# 1330
W_42 = (Ghimj[225]); 
# 1331
W_43 = (Ghimj[226]); 
# 1332
W_45 = (Ghimj[227]); 
# 1333
W_46 = (Ghimj[228]); 
# 1334
W_49 = (Ghimj[229]); 
# 1335
W_51 = (Ghimj[230]); 
# 1336
W_52 = (Ghimj[231]); 
# 1337
W_58 = (Ghimj[232]); 
# 1338
W_62 = (Ghimj[233]); 
# 1339
W_64 = (Ghimj[234]); 
# 1340
W_65 = (Ghimj[235]); 
# 1341
W_67 = (Ghimj[236]); 
# 1342
W_68 = (Ghimj[237]); 
# 1343
W_71 = (Ghimj[238]); 
# 1344
a = ((-W_42) / (Ghimj[141])); 
# 1345
W_42 = (-a); 
# 1346
W_67 = (W_67 + (a * (Ghimj[142]))); 
# 1347
a = ((-W_43) / (Ghimj[143])); 
# 1348
W_43 = (-a); 
# 1349
W_67 = (W_67 + (a * (Ghimj[144]))); 
# 1350
a = ((-W_45) / (Ghimj[149])); 
# 1351
W_45 = (-a); 
# 1352
W_67 = (W_67 + (a * (Ghimj[150]))); 
# 1353
W_68 = (W_68 + (a * (Ghimj[151]))); 
# 1354
a = ((-W_46) / (Ghimj[152])); 
# 1355
W_46 = (-a); 
# 1356
W_67 = (W_67 + (a * (Ghimj[153]))); 
# 1357
W_68 = (W_68 + (a * (Ghimj[154]))); 
# 1358
a = ((-W_49) / (Ghimj[174])); 
# 1359
W_49 = (-a); 
# 1360
W_67 = (W_67 + (a * (Ghimj[175]))); 
# 1361
W_68 = (W_68 + (a * (Ghimj[176]))); 
# 1362
a = ((-W_51) / (Ghimj[181])); 
# 1363
W_51 = (-a); 
# 1364
W_67 = (W_67 + (a * (Ghimj[182]))); 
# 1365
W_68 = (W_68 + (a * (Ghimj[183]))); 
# 1366
a = ((-W_52) / (Ghimj[184])); 
# 1367
W_52 = (-a); 
# 1368
W_65 = (W_65 + (a * (Ghimj[185]))); 
# 1369
W_67 = (W_67 + (a * (Ghimj[186]))); 
# 1370
W_71 = (W_71 + (a * (Ghimj[187]))); 
# 1371
(Ghimj[225]) = W_42; 
# 1372
(Ghimj[226]) = W_43; 
# 1373
(Ghimj[227]) = W_45; 
# 1374
(Ghimj[228]) = W_46; 
# 1375
(Ghimj[229]) = W_49; 
# 1376
(Ghimj[230]) = W_51; 
# 1377
(Ghimj[231]) = W_52; 
# 1378
(Ghimj[232]) = W_58; 
# 1379
(Ghimj[233]) = W_62; 
# 1380
(Ghimj[234]) = W_64; 
# 1381
(Ghimj[235]) = W_65; 
# 1382
(Ghimj[236]) = W_67; 
# 1383
(Ghimj[237]) = W_68; 
# 1384
(Ghimj[238]) = W_71; 
# 1385
W_56 = (Ghimj[239]); 
# 1386
W_59 = (Ghimj[240]); 
# 1387
W_61 = (Ghimj[241]); 
# 1388
W_63 = (Ghimj[242]); 
# 1389
W_65 = (Ghimj[243]); 
# 1390
W_67 = (Ghimj[244]); 
# 1391
W_68 = (Ghimj[245]); 
# 1392
W_69 = (Ghimj[246]); 
# 1393
W_72 = (Ghimj[247]); 
# 1394
a = ((-W_56) / (Ghimj[207])); 
# 1395
W_56 = (-a); 
# 1396
W_59 = (W_59 + (a * (Ghimj[208]))); 
# 1397
W_63 = (W_63 + (a * (Ghimj[209]))); 
# 1398
W_67 = (W_67 + (a * (Ghimj[210]))); 
# 1399
W_68 = (W_68 + (a * (Ghimj[211]))); 
# 1400
W_69 = (W_69 + (a * (Ghimj[212]))); 
# 1401
W_72 = (W_72 + (a * (Ghimj[213]))); 
# 1402
(Ghimj[239]) = W_56; 
# 1403
(Ghimj[240]) = W_59; 
# 1404
(Ghimj[241]) = W_61; 
# 1405
(Ghimj[242]) = W_63; 
# 1406
(Ghimj[243]) = W_65; 
# 1407
(Ghimj[244]) = W_67; 
# 1408
(Ghimj[245]) = W_68; 
# 1409
(Ghimj[246]) = W_69; 
# 1410
(Ghimj[247]) = W_72; 
# 1411
W_53 = (Ghimj[248]); 
# 1412
W_60 = (Ghimj[249]); 
# 1413
W_61 = (Ghimj[250]); 
# 1414
W_63 = (Ghimj[251]); 
# 1415
W_64 = (Ghimj[252]); 
# 1416
W_65 = (Ghimj[253]); 
# 1417
W_66 = (Ghimj[254]); 
# 1418
W_67 = (Ghimj[255]); 
# 1419
W_68 = (Ghimj[256]); 
# 1420
a = ((-W_53) / (Ghimj[188])); 
# 1421
W_53 = (-a); 
# 1422
W_61 = (W_61 + (a * (Ghimj[189]))); 
# 1423
W_63 = (W_63 + (a * (Ghimj[190]))); 
# 1424
W_64 = (W_64 + (a * (Ghimj[191]))); 
# 1425
W_67 = (W_67 + (a * (Ghimj[192]))); 
# 1426
W_68 = (W_68 + (a * (Ghimj[193]))); 
# 1427
(Ghimj[248]) = W_53; 
# 1428
(Ghimj[249]) = W_60; 
# 1429
(Ghimj[250]) = W_61; 
# 1430
(Ghimj[251]) = W_63; 
# 1431
(Ghimj[252]) = W_64; 
# 1432
(Ghimj[253]) = W_65; 
# 1433
(Ghimj[254]) = W_66; 
# 1434
(Ghimj[255]) = W_67; 
# 1435
(Ghimj[256]) = W_68; 
# 1436
W_55 = (Ghimj[257]); 
# 1437
W_61 = (Ghimj[258]); 
# 1438
W_63 = (Ghimj[259]); 
# 1439
W_68 = (Ghimj[260]); 
# 1440
W_69 = (Ghimj[261]); 
# 1441
W_72 = (Ghimj[262]); 
# 1442
a = ((-W_55) / (Ghimj[201])); 
# 1443
W_55 = (-a); 
# 1444
W_61 = (W_61 + (a * (Ghimj[202]))); 
# 1445
W_63 = (W_63 + (a * (Ghimj[203]))); 
# 1446
W_69 = (W_69 + (a * (Ghimj[204]))); 
# 1447
W_72 = (W_72 + (a * (Ghimj[205]))); 
# 1448
(Ghimj[257]) = W_55; 
# 1449
(Ghimj[258]) = W_61; 
# 1450
(Ghimj[259]) = W_63; 
# 1451
(Ghimj[260]) = W_68; 
# 1452
(Ghimj[261]) = W_69; 
# 1453
(Ghimj[262]) = W_72; 
# 1454
W_50 = (Ghimj[263]); 
# 1455
W_51 = (Ghimj[264]); 
# 1456
W_53 = (Ghimj[265]); 
# 1457
W_61 = (Ghimj[266]); 
# 1458
W_62 = (Ghimj[267]); 
# 1459
W_63 = (Ghimj[268]); 
# 1460
W_64 = (Ghimj[269]); 
# 1461
W_65 = (Ghimj[270]); 
# 1462
W_67 = (Ghimj[271]); 
# 1463
W_68 = (Ghimj[272]); 
# 1464
W_69 = (Ghimj[273]); 
# 1465
W_70 = (Ghimj[274]); 
# 1466
W_72 = (Ghimj[275]); 
# 1467
a = ((-W_50) / (Ghimj[177])); 
# 1468
W_50 = (-a); 
# 1469
W_64 = (W_64 + (a * (Ghimj[178]))); 
# 1470
W_65 = (W_65 + (a * (Ghimj[179]))); 
# 1471
W_72 = (W_72 + (a * (Ghimj[180]))); 
# 1472
a = ((-W_51) / (Ghimj[181])); 
# 1473
W_51 = (-a); 
# 1474
W_67 = (W_67 + (a * (Ghimj[182]))); 
# 1475
W_68 = (W_68 + (a * (Ghimj[183]))); 
# 1476
a = ((-W_53) / (Ghimj[188])); 
# 1477
W_53 = (-a); 
# 1478
W_61 = (W_61 + (a * (Ghimj[189]))); 
# 1479
W_63 = (W_63 + (a * (Ghimj[190]))); 
# 1480
W_64 = (W_64 + (a * (Ghimj[191]))); 
# 1481
W_67 = (W_67 + (a * (Ghimj[192]))); 
# 1482
W_68 = (W_68 + (a * (Ghimj[193]))); 
# 1483
a = ((-W_61) / (Ghimj[258])); 
# 1484
W_61 = (-a); 
# 1485
W_63 = (W_63 + (a * (Ghimj[259]))); 
# 1486
W_68 = (W_68 + (a * (Ghimj[260]))); 
# 1487
W_69 = (W_69 + (a * (Ghimj[261]))); 
# 1488
W_72 = (W_72 + (a * (Ghimj[262]))); 
# 1489
(Ghimj[263]) = W_50; 
# 1490
(Ghimj[264]) = W_51; 
# 1491
(Ghimj[265]) = W_53; 
# 1492
(Ghimj[266]) = W_61; 
# 1493
(Ghimj[267]) = W_62; 
# 1494
(Ghimj[268]) = W_63; 
# 1495
(Ghimj[269]) = W_64; 
# 1496
(Ghimj[270]) = W_65; 
# 1497
(Ghimj[271]) = W_67; 
# 1498
(Ghimj[272]) = W_68; 
# 1499
(Ghimj[273]) = W_69; 
# 1500
(Ghimj[274]) = W_70; 
# 1501
(Ghimj[275]) = W_72; 
# 1502
W_55 = (Ghimj[276]); 
# 1503
W_61 = (Ghimj[277]); 
# 1504
W_63 = (Ghimj[278]); 
# 1505
W_67 = (Ghimj[279]); 
# 1506
W_68 = (Ghimj[280]); 
# 1507
W_69 = (Ghimj[281]); 
# 1508
W_72 = (Ghimj[282]); 
# 1509
a = ((-W_55) / (Ghimj[201])); 
# 1510
W_55 = (-a); 
# 1511
W_61 = (W_61 + (a * (Ghimj[202]))); 
# 1512
W_63 = (W_63 + (a * (Ghimj[203]))); 
# 1513
W_69 = (W_69 + (a * (Ghimj[204]))); 
# 1514
W_72 = (W_72 + (a * (Ghimj[205]))); 
# 1515
a = ((-W_61) / (Ghimj[258])); 
# 1516
W_61 = (-a); 
# 1517
W_63 = (W_63 + (a * (Ghimj[259]))); 
# 1518
W_68 = (W_68 + (a * (Ghimj[260]))); 
# 1519
W_69 = (W_69 + (a * (Ghimj[261]))); 
# 1520
W_72 = (W_72 + (a * (Ghimj[262]))); 
# 1521
(Ghimj[276]) = W_55; 
# 1522
(Ghimj[277]) = W_61; 
# 1523
(Ghimj[278]) = W_63; 
# 1524
(Ghimj[279]) = W_67; 
# 1525
(Ghimj[280]) = W_68; 
# 1526
(Ghimj[281]) = W_69; 
# 1527
(Ghimj[282]) = W_72; 
# 1528
W_53 = (Ghimj[283]); 
# 1529
W_61 = (Ghimj[284]); 
# 1530
W_63 = (Ghimj[285]); 
# 1531
W_64 = (Ghimj[286]); 
# 1532
W_65 = (Ghimj[287]); 
# 1533
W_67 = (Ghimj[288]); 
# 1534
W_68 = (Ghimj[289]); 
# 1535
W_69 = (Ghimj[290]); 
# 1536
W_72 = (Ghimj[291]); 
# 1537
a = ((-W_53) / (Ghimj[188])); 
# 1538
W_53 = (-a); 
# 1539
W_61 = (W_61 + (a * (Ghimj[189]))); 
# 1540
W_63 = (W_63 + (a * (Ghimj[190]))); 
# 1541
W_64 = (W_64 + (a * (Ghimj[191]))); 
# 1542
W_67 = (W_67 + (a * (Ghimj[192]))); 
# 1543
W_68 = (W_68 + (a * (Ghimj[193]))); 
# 1544
a = ((-W_61) / (Ghimj[258])); 
# 1545
W_61 = (-a); 
# 1546
W_63 = (W_63 + (a * (Ghimj[259]))); 
# 1547
W_68 = (W_68 + (a * (Ghimj[260]))); 
# 1548
W_69 = (W_69 + (a * (Ghimj[261]))); 
# 1549
W_72 = (W_72 + (a * (Ghimj[262]))); 
# 1550
a = ((-W_63) / (Ghimj[278])); 
# 1551
W_63 = (-a); 
# 1552
W_67 = (W_67 + (a * (Ghimj[279]))); 
# 1553
W_68 = (W_68 + (a * (Ghimj[280]))); 
# 1554
W_69 = (W_69 + (a * (Ghimj[281]))); 
# 1555
W_72 = (W_72 + (a * (Ghimj[282]))); 
# 1556
(Ghimj[283]) = W_53; 
# 1557
(Ghimj[284]) = W_61; 
# 1558
(Ghimj[285]) = W_63; 
# 1559
(Ghimj[286]) = W_64; 
# 1560
(Ghimj[287]) = W_65; 
# 1561
(Ghimj[288]) = W_67; 
# 1562
(Ghimj[289]) = W_68; 
# 1563
(Ghimj[290]) = W_69; 
# 1564
(Ghimj[291]) = W_72; 
# 1565
W_32 = (Ghimj[292]); 
# 1566
W_33 = (Ghimj[293]); 
# 1567
W_34 = (Ghimj[294]); 
# 1568
W_35 = (Ghimj[295]); 
# 1569
W_36 = (Ghimj[296]); 
# 1570
W_37 = (Ghimj[297]); 
# 1571
W_39 = (Ghimj[298]); 
# 1572
W_40 = (Ghimj[299]); 
# 1573
W_41 = (Ghimj[300]); 
# 1574
W_44 = (Ghimj[301]); 
# 1575
W_47 = (Ghimj[302]); 
# 1576
W_50 = (Ghimj[303]); 
# 1577
W_51 = (Ghimj[304]); 
# 1578
W_52 = (Ghimj[305]); 
# 1579
W_57 = (Ghimj[306]); 
# 1580
W_58 = (Ghimj[307]); 
# 1581
W_59 = (Ghimj[308]); 
# 1582
W_60 = (Ghimj[309]); 
# 1583
W_61 = (Ghimj[310]); 
# 1584
W_62 = (Ghimj[311]); 
# 1585
W_63 = (Ghimj[312]); 
# 1586
W_64 = (Ghimj[313]); 
# 1587
W_65 = (Ghimj[314]); 
# 1588
W_66 = (Ghimj[315]); 
# 1589
W_67 = (Ghimj[316]); 
# 1590
W_68 = (Ghimj[317]); 
# 1591
W_69 = (Ghimj[318]); 
# 1592
W_70 = (Ghimj[319]); 
# 1593
W_71 = (Ghimj[320]); 
# 1594
W_72 = (Ghimj[321]); 
# 1595
a = ((-W_32) / (Ghimj[119])); 
# 1596
W_32 = (-a); 
# 1597
W_65 = (W_65 + (a * (Ghimj[120]))); 
# 1598
a = ((-W_33) / (Ghimj[121])); 
# 1599
W_33 = (-a); 
# 1600
W_65 = (W_65 + (a * (Ghimj[122]))); 
# 1601
a = ((-W_34) / (Ghimj[123])); 
# 1602
W_34 = (-a); 
# 1603
W_65 = (W_65 + (a * (Ghimj[124]))); 
# 1604
a = ((-W_35) / (Ghimj[125])); 
# 1605
W_35 = (-a); 
# 1606
W_65 = (W_65 + (a * (Ghimj[126]))); 
# 1607
a = ((-W_36) / (Ghimj[127])); 
# 1608
W_36 = (-a); 
# 1609
W_65 = (W_65 + (a * (Ghimj[128]))); 
# 1610
a = ((-W_37) / (Ghimj[129])); 
# 1611
W_37 = (-a); 
# 1612
W_65 = (W_65 + (a * (Ghimj[130]))); 
# 1613
a = ((-W_39) / (Ghimj[135])); 
# 1614
W_39 = (-a); 
# 1615
W_65 = (W_65 + (a * (Ghimj[136]))); 
# 1616
a = ((-W_40) / (Ghimj[137])); 
# 1617
W_40 = (-a); 
# 1618
W_65 = (W_65 + (a * (Ghimj[138]))); 
# 1619
a = ((-W_41) / (Ghimj[139])); 
# 1620
W_41 = (-a); 
# 1621
W_47 = (W_47 + (a * (Ghimj[140]))); 
# 1622
a = ((-W_44) / (Ghimj[145])); 
# 1623
W_44 = (-a); 
# 1624
W_50 = (W_50 + (a * (Ghimj[146]))); 
# 1625
W_60 = (W_60 + (a * (Ghimj[147]))); 
# 1626
W_65 = (W_65 + (a * (Ghimj[148]))); 
# 1627
a = ((-W_47) / (Ghimj[165])); 
# 1628
W_47 = (-a); 
# 1629
W_52 = (W_52 + (a * (Ghimj[166]))); 
# 1630
W_57 = (W_57 + (a * (Ghimj[167]))); 
# 1631
W_58 = (W_58 + (a * (Ghimj[168]))); 
# 1632
W_65 = (W_65 + (a * (Ghimj[169]))); 
# 1633
W_71 = (W_71 + (a * (Ghimj[170]))); 
# 1634
a = ((-W_50) / (Ghimj[177])); 
# 1635
W_50 = (-a); 
# 1636
W_64 = (W_64 + (a * (Ghimj[178]))); 
# 1637
W_65 = (W_65 + (a * (Ghimj[179]))); 
# 1638
W_72 = (W_72 + (a * (Ghimj[180]))); 
# 1639
a = ((-W_51) / (Ghimj[181])); 
# 1640
W_51 = (-a); 
# 1641
W_67 = (W_67 + (a * (Ghimj[182]))); 
# 1642
W_68 = (W_68 + (a * (Ghimj[183]))); 
# 1643
a = ((-W_52) / (Ghimj[184])); 
# 1644
W_52 = (-a); 
# 1645
W_65 = (W_65 + (a * (Ghimj[185]))); 
# 1646
W_67 = (W_67 + (a * (Ghimj[186]))); 
# 1647
W_71 = (W_71 + (a * (Ghimj[187]))); 
# 1648
a = ((-W_57) / (Ghimj[218])); 
# 1649
W_57 = (-a); 
# 1650
W_62 = (W_62 + (a * (Ghimj[219]))); 
# 1651
W_64 = (W_64 + (a * (Ghimj[220]))); 
# 1652
W_65 = (W_65 + (a * (Ghimj[221]))); 
# 1653
W_67 = (W_67 + (a * (Ghimj[222]))); 
# 1654
W_68 = (W_68 + (a * (Ghimj[223]))); 
# 1655
W_71 = (W_71 + (a * (Ghimj[224]))); 
# 1656
a = ((-W_58) / (Ghimj[232])); 
# 1657
W_58 = (-a); 
# 1658
W_62 = (W_62 + (a * (Ghimj[233]))); 
# 1659
W_64 = (W_64 + (a * (Ghimj[234]))); 
# 1660
W_65 = (W_65 + (a * (Ghimj[235]))); 
# 1661
W_67 = (W_67 + (a * (Ghimj[236]))); 
# 1662
W_68 = (W_68 + (a * (Ghimj[237]))); 
# 1663
W_71 = (W_71 + (a * (Ghimj[238]))); 
# 1664
a = ((-W_59) / (Ghimj[240])); 
# 1665
W_59 = (-a); 
# 1666
W_61 = (W_61 + (a * (Ghimj[241]))); 
# 1667
W_63 = (W_63 + (a * (Ghimj[242]))); 
# 1668
W_65 = (W_65 + (a * (Ghimj[243]))); 
# 1669
W_67 = (W_67 + (a * (Ghimj[244]))); 
# 1670
W_68 = (W_68 + (a * (Ghimj[245]))); 
# 1671
W_69 = (W_69 + (a * (Ghimj[246]))); 
# 1672
W_72 = (W_72 + (a * (Ghimj[247]))); 
# 1673
a = ((-W_60) / (Ghimj[249])); 
# 1674
W_60 = (-a); 
# 1675
W_61 = (W_61 + (a * (Ghimj[250]))); 
# 1676
W_63 = (W_63 + (a * (Ghimj[251]))); 
# 1677
W_64 = (W_64 + (a * (Ghimj[252]))); 
# 1678
W_65 = (W_65 + (a * (Ghimj[253]))); 
# 1679
W_66 = (W_66 + (a * (Ghimj[254]))); 
# 1680
W_67 = (W_67 + (a * (Ghimj[255]))); 
# 1681
W_68 = (W_68 + (a * (Ghimj[256]))); 
# 1682
a = ((-W_61) / (Ghimj[258])); 
# 1683
W_61 = (-a); 
# 1684
W_63 = (W_63 + (a * (Ghimj[259]))); 
# 1685
W_68 = (W_68 + (a * (Ghimj[260]))); 
# 1686
W_69 = (W_69 + (a * (Ghimj[261]))); 
# 1687
W_72 = (W_72 + (a * (Ghimj[262]))); 
# 1688
a = ((-W_62) / (Ghimj[267])); 
# 1689
W_62 = (-a); 
# 1690
W_63 = (W_63 + (a * (Ghimj[268]))); 
# 1691
W_64 = (W_64 + (a * (Ghimj[269]))); 
# 1692
W_65 = (W_65 + (a * (Ghimj[270]))); 
# 1693
W_67 = (W_67 + (a * (Ghimj[271]))); 
# 1694
W_68 = (W_68 + (a * (Ghimj[272]))); 
# 1695
W_69 = (W_69 + (a * (Ghimj[273]))); 
# 1696
W_70 = (W_70 + (a * (Ghimj[274]))); 
# 1697
W_72 = (W_72 + (a * (Ghimj[275]))); 
# 1698
a = ((-W_63) / (Ghimj[278])); 
# 1699
W_63 = (-a); 
# 1700
W_67 = (W_67 + (a * (Ghimj[279]))); 
# 1701
W_68 = (W_68 + (a * (Ghimj[280]))); 
# 1702
W_69 = (W_69 + (a * (Ghimj[281]))); 
# 1703
W_72 = (W_72 + (a * (Ghimj[282]))); 
# 1704
a = ((-W_64) / (Ghimj[286])); 
# 1705
W_64 = (-a); 
# 1706
W_65 = (W_65 + (a * (Ghimj[287]))); 
# 1707
W_67 = (W_67 + (a * (Ghimj[288]))); 
# 1708
W_68 = (W_68 + (a * (Ghimj[289]))); 
# 1709
W_69 = (W_69 + (a * (Ghimj[290]))); 
# 1710
W_72 = (W_72 + (a * (Ghimj[291]))); 
# 1711
(Ghimj[292]) = W_32; 
# 1712
(Ghimj[293]) = W_33; 
# 1713
(Ghimj[294]) = W_34; 
# 1714
(Ghimj[295]) = W_35; 
# 1715
(Ghimj[296]) = W_36; 
# 1716
(Ghimj[297]) = W_37; 
# 1717
(Ghimj[298]) = W_39; 
# 1718
(Ghimj[299]) = W_40; 
# 1719
(Ghimj[300]) = W_41; 
# 1720
(Ghimj[301]) = W_44; 
# 1721
(Ghimj[302]) = W_47; 
# 1722
(Ghimj[303]) = W_50; 
# 1723
(Ghimj[304]) = W_51; 
# 1724
(Ghimj[305]) = W_52; 
# 1725
(Ghimj[306]) = W_57; 
# 1726
(Ghimj[307]) = W_58; 
# 1727
(Ghimj[308]) = W_59; 
# 1728
(Ghimj[309]) = W_60; 
# 1729
(Ghimj[310]) = W_61; 
# 1730
(Ghimj[311]) = W_62; 
# 1731
(Ghimj[312]) = W_63; 
# 1732
(Ghimj[313]) = W_64; 
# 1733
(Ghimj[314]) = W_65; 
# 1734
(Ghimj[315]) = W_66; 
# 1735
(Ghimj[316]) = W_67; 
# 1736
(Ghimj[317]) = W_68; 
# 1737
(Ghimj[318]) = W_69; 
# 1738
(Ghimj[319]) = W_70; 
# 1739
(Ghimj[320]) = W_71; 
# 1740
(Ghimj[321]) = W_72; 
# 1741
W_28 = (Ghimj[322]); 
# 1742
W_41 = (Ghimj[323]); 
# 1743
W_47 = (Ghimj[324]); 
# 1744
W_52 = (Ghimj[325]); 
# 1745
W_57 = (Ghimj[326]); 
# 1746
W_58 = (Ghimj[327]); 
# 1747
W_60 = (Ghimj[328]); 
# 1748
W_61 = (Ghimj[329]); 
# 1749
W_62 = (Ghimj[330]); 
# 1750
W_63 = (Ghimj[331]); 
# 1751
W_64 = (Ghimj[332]); 
# 1752
W_65 = (Ghimj[333]); 
# 1753
W_66 = (Ghimj[334]); 
# 1754
W_67 = (Ghimj[335]); 
# 1755
W_68 = (Ghimj[336]); 
# 1756
W_69 = (Ghimj[337]); 
# 1757
W_70 = (Ghimj[338]); 
# 1758
W_71 = (Ghimj[339]); 
# 1759
W_72 = (Ghimj[340]); 
# 1760
a = ((-W_28) / (Ghimj[110])); 
# 1761
W_28 = (-a); 
# 1762
W_41 = (W_41 + (a * (Ghimj[111]))); 
# 1763
a = ((-W_41) / (Ghimj[139])); 
# 1764
W_41 = (-a); 
# 1765
W_47 = (W_47 + (a * (Ghimj[140]))); 
# 1766
a = ((-W_47) / (Ghimj[165])); 
# 1767
W_47 = (-a); 
# 1768
W_52 = (W_52 + (a * (Ghimj[166]))); 
# 1769
W_57 = (W_57 + (a * (Ghimj[167]))); 
# 1770
W_58 = (W_58 + (a * (Ghimj[168]))); 
# 1771
W_65 = (W_65 + (a * (Ghimj[169]))); 
# 1772
W_71 = (W_71 + (a * (Ghimj[170]))); 
# 1773
a = ((-W_52) / (Ghimj[184])); 
# 1774
W_52 = (-a); 
# 1775
W_65 = (W_65 + (a * (Ghimj[185]))); 
# 1776
W_67 = (W_67 + (a * (Ghimj[186]))); 
# 1777
W_71 = (W_71 + (a * (Ghimj[187]))); 
# 1778
a = ((-W_57) / (Ghimj[218])); 
# 1779
W_57 = (-a); 
# 1780
W_62 = (W_62 + (a * (Ghimj[219]))); 
# 1781
W_64 = (W_64 + (a * (Ghimj[220]))); 
# 1782
W_65 = (W_65 + (a * (Ghimj[221]))); 
# 1783
W_67 = (W_67 + (a * (Ghimj[222]))); 
# 1784
W_68 = (W_68 + (a * (Ghimj[223]))); 
# 1785
W_71 = (W_71 + (a * (Ghimj[224]))); 
# 1786
a = ((-W_58) / (Ghimj[232])); 
# 1787
W_58 = (-a); 
# 1788
W_62 = (W_62 + (a * (Ghimj[233]))); 
# 1789
W_64 = (W_64 + (a * (Ghimj[234]))); 
# 1790
W_65 = (W_65 + (a * (Ghimj[235]))); 
# 1791
W_67 = (W_67 + (a * (Ghimj[236]))); 
# 1792
W_68 = (W_68 + (a * (Ghimj[237]))); 
# 1793
W_71 = (W_71 + (a * (Ghimj[238]))); 
# 1794
a = ((-W_60) / (Ghimj[249])); 
# 1795
W_60 = (-a); 
# 1796
W_61 = (W_61 + (a * (Ghimj[250]))); 
# 1797
W_63 = (W_63 + (a * (Ghimj[251]))); 
# 1798
W_64 = (W_64 + (a * (Ghimj[252]))); 
# 1799
W_65 = (W_65 + (a * (Ghimj[253]))); 
# 1800
W_66 = (W_66 + (a * (Ghimj[254]))); 
# 1801
W_67 = (W_67 + (a * (Ghimj[255]))); 
# 1802
W_68 = (W_68 + (a * (Ghimj[256]))); 
# 1803
a = ((-W_61) / (Ghimj[258])); 
# 1804
W_61 = (-a); 
# 1805
W_63 = (W_63 + (a * (Ghimj[259]))); 
# 1806
W_68 = (W_68 + (a * (Ghimj[260]))); 
# 1807
W_69 = (W_69 + (a * (Ghimj[261]))); 
# 1808
W_72 = (W_72 + (a * (Ghimj[262]))); 
# 1809
a = ((-W_62) / (Ghimj[267])); 
# 1810
W_62 = (-a); 
# 1811
W_63 = (W_63 + (a * (Ghimj[268]))); 
# 1812
W_64 = (W_64 + (a * (Ghimj[269]))); 
# 1813
W_65 = (W_65 + (a * (Ghimj[270]))); 
# 1814
W_67 = (W_67 + (a * (Ghimj[271]))); 
# 1815
W_68 = (W_68 + (a * (Ghimj[272]))); 
# 1816
W_69 = (W_69 + (a * (Ghimj[273]))); 
# 1817
W_70 = (W_70 + (a * (Ghimj[274]))); 
# 1818
W_72 = (W_72 + (a * (Ghimj[275]))); 
# 1819
a = ((-W_63) / (Ghimj[278])); 
# 1820
W_63 = (-a); 
# 1821
W_67 = (W_67 + (a * (Ghimj[279]))); 
# 1822
W_68 = (W_68 + (a * (Ghimj[280]))); 
# 1823
W_69 = (W_69 + (a * (Ghimj[281]))); 
# 1824
W_72 = (W_72 + (a * (Ghimj[282]))); 
# 1825
a = ((-W_64) / (Ghimj[286])); 
# 1826
W_64 = (-a); 
# 1827
W_65 = (W_65 + (a * (Ghimj[287]))); 
# 1828
W_67 = (W_67 + (a * (Ghimj[288]))); 
# 1829
W_68 = (W_68 + (a * (Ghimj[289]))); 
# 1830
W_69 = (W_69 + (a * (Ghimj[290]))); 
# 1831
W_72 = (W_72 + (a * (Ghimj[291]))); 
# 1832
a = ((-W_65) / (Ghimj[314])); 
# 1833
W_65 = (-a); 
# 1834
W_66 = (W_66 + (a * (Ghimj[315]))); 
# 1835
W_67 = (W_67 + (a * (Ghimj[316]))); 
# 1836
W_68 = (W_68 + (a * (Ghimj[317]))); 
# 1837
W_69 = (W_69 + (a * (Ghimj[318]))); 
# 1838
W_70 = (W_70 + (a * (Ghimj[319]))); 
# 1839
W_71 = (W_71 + (a * (Ghimj[320]))); 
# 1840
W_72 = (W_72 + (a * (Ghimj[321]))); 
# 1841
(Ghimj[322]) = W_28; 
# 1842
(Ghimj[323]) = W_41; 
# 1843
(Ghimj[324]) = W_47; 
# 1844
(Ghimj[325]) = W_52; 
# 1845
(Ghimj[326]) = W_57; 
# 1846
(Ghimj[327]) = W_58; 
# 1847
(Ghimj[328]) = W_60; 
# 1848
(Ghimj[329]) = W_61; 
# 1849
(Ghimj[330]) = W_62; 
# 1850
(Ghimj[331]) = W_63; 
# 1851
(Ghimj[332]) = W_64; 
# 1852
(Ghimj[333]) = W_65; 
# 1853
(Ghimj[334]) = W_66; 
# 1854
(Ghimj[335]) = W_67; 
# 1855
(Ghimj[336]) = W_68; 
# 1856
(Ghimj[337]) = W_69; 
# 1857
(Ghimj[338]) = W_70; 
# 1858
(Ghimj[339]) = W_71; 
# 1859
(Ghimj[340]) = W_72; 
# 1860
W_39 = (Ghimj[341]); 
# 1861
W_42 = (Ghimj[342]); 
# 1862
W_43 = (Ghimj[343]); 
# 1863
W_45 = (Ghimj[344]); 
# 1864
W_46 = (Ghimj[345]); 
# 1865
W_48 = (Ghimj[346]); 
# 1866
W_49 = (Ghimj[347]); 
# 1867
W_51 = (Ghimj[348]); 
# 1868
W_52 = (Ghimj[349]); 
# 1869
W_54 = (Ghimj[350]); 
# 1870
W_56 = (Ghimj[351]); 
# 1871
W_58 = (Ghimj[352]); 
# 1872
W_59 = (Ghimj[353]); 
# 1873
W_61 = (Ghimj[354]); 
# 1874
W_62 = (Ghimj[355]); 
# 1875
W_63 = (Ghimj[356]); 
# 1876
W_64 = (Ghimj[357]); 
# 1877
W_65 = (Ghimj[358]); 
# 1878
W_66 = (Ghimj[359]); 
# 1879
W_67 = (Ghimj[360]); 
# 1880
W_68 = (Ghimj[361]); 
# 1881
W_69 = (Ghimj[362]); 
# 1882
W_70 = (Ghimj[363]); 
# 1883
W_71 = (Ghimj[364]); 
# 1884
W_72 = (Ghimj[365]); 
# 1885
a = ((-W_39) / (Ghimj[135])); 
# 1886
W_39 = (-a); 
# 1887
W_65 = (W_65 + (a * (Ghimj[136]))); 
# 1888
a = ((-W_42) / (Ghimj[141])); 
# 1889
W_42 = (-a); 
# 1890
W_67 = (W_67 + (a * (Ghimj[142]))); 
# 1891
a = ((-W_43) / (Ghimj[143])); 
# 1892
W_43 = (-a); 
# 1893
W_67 = (W_67 + (a * (Ghimj[144]))); 
# 1894
a = ((-W_45) / (Ghimj[149])); 
# 1895
W_45 = (-a); 
# 1896
W_67 = (W_67 + (a * (Ghimj[150]))); 
# 1897
W_68 = (W_68 + (a * (Ghimj[151]))); 
# 1898
a = ((-W_46) / (Ghimj[152])); 
# 1899
W_46 = (-a); 
# 1900
W_67 = (W_67 + (a * (Ghimj[153]))); 
# 1901
W_68 = (W_68 + (a * (Ghimj[154]))); 
# 1902
a = ((-W_48) / (Ghimj[171])); 
# 1903
W_48 = (-a); 
# 1904
W_69 = (W_69 + (a * (Ghimj[172]))); 
# 1905
W_72 = (W_72 + (a * (Ghimj[173]))); 
# 1906
a = ((-W_49) / (Ghimj[174])); 
# 1907
W_49 = (-a); 
# 1908
W_67 = (W_67 + (a * (Ghimj[175]))); 
# 1909
W_68 = (W_68 + (a * (Ghimj[176]))); 
# 1910
a = ((-W_51) / (Ghimj[181])); 
# 1911
W_51 = (-a); 
# 1912
W_67 = (W_67 + (a * (Ghimj[182]))); 
# 1913
W_68 = (W_68 + (a * (Ghimj[183]))); 
# 1914
a = ((-W_52) / (Ghimj[184])); 
# 1915
W_52 = (-a); 
# 1916
W_65 = (W_65 + (a * (Ghimj[185]))); 
# 1917
W_67 = (W_67 + (a * (Ghimj[186]))); 
# 1918
W_71 = (W_71 + (a * (Ghimj[187]))); 
# 1919
a = ((-W_54) / (Ghimj[194])); 
# 1920
W_54 = (-a); 
# 1921
W_56 = (W_56 + (a * (Ghimj[195]))); 
# 1922
W_63 = (W_63 + (a * (Ghimj[196]))); 
# 1923
W_67 = (W_67 + (a * (Ghimj[197]))); 
# 1924
W_68 = (W_68 + (a * (Ghimj[198]))); 
# 1925
W_69 = (W_69 + (a * (Ghimj[199]))); 
# 1926
a = ((-W_56) / (Ghimj[207])); 
# 1927
W_56 = (-a); 
# 1928
W_59 = (W_59 + (a * (Ghimj[208]))); 
# 1929
W_63 = (W_63 + (a * (Ghimj[209]))); 
# 1930
W_67 = (W_67 + (a * (Ghimj[210]))); 
# 1931
W_68 = (W_68 + (a * (Ghimj[211]))); 
# 1932
W_69 = (W_69 + (a * (Ghimj[212]))); 
# 1933
W_72 = (W_72 + (a * (Ghimj[213]))); 
# 1934
a = ((-W_58) / (Ghimj[232])); 
# 1935
W_58 = (-a); 
# 1936
W_62 = (W_62 + (a * (Ghimj[233]))); 
# 1937
W_64 = (W_64 + (a * (Ghimj[234]))); 
# 1938
W_65 = (W_65 + (a * (Ghimj[235]))); 
# 1939
W_67 = (W_67 + (a * (Ghimj[236]))); 
# 1940
W_68 = (W_68 + (a * (Ghimj[237]))); 
# 1941
W_71 = (W_71 + (a * (Ghimj[238]))); 
# 1942
a = ((-W_59) / (Ghimj[240])); 
# 1943
W_59 = (-a); 
# 1944
W_61 = (W_61 + (a * (Ghimj[241]))); 
# 1945
W_63 = (W_63 + (a * (Ghimj[242]))); 
# 1946
W_65 = (W_65 + (a * (Ghimj[243]))); 
# 1947
W_67 = (W_67 + (a * (Ghimj[244]))); 
# 1948
W_68 = (W_68 + (a * (Ghimj[245]))); 
# 1949
W_69 = (W_69 + (a * (Ghimj[246]))); 
# 1950
W_72 = (W_72 + (a * (Ghimj[247]))); 
# 1951
a = ((-W_61) / (Ghimj[258])); 
# 1952
W_61 = (-a); 
# 1953
W_63 = (W_63 + (a * (Ghimj[259]))); 
# 1954
W_68 = (W_68 + (a * (Ghimj[260]))); 
# 1955
W_69 = (W_69 + (a * (Ghimj[261]))); 
# 1956
W_72 = (W_72 + (a * (Ghimj[262]))); 
# 1957
a = ((-W_62) / (Ghimj[267])); 
# 1958
W_62 = (-a); 
# 1959
W_63 = (W_63 + (a * (Ghimj[268]))); 
# 1960
W_64 = (W_64 + (a * (Ghimj[269]))); 
# 1961
W_65 = (W_65 + (a * (Ghimj[270]))); 
# 1962
W_67 = (W_67 + (a * (Ghimj[271]))); 
# 1963
W_68 = (W_68 + (a * (Ghimj[272]))); 
# 1964
W_69 = (W_69 + (a * (Ghimj[273]))); 
# 1965
W_70 = (W_70 + (a * (Ghimj[274]))); 
# 1966
W_72 = (W_72 + (a * (Ghimj[275]))); 
# 1967
a = ((-W_63) / (Ghimj[278])); 
# 1968
W_63 = (-a); 
# 1969
W_67 = (W_67 + (a * (Ghimj[279]))); 
# 1970
W_68 = (W_68 + (a * (Ghimj[280]))); 
# 1971
W_69 = (W_69 + (a * (Ghimj[281]))); 
# 1972
W_72 = (W_72 + (a * (Ghimj[282]))); 
# 1973
a = ((-W_64) / (Ghimj[286])); 
# 1974
W_64 = (-a); 
# 1975
W_65 = (W_65 + (a * (Ghimj[287]))); 
# 1976
W_67 = (W_67 + (a * (Ghimj[288]))); 
# 1977
W_68 = (W_68 + (a * (Ghimj[289]))); 
# 1978
W_69 = (W_69 + (a * (Ghimj[290]))); 
# 1979
W_72 = (W_72 + (a * (Ghimj[291]))); 
# 1980
a = ((-W_65) / (Ghimj[314])); 
# 1981
W_65 = (-a); 
# 1982
W_66 = (W_66 + (a * (Ghimj[315]))); 
# 1983
W_67 = (W_67 + (a * (Ghimj[316]))); 
# 1984
W_68 = (W_68 + (a * (Ghimj[317]))); 
# 1985
W_69 = (W_69 + (a * (Ghimj[318]))); 
# 1986
W_70 = (W_70 + (a * (Ghimj[319]))); 
# 1987
W_71 = (W_71 + (a * (Ghimj[320]))); 
# 1988
W_72 = (W_72 + (a * (Ghimj[321]))); 
# 1989
a = ((-W_66) / (Ghimj[334])); 
# 1990
W_66 = (-a); 
# 1991
W_67 = (W_67 + (a * (Ghimj[335]))); 
# 1992
W_68 = (W_68 + (a * (Ghimj[336]))); 
# 1993
W_69 = (W_69 + (a * (Ghimj[337]))); 
# 1994
W_70 = (W_70 + (a * (Ghimj[338]))); 
# 1995
W_71 = (W_71 + (a * (Ghimj[339]))); 
# 1996
W_72 = (W_72 + (a * (Ghimj[340]))); 
# 1997
(Ghimj[341]) = W_39; 
# 1998
(Ghimj[342]) = W_42; 
# 1999
(Ghimj[343]) = W_43; 
# 2000
(Ghimj[344]) = W_45; 
# 2001
(Ghimj[345]) = W_46; 
# 2002
(Ghimj[346]) = W_48; 
# 2003
(Ghimj[347]) = W_49; 
# 2004
(Ghimj[348]) = W_51; 
# 2005
(Ghimj[349]) = W_52; 
# 2006
(Ghimj[350]) = W_54; 
# 2007
(Ghimj[351]) = W_56; 
# 2008
(Ghimj[352]) = W_58; 
# 2009
(Ghimj[353]) = W_59; 
# 2010
(Ghimj[354]) = W_61; 
# 2011
(Ghimj[355]) = W_62; 
# 2012
(Ghimj[356]) = W_63; 
# 2013
(Ghimj[357]) = W_64; 
# 2014
(Ghimj[358]) = W_65; 
# 2015
(Ghimj[359]) = W_66; 
# 2016
(Ghimj[360]) = W_67; 
# 2017
(Ghimj[361]) = W_68; 
# 2018
(Ghimj[362]) = W_69; 
# 2019
(Ghimj[363]) = W_70; 
# 2020
(Ghimj[364]) = W_71; 
# 2021
(Ghimj[365]) = W_72; 
# 2022
W_32 = (Ghimj[366]); 
# 2023
W_33 = (Ghimj[367]); 
# 2024
W_34 = (Ghimj[368]); 
# 2025
W_35 = (Ghimj[369]); 
# 2026
W_36 = (Ghimj[370]); 
# 2027
W_37 = (Ghimj[371]); 
# 2028
W_45 = (Ghimj[372]); 
# 2029
W_46 = (Ghimj[373]); 
# 2030
W_48 = (Ghimj[374]); 
# 2031
W_49 = (Ghimj[375]); 
# 2032
W_51 = (Ghimj[376]); 
# 2033
W_54 = (Ghimj[377]); 
# 2034
W_56 = (Ghimj[378]); 
# 2035
W_57 = (Ghimj[379]); 
# 2036
W_59 = (Ghimj[380]); 
# 2037
W_61 = (Ghimj[381]); 
# 2038
W_62 = (Ghimj[382]); 
# 2039
W_63 = (Ghimj[383]); 
# 2040
W_64 = (Ghimj[384]); 
# 2041
W_65 = (Ghimj[385]); 
# 2042
W_66 = (Ghimj[386]); 
# 2043
W_67 = (Ghimj[387]); 
# 2044
W_68 = (Ghimj[388]); 
# 2045
W_69 = (Ghimj[389]); 
# 2046
W_70 = (Ghimj[390]); 
# 2047
W_71 = (Ghimj[391]); 
# 2048
W_72 = (Ghimj[392]); 
# 2049
a = ((-W_32) / (Ghimj[119])); 
# 2050
W_32 = (-a); 
# 2051
W_65 = (W_65 + (a * (Ghimj[120]))); 
# 2052
a = ((-W_33) / (Ghimj[121])); 
# 2053
W_33 = (-a); 
# 2054
W_65 = (W_65 + (a * (Ghimj[122]))); 
# 2055
a = ((-W_34) / (Ghimj[123])); 
# 2056
W_34 = (-a); 
# 2057
W_65 = (W_65 + (a * (Ghimj[124]))); 
# 2058
a = ((-W_35) / (Ghimj[125])); 
# 2059
W_35 = (-a); 
# 2060
W_65 = (W_65 + (a * (Ghimj[126]))); 
# 2061
a = ((-W_36) / (Ghimj[127])); 
# 2062
W_36 = (-a); 
# 2063
W_65 = (W_65 + (a * (Ghimj[128]))); 
# 2064
a = ((-W_37) / (Ghimj[129])); 
# 2065
W_37 = (-a); 
# 2066
W_65 = (W_65 + (a * (Ghimj[130]))); 
# 2067
a = ((-W_45) / (Ghimj[149])); 
# 2068
W_45 = (-a); 
# 2069
W_67 = (W_67 + (a * (Ghimj[150]))); 
# 2070
W_68 = (W_68 + (a * (Ghimj[151]))); 
# 2071
a = ((-W_46) / (Ghimj[152])); 
# 2072
W_46 = (-a); 
# 2073
W_67 = (W_67 + (a * (Ghimj[153]))); 
# 2074
W_68 = (W_68 + (a * (Ghimj[154]))); 
# 2075
a = ((-W_48) / (Ghimj[171])); 
# 2076
W_48 = (-a); 
# 2077
W_69 = (W_69 + (a * (Ghimj[172]))); 
# 2078
W_72 = (W_72 + (a * (Ghimj[173]))); 
# 2079
a = ((-W_49) / (Ghimj[174])); 
# 2080
W_49 = (-a); 
# 2081
W_67 = (W_67 + (a * (Ghimj[175]))); 
# 2082
W_68 = (W_68 + (a * (Ghimj[176]))); 
# 2083
a = ((-W_51) / (Ghimj[181])); 
# 2084
W_51 = (-a); 
# 2085
W_67 = (W_67 + (a * (Ghimj[182]))); 
# 2086
W_68 = (W_68 + (a * (Ghimj[183]))); 
# 2087
a = ((-W_54) / (Ghimj[194])); 
# 2088
W_54 = (-a); 
# 2089
W_56 = (W_56 + (a * (Ghimj[195]))); 
# 2090
W_63 = (W_63 + (a * (Ghimj[196]))); 
# 2091
W_67 = (W_67 + (a * (Ghimj[197]))); 
# 2092
W_68 = (W_68 + (a * (Ghimj[198]))); 
# 2093
W_69 = (W_69 + (a * (Ghimj[199]))); 
# 2094
a = ((-W_56) / (Ghimj[207])); 
# 2095
W_56 = (-a); 
# 2096
W_59 = (W_59 + (a * (Ghimj[208]))); 
# 2097
W_63 = (W_63 + (a * (Ghimj[209]))); 
# 2098
W_67 = (W_67 + (a * (Ghimj[210]))); 
# 2099
W_68 = (W_68 + (a * (Ghimj[211]))); 
# 2100
W_69 = (W_69 + (a * (Ghimj[212]))); 
# 2101
W_72 = (W_72 + (a * (Ghimj[213]))); 
# 2102
a = ((-W_57) / (Ghimj[218])); 
# 2103
W_57 = (-a); 
# 2104
W_62 = (W_62 + (a * (Ghimj[219]))); 
# 2105
W_64 = (W_64 + (a * (Ghimj[220]))); 
# 2106
W_65 = (W_65 + (a * (Ghimj[221]))); 
# 2107
W_67 = (W_67 + (a * (Ghimj[222]))); 
# 2108
W_68 = (W_68 + (a * (Ghimj[223]))); 
# 2109
W_71 = (W_71 + (a * (Ghimj[224]))); 
# 2110
a = ((-W_59) / (Ghimj[240])); 
# 2111
W_59 = (-a); 
# 2112
W_61 = (W_61 + (a * (Ghimj[241]))); 
# 2113
W_63 = (W_63 + (a * (Ghimj[242]))); 
# 2114
W_65 = (W_65 + (a * (Ghimj[243]))); 
# 2115
W_67 = (W_67 + (a * (Ghimj[244]))); 
# 2116
W_68 = (W_68 + (a * (Ghimj[245]))); 
# 2117
W_69 = (W_69 + (a * (Ghimj[246]))); 
# 2118
W_72 = (W_72 + (a * (Ghimj[247]))); 
# 2119
a = ((-W_61) / (Ghimj[258])); 
# 2120
W_61 = (-a); 
# 2121
W_63 = (W_63 + (a * (Ghimj[259]))); 
# 2122
W_68 = (W_68 + (a * (Ghimj[260]))); 
# 2123
W_69 = (W_69 + (a * (Ghimj[261]))); 
# 2124
W_72 = (W_72 + (a * (Ghimj[262]))); 
# 2125
a = ((-W_62) / (Ghimj[267])); 
# 2126
W_62 = (-a); 
# 2127
W_63 = (W_63 + (a * (Ghimj[268]))); 
# 2128
W_64 = (W_64 + (a * (Ghimj[269]))); 
# 2129
W_65 = (W_65 + (a * (Ghimj[270]))); 
# 2130
W_67 = (W_67 + (a * (Ghimj[271]))); 
# 2131
W_68 = (W_68 + (a * (Ghimj[272]))); 
# 2132
W_69 = (W_69 + (a * (Ghimj[273]))); 
# 2133
W_70 = (W_70 + (a * (Ghimj[274]))); 
# 2134
W_72 = (W_72 + (a * (Ghimj[275]))); 
# 2135
a = ((-W_63) / (Ghimj[278])); 
# 2136
W_63 = (-a); 
# 2137
W_67 = (W_67 + (a * (Ghimj[279]))); 
# 2138
W_68 = (W_68 + (a * (Ghimj[280]))); 
# 2139
W_69 = (W_69 + (a * (Ghimj[281]))); 
# 2140
W_72 = (W_72 + (a * (Ghimj[282]))); 
# 2141
a = ((-W_64) / (Ghimj[286])); 
# 2142
W_64 = (-a); 
# 2143
W_65 = (W_65 + (a * (Ghimj[287]))); 
# 2144
W_67 = (W_67 + (a * (Ghimj[288]))); 
# 2145
W_68 = (W_68 + (a * (Ghimj[289]))); 
# 2146
W_69 = (W_69 + (a * (Ghimj[290]))); 
# 2147
W_72 = (W_72 + (a * (Ghimj[291]))); 
# 2148
a = ((-W_65) / (Ghimj[314])); 
# 2149
W_65 = (-a); 
# 2150
W_66 = (W_66 + (a * (Ghimj[315]))); 
# 2151
W_67 = (W_67 + (a * (Ghimj[316]))); 
# 2152
W_68 = (W_68 + (a * (Ghimj[317]))); 
# 2153
W_69 = (W_69 + (a * (Ghimj[318]))); 
# 2154
W_70 = (W_70 + (a * (Ghimj[319]))); 
# 2155
W_71 = (W_71 + (a * (Ghimj[320]))); 
# 2156
W_72 = (W_72 + (a * (Ghimj[321]))); 
# 2157
a = ((-W_66) / (Ghimj[334])); 
# 2158
W_66 = (-a); 
# 2159
W_67 = (W_67 + (a * (Ghimj[335]))); 
# 2160
W_68 = (W_68 + (a * (Ghimj[336]))); 
# 2161
W_69 = (W_69 + (a * (Ghimj[337]))); 
# 2162
W_70 = (W_70 + (a * (Ghimj[338]))); 
# 2163
W_71 = (W_71 + (a * (Ghimj[339]))); 
# 2164
W_72 = (W_72 + (a * (Ghimj[340]))); 
# 2165
a = ((-W_67) / (Ghimj[360])); 
# 2166
W_67 = (-a); 
# 2167
W_68 = (W_68 + (a * (Ghimj[361]))); 
# 2168
W_69 = (W_69 + (a * (Ghimj[362]))); 
# 2169
W_70 = (W_70 + (a * (Ghimj[363]))); 
# 2170
W_71 = (W_71 + (a * (Ghimj[364]))); 
# 2171
W_72 = (W_72 + (a * (Ghimj[365]))); 
# 2172
(Ghimj[366]) = W_32; 
# 2173
(Ghimj[367]) = W_33; 
# 2174
(Ghimj[368]) = W_34; 
# 2175
(Ghimj[369]) = W_35; 
# 2176
(Ghimj[370]) = W_36; 
# 2177
(Ghimj[371]) = W_37; 
# 2178
(Ghimj[372]) = W_45; 
# 2179
(Ghimj[373]) = W_46; 
# 2180
(Ghimj[374]) = W_48; 
# 2181
(Ghimj[375]) = W_49; 
# 2182
(Ghimj[376]) = W_51; 
# 2183
(Ghimj[377]) = W_54; 
# 2184
(Ghimj[378]) = W_56; 
# 2185
(Ghimj[379]) = W_57; 
# 2186
(Ghimj[380]) = W_59; 
# 2187
(Ghimj[381]) = W_61; 
# 2188
(Ghimj[382]) = W_62; 
# 2189
(Ghimj[383]) = W_63; 
# 2190
(Ghimj[384]) = W_64; 
# 2191
(Ghimj[385]) = W_65; 
# 2192
(Ghimj[386]) = W_66; 
# 2193
(Ghimj[387]) = W_67; 
# 2194
(Ghimj[388]) = W_68; 
# 2195
(Ghimj[389]) = W_69; 
# 2196
(Ghimj[390]) = W_70; 
# 2197
(Ghimj[391]) = W_71; 
# 2198
(Ghimj[392]) = W_72; 
# 2199
W_30 = (Ghimj[393]); 
# 2200
W_48 = (Ghimj[394]); 
# 2201
W_55 = (Ghimj[395]); 
# 2202
W_61 = (Ghimj[396]); 
# 2203
W_63 = (Ghimj[397]); 
# 2204
W_66 = (Ghimj[398]); 
# 2205
W_67 = (Ghimj[399]); 
# 2206
W_68 = (Ghimj[400]); 
# 2207
W_69 = (Ghimj[401]); 
# 2208
W_70 = (Ghimj[402]); 
# 2209
W_71 = (Ghimj[403]); 
# 2210
W_72 = (Ghimj[404]); 
# 2211
a = ((-W_30) / (Ghimj[114])); 
# 2212
W_30 = (-a); 
# 2213
W_69 = (W_69 + (a * (Ghimj[115]))); 
# 2214
a = ((-W_48) / (Ghimj[171])); 
# 2215
W_48 = (-a); 
# 2216
W_69 = (W_69 + (a * (Ghimj[172]))); 
# 2217
W_72 = (W_72 + (a * (Ghimj[173]))); 
# 2218
a = ((-W_55) / (Ghimj[201])); 
# 2219
W_55 = (-a); 
# 2220
W_61 = (W_61 + (a * (Ghimj[202]))); 
# 2221
W_63 = (W_63 + (a * (Ghimj[203]))); 
# 2222
W_69 = (W_69 + (a * (Ghimj[204]))); 
# 2223
W_72 = (W_72 + (a * (Ghimj[205]))); 
# 2224
a = ((-W_61) / (Ghimj[258])); 
# 2225
W_61 = (-a); 
# 2226
W_63 = (W_63 + (a * (Ghimj[259]))); 
# 2227
W_68 = (W_68 + (a * (Ghimj[260]))); 
# 2228
W_69 = (W_69 + (a * (Ghimj[261]))); 
# 2229
W_72 = (W_72 + (a * (Ghimj[262]))); 
# 2230
a = ((-W_63) / (Ghimj[278])); 
# 2231
W_63 = (-a); 
# 2232
W_67 = (W_67 + (a * (Ghimj[279]))); 
# 2233
W_68 = (W_68 + (a * (Ghimj[280]))); 
# 2234
W_69 = (W_69 + (a * (Ghimj[281]))); 
# 2235
W_72 = (W_72 + (a * (Ghimj[282]))); 
# 2236
a = ((-W_66) / (Ghimj[334])); 
# 2237
W_66 = (-a); 
# 2238
W_67 = (W_67 + (a * (Ghimj[335]))); 
# 2239
W_68 = (W_68 + (a * (Ghimj[336]))); 
# 2240
W_69 = (W_69 + (a * (Ghimj[337]))); 
# 2241
W_70 = (W_70 + (a * (Ghimj[338]))); 
# 2242
W_71 = (W_71 + (a * (Ghimj[339]))); 
# 2243
W_72 = (W_72 + (a * (Ghimj[340]))); 
# 2244
a = ((-W_67) / (Ghimj[360])); 
# 2245
W_67 = (-a); 
# 2246
W_68 = (W_68 + (a * (Ghimj[361]))); 
# 2247
W_69 = (W_69 + (a * (Ghimj[362]))); 
# 2248
W_70 = (W_70 + (a * (Ghimj[363]))); 
# 2249
W_71 = (W_71 + (a * (Ghimj[364]))); 
# 2250
W_72 = (W_72 + (a * (Ghimj[365]))); 
# 2251
a = ((-W_68) / (Ghimj[388])); 
# 2252
W_68 = (-a); 
# 2253
W_69 = (W_69 + (a * (Ghimj[389]))); 
# 2254
W_70 = (W_70 + (a * (Ghimj[390]))); 
# 2255
W_71 = (W_71 + (a * (Ghimj[391]))); 
# 2256
W_72 = (W_72 + (a * (Ghimj[392]))); 
# 2257
(Ghimj[393]) = W_30; 
# 2258
(Ghimj[394]) = W_48; 
# 2259
(Ghimj[395]) = W_55; 
# 2260
(Ghimj[396]) = W_61; 
# 2261
(Ghimj[397]) = W_63; 
# 2262
(Ghimj[398]) = W_66; 
# 2263
(Ghimj[399]) = W_67; 
# 2264
(Ghimj[400]) = W_68; 
# 2265
(Ghimj[401]) = W_69; 
# 2266
(Ghimj[402]) = W_70; 
# 2267
(Ghimj[403]) = W_71; 
# 2268
(Ghimj[404]) = W_72; 
# 2269
W_42 = (Ghimj[405]); 
# 2270
W_43 = (Ghimj[406]); 
# 2271
W_49 = (Ghimj[407]); 
# 2272
W_50 = (Ghimj[408]); 
# 2273
W_51 = (Ghimj[409]); 
# 2274
W_60 = (Ghimj[410]); 
# 2275
W_61 = (Ghimj[411]); 
# 2276
W_63 = (Ghimj[412]); 
# 2277
W_64 = (Ghimj[413]); 
# 2278
W_65 = (Ghimj[414]); 
# 2279
W_66 = (Ghimj[415]); 
# 2280
W_67 = (Ghimj[416]); 
# 2281
W_68 = (Ghimj[417]); 
# 2282
W_69 = (Ghimj[418]); 
# 2283
W_70 = (Ghimj[419]); 
# 2284
W_71 = (Ghimj[420]); 
# 2285
W_72 = (Ghimj[421]); 
# 2286
a = ((-W_42) / (Ghimj[141])); 
# 2287
W_42 = (-a); 
# 2288
W_67 = (W_67 + (a * (Ghimj[142]))); 
# 2289
a = ((-W_43) / (Ghimj[143])); 
# 2290
W_43 = (-a); 
# 2291
W_67 = (W_67 + (a * (Ghimj[144]))); 
# 2292
a = ((-W_49) / (Ghimj[174])); 
# 2293
W_49 = (-a); 
# 2294
W_67 = (W_67 + (a * (Ghimj[175]))); 
# 2295
W_68 = (W_68 + (a * (Ghimj[176]))); 
# 2296
a = ((-W_50) / (Ghimj[177])); 
# 2297
W_50 = (-a); 
# 2298
W_64 = (W_64 + (a * (Ghimj[178]))); 
# 2299
W_65 = (W_65 + (a * (Ghimj[179]))); 
# 2300
W_72 = (W_72 + (a * (Ghimj[180]))); 
# 2301
a = ((-W_51) / (Ghimj[181])); 
# 2302
W_51 = (-a); 
# 2303
W_67 = (W_67 + (a * (Ghimj[182]))); 
# 2304
W_68 = (W_68 + (a * (Ghimj[183]))); 
# 2305
a = ((-W_60) / (Ghimj[249])); 
# 2306
W_60 = (-a); 
# 2307
W_61 = (W_61 + (a * (Ghimj[250]))); 
# 2308
W_63 = (W_63 + (a * (Ghimj[251]))); 
# 2309
W_64 = (W_64 + (a * (Ghimj[252]))); 
# 2310
W_65 = (W_65 + (a * (Ghimj[253]))); 
# 2311
W_66 = (W_66 + (a * (Ghimj[254]))); 
# 2312
W_67 = (W_67 + (a * (Ghimj[255]))); 
# 2313
W_68 = (W_68 + (a * (Ghimj[256]))); 
# 2314
a = ((-W_61) / (Ghimj[258])); 
# 2315
W_61 = (-a); 
# 2316
W_63 = (W_63 + (a * (Ghimj[259]))); 
# 2317
W_68 = (W_68 + (a * (Ghimj[260]))); 
# 2318
W_69 = (W_69 + (a * (Ghimj[261]))); 
# 2319
W_72 = (W_72 + (a * (Ghimj[262]))); 
# 2320
a = ((-W_63) / (Ghimj[278])); 
# 2321
W_63 = (-a); 
# 2322
W_67 = (W_67 + (a * (Ghimj[279]))); 
# 2323
W_68 = (W_68 + (a * (Ghimj[280]))); 
# 2324
W_69 = (W_69 + (a * (Ghimj[281]))); 
# 2325
W_72 = (W_72 + (a * (Ghimj[282]))); 
# 2326
a = ((-W_64) / (Ghimj[286])); 
# 2327
W_64 = (-a); 
# 2328
W_65 = (W_65 + (a * (Ghimj[287]))); 
# 2329
W_67 = (W_67 + (a * (Ghimj[288]))); 
# 2330
W_68 = (W_68 + (a * (Ghimj[289]))); 
# 2331
W_69 = (W_69 + (a * (Ghimj[290]))); 
# 2332
W_72 = (W_72 + (a * (Ghimj[291]))); 
# 2333
a = ((-W_65) / (Ghimj[314])); 
# 2334
W_65 = (-a); 
# 2335
W_66 = (W_66 + (a * (Ghimj[315]))); 
# 2336
W_67 = (W_67 + (a * (Ghimj[316]))); 
# 2337
W_68 = (W_68 + (a * (Ghimj[317]))); 
# 2338
W_69 = (W_69 + (a * (Ghimj[318]))); 
# 2339
W_70 = (W_70 + (a * (Ghimj[319]))); 
# 2340
W_71 = (W_71 + (a * (Ghimj[320]))); 
# 2341
W_72 = (W_72 + (a * (Ghimj[321]))); 
# 2342
a = ((-W_66) / (Ghimj[334])); 
# 2343
W_66 = (-a); 
# 2344
W_67 = (W_67 + (a * (Ghimj[335]))); 
# 2345
W_68 = (W_68 + (a * (Ghimj[336]))); 
# 2346
W_69 = (W_69 + (a * (Ghimj[337]))); 
# 2347
W_70 = (W_70 + (a * (Ghimj[338]))); 
# 2348
W_71 = (W_71 + (a * (Ghimj[339]))); 
# 2349
W_72 = (W_72 + (a * (Ghimj[340]))); 
# 2350
a = ((-W_67) / (Ghimj[360])); 
# 2351
W_67 = (-a); 
# 2352
W_68 = (W_68 + (a * (Ghimj[361]))); 
# 2353
W_69 = (W_69 + (a * (Ghimj[362]))); 
# 2354
W_70 = (W_70 + (a * (Ghimj[363]))); 
# 2355
W_71 = (W_71 + (a * (Ghimj[364]))); 
# 2356
W_72 = (W_72 + (a * (Ghimj[365]))); 
# 2357
a = ((-W_68) / (Ghimj[388])); 
# 2358
W_68 = (-a); 
# 2359
W_69 = (W_69 + (a * (Ghimj[389]))); 
# 2360
W_70 = (W_70 + (a * (Ghimj[390]))); 
# 2361
W_71 = (W_71 + (a * (Ghimj[391]))); 
# 2362
W_72 = (W_72 + (a * (Ghimj[392]))); 
# 2363
a = ((-W_69) / (Ghimj[401])); 
# 2364
W_69 = (-a); 
# 2365
W_70 = (W_70 + (a * (Ghimj[402]))); 
# 2366
W_71 = (W_71 + (a * (Ghimj[403]))); 
# 2367
W_72 = (W_72 + (a * (Ghimj[404]))); 
# 2368
(Ghimj[405]) = W_42; 
# 2369
(Ghimj[406]) = W_43; 
# 2370
(Ghimj[407]) = W_49; 
# 2371
(Ghimj[408]) = W_50; 
# 2372
(Ghimj[409]) = W_51; 
# 2373
(Ghimj[410]) = W_60; 
# 2374
(Ghimj[411]) = W_61; 
# 2375
(Ghimj[412]) = W_63; 
# 2376
(Ghimj[413]) = W_64; 
# 2377
(Ghimj[414]) = W_65; 
# 2378
(Ghimj[415]) = W_66; 
# 2379
(Ghimj[416]) = W_67; 
# 2380
(Ghimj[417]) = W_68; 
# 2381
(Ghimj[418]) = W_69; 
# 2382
(Ghimj[419]) = W_70; 
# 2383
(Ghimj[420]) = W_71; 
# 2384
(Ghimj[421]) = W_72; 
# 2385
W_31 = (Ghimj[422]); 
# 2386
W_38 = (Ghimj[423]); 
# 2387
W_40 = (Ghimj[424]); 
# 2388
W_44 = (Ghimj[425]); 
# 2389
W_50 = (Ghimj[426]); 
# 2390
W_52 = (Ghimj[427]); 
# 2391
W_60 = (Ghimj[428]); 
# 2392
W_61 = (Ghimj[429]); 
# 2393
W_62 = (Ghimj[430]); 
# 2394
W_63 = (Ghimj[431]); 
# 2395
W_64 = (Ghimj[432]); 
# 2396
W_65 = (Ghimj[433]); 
# 2397
W_66 = (Ghimj[434]); 
# 2398
W_67 = (Ghimj[435]); 
# 2399
W_68 = (Ghimj[436]); 
# 2400
W_69 = (Ghimj[437]); 
# 2401
W_70 = (Ghimj[438]); 
# 2402
W_71 = (Ghimj[439]); 
# 2403
W_72 = (Ghimj[440]); 
# 2404
a = ((-W_31) / (Ghimj[116])); 
# 2405
W_31 = (-a); 
# 2406
W_40 = (W_40 + (a * (Ghimj[117]))); 
# 2407
W_65 = (W_65 + (a * (Ghimj[118]))); 
# 2408
a = ((-W_38) / (Ghimj[131])); 
# 2409
W_38 = (-a); 
# 2410
W_60 = (W_60 + (a * (Ghimj[132]))); 
# 2411
W_66 = (W_66 + (a * (Ghimj[133]))); 
# 2412
W_71 = (W_71 + (a * (Ghimj[134]))); 
# 2413
a = ((-W_40) / (Ghimj[137])); 
# 2414
W_40 = (-a); 
# 2415
W_65 = (W_65 + (a * (Ghimj[138]))); 
# 2416
a = ((-W_44) / (Ghimj[145])); 
# 2417
W_44 = (-a); 
# 2418
W_50 = (W_50 + (a * (Ghimj[146]))); 
# 2419
W_60 = (W_60 + (a * (Ghimj[147]))); 
# 2420
W_65 = (W_65 + (a * (Ghimj[148]))); 
# 2421
a = ((-W_50) / (Ghimj[177])); 
# 2422
W_50 = (-a); 
# 2423
W_64 = (W_64 + (a * (Ghimj[178]))); 
# 2424
W_65 = (W_65 + (a * (Ghimj[179]))); 
# 2425
W_72 = (W_72 + (a * (Ghimj[180]))); 
# 2426
a = ((-W_52) / (Ghimj[184])); 
# 2427
W_52 = (-a); 
# 2428
W_65 = (W_65 + (a * (Ghimj[185]))); 
# 2429
W_67 = (W_67 + (a * (Ghimj[186]))); 
# 2430
W_71 = (W_71 + (a * (Ghimj[187]))); 
# 2431
a = ((-W_60) / (Ghimj[249])); 
# 2432
W_60 = (-a); 
# 2433
W_61 = (W_61 + (a * (Ghimj[250]))); 
# 2434
W_63 = (W_63 + (a * (Ghimj[251]))); 
# 2435
W_64 = (W_64 + (a * (Ghimj[252]))); 
# 2436
W_65 = (W_65 + (a * (Ghimj[253]))); 
# 2437
W_66 = (W_66 + (a * (Ghimj[254]))); 
# 2438
W_67 = (W_67 + (a * (Ghimj[255]))); 
# 2439
W_68 = (W_68 + (a * (Ghimj[256]))); 
# 2440
a = ((-W_61) / (Ghimj[258])); 
# 2441
W_61 = (-a); 
# 2442
W_63 = (W_63 + (a * (Ghimj[259]))); 
# 2443
W_68 = (W_68 + (a * (Ghimj[260]))); 
# 2444
W_69 = (W_69 + (a * (Ghimj[261]))); 
# 2445
W_72 = (W_72 + (a * (Ghimj[262]))); 
# 2446
a = ((-W_62) / (Ghimj[267])); 
# 2447
W_62 = (-a); 
# 2448
W_63 = (W_63 + (a * (Ghimj[268]))); 
# 2449
W_64 = (W_64 + (a * (Ghimj[269]))); 
# 2450
W_65 = (W_65 + (a * (Ghimj[270]))); 
# 2451
W_67 = (W_67 + (a * (Ghimj[271]))); 
# 2452
W_68 = (W_68 + (a * (Ghimj[272]))); 
# 2453
W_69 = (W_69 + (a * (Ghimj[273]))); 
# 2454
W_70 = (W_70 + (a * (Ghimj[274]))); 
# 2455
W_72 = (W_72 + (a * (Ghimj[275]))); 
# 2456
a = ((-W_63) / (Ghimj[278])); 
# 2457
W_63 = (-a); 
# 2458
W_67 = (W_67 + (a * (Ghimj[279]))); 
# 2459
W_68 = (W_68 + (a * (Ghimj[280]))); 
# 2460
W_69 = (W_69 + (a * (Ghimj[281]))); 
# 2461
W_72 = (W_72 + (a * (Ghimj[282]))); 
# 2462
a = ((-W_64) / (Ghimj[286])); 
# 2463
W_64 = (-a); 
# 2464
W_65 = (W_65 + (a * (Ghimj[287]))); 
# 2465
W_67 = (W_67 + (a * (Ghimj[288]))); 
# 2466
W_68 = (W_68 + (a * (Ghimj[289]))); 
# 2467
W_69 = (W_69 + (a * (Ghimj[290]))); 
# 2468
W_72 = (W_72 + (a * (Ghimj[291]))); 
# 2469
a = ((-W_65) / (Ghimj[314])); 
# 2470
W_65 = (-a); 
# 2471
W_66 = (W_66 + (a * (Ghimj[315]))); 
# 2472
W_67 = (W_67 + (a * (Ghimj[316]))); 
# 2473
W_68 = (W_68 + (a * (Ghimj[317]))); 
# 2474
W_69 = (W_69 + (a * (Ghimj[318]))); 
# 2475
W_70 = (W_70 + (a * (Ghimj[319]))); 
# 2476
W_71 = (W_71 + (a * (Ghimj[320]))); 
# 2477
W_72 = (W_72 + (a * (Ghimj[321]))); 
# 2478
a = ((-W_66) / (Ghimj[334])); 
# 2479
W_66 = (-a); 
# 2480
W_67 = (W_67 + (a * (Ghimj[335]))); 
# 2481
W_68 = (W_68 + (a * (Ghimj[336]))); 
# 2482
W_69 = (W_69 + (a * (Ghimj[337]))); 
# 2483
W_70 = (W_70 + (a * (Ghimj[338]))); 
# 2484
W_71 = (W_71 + (a * (Ghimj[339]))); 
# 2485
W_72 = (W_72 + (a * (Ghimj[340]))); 
# 2486
a = ((-W_67) / (Ghimj[360])); 
# 2487
W_67 = (-a); 
# 2488
W_68 = (W_68 + (a * (Ghimj[361]))); 
# 2489
W_69 = (W_69 + (a * (Ghimj[362]))); 
# 2490
W_70 = (W_70 + (a * (Ghimj[363]))); 
# 2491
W_71 = (W_71 + (a * (Ghimj[364]))); 
# 2492
W_72 = (W_72 + (a * (Ghimj[365]))); 
# 2493
a = ((-W_68) / (Ghimj[388])); 
# 2494
W_68 = (-a); 
# 2495
W_69 = (W_69 + (a * (Ghimj[389]))); 
# 2496
W_70 = (W_70 + (a * (Ghimj[390]))); 
# 2497
W_71 = (W_71 + (a * (Ghimj[391]))); 
# 2498
W_72 = (W_72 + (a * (Ghimj[392]))); 
# 2499
a = ((-W_69) / (Ghimj[401])); 
# 2500
W_69 = (-a); 
# 2501
W_70 = (W_70 + (a * (Ghimj[402]))); 
# 2502
W_71 = (W_71 + (a * (Ghimj[403]))); 
# 2503
W_72 = (W_72 + (a * (Ghimj[404]))); 
# 2504
a = ((-W_70) / (Ghimj[419])); 
# 2505
W_70 = (-a); 
# 2506
W_71 = (W_71 + (a * (Ghimj[420]))); 
# 2507
W_72 = (W_72 + (a * (Ghimj[421]))); 
# 2508
(Ghimj[422]) = W_31; 
# 2509
(Ghimj[423]) = W_38; 
# 2510
(Ghimj[424]) = W_40; 
# 2511
(Ghimj[425]) = W_44; 
# 2512
(Ghimj[426]) = W_50; 
# 2513
(Ghimj[427]) = W_52; 
# 2514
(Ghimj[428]) = W_60; 
# 2515
(Ghimj[429]) = W_61; 
# 2516
(Ghimj[430]) = W_62; 
# 2517
(Ghimj[431]) = W_63; 
# 2518
(Ghimj[432]) = W_64; 
# 2519
(Ghimj[433]) = W_65; 
# 2520
(Ghimj[434]) = W_66; 
# 2521
(Ghimj[435]) = W_67; 
# 2522
(Ghimj[436]) = W_68; 
# 2523
(Ghimj[437]) = W_69; 
# 2524
(Ghimj[438]) = W_70; 
# 2525
(Ghimj[439]) = W_71; 
# 2526
(Ghimj[440]) = W_72; 
# 2527
W_48 = (Ghimj[441]); 
# 2528
W_55 = (Ghimj[442]); 
# 2529
W_61 = (Ghimj[443]); 
# 2530
W_63 = (Ghimj[444]); 
# 2531
W_64 = (Ghimj[445]); 
# 2532
W_65 = (Ghimj[446]); 
# 2533
W_66 = (Ghimj[447]); 
# 2534
W_67 = (Ghimj[448]); 
# 2535
W_68 = (Ghimj[449]); 
# 2536
W_69 = (Ghimj[450]); 
# 2537
W_70 = (Ghimj[451]); 
# 2538
W_71 = (Ghimj[452]); 
# 2539
W_72 = (Ghimj[453]); 
# 2540
a = ((-W_48) / (Ghimj[171])); 
# 2541
W_48 = (-a); 
# 2542
W_69 = (W_69 + (a * (Ghimj[172]))); 
# 2543
W_72 = (W_72 + (a * (Ghimj[173]))); 
# 2544
a = ((-W_55) / (Ghimj[201])); 
# 2545
W_55 = (-a); 
# 2546
W_61 = (W_61 + (a * (Ghimj[202]))); 
# 2547
W_63 = (W_63 + (a * (Ghimj[203]))); 
# 2548
W_69 = (W_69 + (a * (Ghimj[204]))); 
# 2549
W_72 = (W_72 + (a * (Ghimj[205]))); 
# 2550
a = ((-W_61) / (Ghimj[258])); 
# 2551
W_61 = (-a); 
# 2552
W_63 = (W_63 + (a * (Ghimj[259]))); 
# 2553
W_68 = (W_68 + (a * (Ghimj[260]))); 
# 2554
W_69 = (W_69 + (a * (Ghimj[261]))); 
# 2555
W_72 = (W_72 + (a * (Ghimj[262]))); 
# 2556
a = ((-W_63) / (Ghimj[278])); 
# 2557
W_63 = (-a); 
# 2558
W_67 = (W_67 + (a * (Ghimj[279]))); 
# 2559
W_68 = (W_68 + (a * (Ghimj[280]))); 
# 2560
W_69 = (W_69 + (a * (Ghimj[281]))); 
# 2561
W_72 = (W_72 + (a * (Ghimj[282]))); 
# 2562
a = ((-W_64) / (Ghimj[286])); 
# 2563
W_64 = (-a); 
# 2564
W_65 = (W_65 + (a * (Ghimj[287]))); 
# 2565
W_67 = (W_67 + (a * (Ghimj[288]))); 
# 2566
W_68 = (W_68 + (a * (Ghimj[289]))); 
# 2567
W_69 = (W_69 + (a * (Ghimj[290]))); 
# 2568
W_72 = (W_72 + (a * (Ghimj[291]))); 
# 2569
a = ((-W_65) / (Ghimj[314])); 
# 2570
W_65 = (-a); 
# 2571
W_66 = (W_66 + (a * (Ghimj[315]))); 
# 2572
W_67 = (W_67 + (a * (Ghimj[316]))); 
# 2573
W_68 = (W_68 + (a * (Ghimj[317]))); 
# 2574
W_69 = (W_69 + (a * (Ghimj[318]))); 
# 2575
W_70 = (W_70 + (a * (Ghimj[319]))); 
# 2576
W_71 = (W_71 + (a * (Ghimj[320]))); 
# 2577
W_72 = (W_72 + (a * (Ghimj[321]))); 
# 2578
a = ((-W_66) / (Ghimj[334])); 
# 2579
W_66 = (-a); 
# 2580
W_67 = (W_67 + (a * (Ghimj[335]))); 
# 2581
W_68 = (W_68 + (a * (Ghimj[336]))); 
# 2582
W_69 = (W_69 + (a * (Ghimj[337]))); 
# 2583
W_70 = (W_70 + (a * (Ghimj[338]))); 
# 2584
W_71 = (W_71 + (a * (Ghimj[339]))); 
# 2585
W_72 = (W_72 + (a * (Ghimj[340]))); 
# 2586
a = ((-W_67) / (Ghimj[360])); 
# 2587
W_67 = (-a); 
# 2588
W_68 = (W_68 + (a * (Ghimj[361]))); 
# 2589
W_69 = (W_69 + (a * (Ghimj[362]))); 
# 2590
W_70 = (W_70 + (a * (Ghimj[363]))); 
# 2591
W_71 = (W_71 + (a * (Ghimj[364]))); 
# 2592
W_72 = (W_72 + (a * (Ghimj[365]))); 
# 2593
a = ((-W_68) / (Ghimj[388])); 
# 2594
W_68 = (-a); 
# 2595
W_69 = (W_69 + (a * (Ghimj[389]))); 
# 2596
W_70 = (W_70 + (a * (Ghimj[390]))); 
# 2597
W_71 = (W_71 + (a * (Ghimj[391]))); 
# 2598
W_72 = (W_72 + (a * (Ghimj[392]))); 
# 2599
a = ((-W_69) / (Ghimj[401])); 
# 2600
W_69 = (-a); 
# 2601
W_70 = (W_70 + (a * (Ghimj[402]))); 
# 2602
W_71 = (W_71 + (a * (Ghimj[403]))); 
# 2603
W_72 = (W_72 + (a * (Ghimj[404]))); 
# 2604
a = ((-W_70) / (Ghimj[419])); 
# 2605
W_70 = (-a); 
# 2606
W_71 = (W_71 + (a * (Ghimj[420]))); 
# 2607
W_72 = (W_72 + (a * (Ghimj[421]))); 
# 2608
a = ((-W_71) / (Ghimj[439])); 
# 2609
W_71 = (-a); 
# 2610
W_72 = (W_72 + (a * (Ghimj[440]))); 
# 2611
(Ghimj[441]) = W_48; 
# 2612
(Ghimj[442]) = W_55; 
# 2613
(Ghimj[443]) = W_61; 
# 2614
(Ghimj[444]) = W_63; 
# 2615
(Ghimj[445]) = W_64; 
# 2616
(Ghimj[446]) = W_65; 
# 2617
(Ghimj[447]) = W_66; 
# 2618
(Ghimj[448]) = W_67; 
# 2619
(Ghimj[449]) = W_68; 
# 2620
(Ghimj[450]) = W_69; 
# 2621
(Ghimj[451]) = W_70; 
# 2622
(Ghimj[452]) = W_71; 
# 2623
(Ghimj[453]) = W_72; 
# 2624
} 
#endif
# 2626 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) void ros_Decomp(double *__restrict__ Ghimj, int &Ndec, int VL_GLO) 
# 2627
{int volatile ___ = 1;(void)Ghimj;(void)Ndec;(void)VL_GLO;
# 2630
::exit(___);}
#if 0
# 2627
{ 
# 2628
kppDecomp(Ghimj, VL_GLO); 
# 2629
Ndec++; 
# 2630
} 
#endif
# 2633 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) void ros_PrepareMatrix(double &H, int direction, double gam, double *jac0, double *Ghimj, int &Nsng, int &Ndec, int VL_GLO) 
# 2634
{int volatile ___ = 1;(void)H;(void)direction;(void)gam;(void)jac0;(void)Ghimj;(void)Nsng;(void)Ndec;(void)VL_GLO;
# 2718
::exit(___);}
#if 0
# 2634
{ 
# 2635
int index = ((__device_builtin_variable_blockIdx.x) * (__device_builtin_variable_blockDim.x)) + (__device_builtin_variable_threadIdx.x); 
# 2636
int ising, nConsecutive; 
# 2637
double ghinv; 
# 2639
ghinv = ((1.0) / ((direction * H) * gam)); 
# 2640
for (int i = 0; i < 455; i++) { 
# 2641
(Ghimj[i]) = (-(jac0[i])); }  
# 2643
(Ghimj[0]) += ghinv; 
# 2644
(Ghimj[1]) += ghinv; 
# 2645
(Ghimj[4]) += ghinv; 
# 2646
(Ghimj[8]) += ghinv; 
# 2647
(Ghimj[12]) += ghinv; 
# 2648
(Ghimj[15]) += ghinv; 
# 2649
(Ghimj[18]) += ghinv; 
# 2650
(Ghimj[24]) += ghinv; 
# 2651
(Ghimj[27]) += ghinv; 
# 2652
(Ghimj[33]) += ghinv; 
# 2653
(Ghimj[35]) += ghinv; 
# 2654
(Ghimj[36]) += ghinv; 
# 2655
(Ghimj[40]) += ghinv; 
# 2656
(Ghimj[43]) += ghinv; 
# 2657
(Ghimj[46]) += ghinv; 
# 2658
(Ghimj[48]) += ghinv; 
# 2659
(Ghimj[51]) += ghinv; 
# 2660
(Ghimj[55]) += ghinv; 
# 2661
(Ghimj[68]) += ghinv; 
# 2662
(Ghimj[73]) += ghinv; 
# 2663
(Ghimj[78]) += ghinv; 
# 2664
(Ghimj[82]) += ghinv; 
# 2665
(Ghimj[86]) += ghinv; 
# 2666
(Ghimj[89]) += ghinv; 
# 2667
(Ghimj[92]) += ghinv; 
# 2668
(Ghimj[95]) += ghinv; 
# 2669
(Ghimj[98]) += ghinv; 
# 2670
(Ghimj[103]) += ghinv; 
# 2671
(Ghimj[110]) += ghinv; 
# 2672
(Ghimj[112]) += ghinv; 
# 2673
(Ghimj[114]) += ghinv; 
# 2674
(Ghimj[116]) += ghinv; 
# 2675
(Ghimj[119]) += ghinv; 
# 2676
(Ghimj[121]) += ghinv; 
# 2677
(Ghimj[123]) += ghinv; 
# 2678
(Ghimj[125]) += ghinv; 
# 2679
(Ghimj[127]) += ghinv; 
# 2680
(Ghimj[129]) += ghinv; 
# 2681
(Ghimj[131]) += ghinv; 
# 2682
(Ghimj[135]) += ghinv; 
# 2683
(Ghimj[137]) += ghinv; 
# 2684
(Ghimj[139]) += ghinv; 
# 2685
(Ghimj[141]) += ghinv; 
# 2686
(Ghimj[143]) += ghinv; 
# 2687
(Ghimj[145]) += ghinv; 
# 2688
(Ghimj[149]) += ghinv; 
# 2689
(Ghimj[152]) += ghinv; 
# 2690
(Ghimj[165]) += ghinv; 
# 2691
(Ghimj[171]) += ghinv; 
# 2692
(Ghimj[174]) += ghinv; 
# 2693
(Ghimj[177]) += ghinv; 
# 2694
(Ghimj[181]) += ghinv; 
# 2695
(Ghimj[184]) += ghinv; 
# 2696
(Ghimj[188]) += ghinv; 
# 2697
(Ghimj[194]) += ghinv; 
# 2698
(Ghimj[201]) += ghinv; 
# 2699
(Ghimj[207]) += ghinv; 
# 2700
(Ghimj[218]) += ghinv; 
# 2701
(Ghimj[232]) += ghinv; 
# 2702
(Ghimj[240]) += ghinv; 
# 2703
(Ghimj[249]) += ghinv; 
# 2704
(Ghimj[258]) += ghinv; 
# 2705
(Ghimj[267]) += ghinv; 
# 2706
(Ghimj[278]) += ghinv; 
# 2707
(Ghimj[286]) += ghinv; 
# 2708
(Ghimj[314]) += ghinv; 
# 2709
(Ghimj[334]) += ghinv; 
# 2710
(Ghimj[360]) += ghinv; 
# 2711
(Ghimj[388]) += ghinv; 
# 2712
(Ghimj[401]) += ghinv; 
# 2713
(Ghimj[419]) += ghinv; 
# 2714
(Ghimj[439]) += ghinv; 
# 2715
(Ghimj[453]) += ghinv; 
# 2716
(Ghimj[454]) += ghinv; 
# 2717
ros_Decomp(Ghimj, Ndec, VL_GLO); 
# 2718
} 
#endif
# 2720 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) void Jac_sp(const double *__restrict__ var, const double *__restrict__ fix, const double *__restrict__ 
# 2721
rconst, double *__restrict__ jcb, int &Njac, const int VL_GLO) 
# 2722
{int volatile ___ = 1;(void)var;(void)fix;(void)rconst;(void)jcb;(void)Njac;(void)VL_GLO;
# 3321
::exit(___);}
#if 0
# 2722
{ 
# 2723
int index = ((__device_builtin_variable_blockIdx.x) * (__device_builtin_variable_blockDim.x)) + (__device_builtin_variable_threadIdx.x); 
# 2725
double dummy, B_0, B_1, B_2, B_3, B_4, B_5, B_6, B_7, B_8, B_9, B_10, B_11, B_12, B_13, B_14, B_15, B_16, B_17, B_18, B_19, B_20, B_21, B_22, B_23, B_24, B_25, B_26, B_27, B_28, B_29, B_30, B_31, B_32, B_33, B_34, B_35, B_36, B_37, B_38, B_39, B_40, B_41, B_42, B_43, B_44, B_45, B_46, B_47, B_48, B_49, B_50, B_51, B_52, B_53, B_54, B_55, B_56, B_57, B_58, B_59, B_60, B_61, B_62, B_63, B_64, B_65, B_66, B_67, B_68, B_69, B_70, B_71, B_72, B_73, B_74, B_75, B_76, B_77, B_78, B_79, B_80, B_81, B_82, B_83, B_84, B_85, B_86, B_87, B_88, B_89, B_90, B_91, B_92, B_93, B_94, B_95, B_96, B_97, B_98, B_99, B_100, B_101, B_102, B_103, B_104, B_105, B_106, B_107, B_108, B_109, B_110, B_111, B_112, B_113, B_114, B_115, B_116, B_117, B_118, B_119, B_120, B_121, B_122, B_123, B_124, B_125, B_126, B_127, B_128, B_129, B_130, B_131, B_132, B_133, B_134, B_135, B_136, B_137, B_138, B_139; 
# 2728
Njac++; 
# 2730
B_0 = ((rconst[0]) * (fix[0])); 
# 2731
B_2 = ((rconst[1]) * (fix[0])); 
# 2732
B_4 = ((rconst[2]) * (fix[0])); 
# 2733
B_6 = ((rconst[3]) * (var[66])); 
# 2734
B_7 = ((rconst[3]) * (var[65])); 
# 2735
B_8 = ((rconst[4]) * (var[65])); 
# 2736
B_9 = ((rconst[4]) * (var[40])); 
# 2737
B_10 = ((rconst[5]) * (var[71])); 
# 2738
B_11 = ((rconst[5]) * (var[66])); 
# 2739
B_12 = ((rconst[6]) * (var[71])); 
# 2740
B_13 = ((rconst[6]) * (var[65])); 
# 2741
B_14 = (((rconst[7]) * (2)) * (var[71])); 
# 2742
B_15 = ((rconst[8]) * (var[47])); 
# 2743
B_16 = ((rconst[8]) * (var[41])); 
# 2744
B_17 = ((1.800000000000000004e-12) * (var[65])); 
# 2745
B_18 = ((1.800000000000000004e-12) * (var[52])); 
# 2746
B_19 = (((rconst[10]) * (2)) * (var[47])); 
# 2747
B_20 = (1000000.0); 
# 2748
B_21 = ((rconst[12]) * (var[67])); 
# 2749
B_22 = ((rconst[12]) * (var[66])); 
# 2750
B_23 = (((rconst[13]) * (2)) * (var[69])); 
# 2751
B_24 = (((rconst[14]) * (2)) * (var[69])); 
# 2752
B_25 = (((rconst[15]) * (2)) * (var[69])); 
# 2753
B_26 = (((rconst[16]) * (2)) * (var[69])); 
# 2754
B_27 = (rconst[17]); 
# 2755
B_28 = ((rconst[18]) * (var[67])); 
# 2756
B_29 = ((rconst[18]) * (var[52])); 
# 2757
B_30 = ((rconst[19]) * (var[71])); 
# 2758
B_31 = ((rconst[19]) * (var[69])); 
# 2759
B_32 = ((rconst[20]) * (var[65])); 
# 2760
B_33 = ((rconst[20]) * (var[58])); 
# 2761
B_34 = ((rconst[21]) * (var[69])); 
# 2762
B_35 = ((rconst[21]) * (var[48])); 
# 2763
B_36 = ((rconst[22]) * (var[69])); 
# 2764
B_37 = ((rconst[22]) * (var[55])); 
# 2765
B_38 = (rconst[23]); 
# 2766
B_39 = ((rconst[24]) * (var[67])); 
# 2767
B_40 = ((rconst[24]) * (var[63])); 
# 2768
B_41 = ((rconst[25]) * (var[67])); 
# 2769
B_42 = ((rconst[25]) * (var[42])); 
# 2770
B_43 = ((rconst[26]) * (var[67])); 
# 2771
B_44 = ((rconst[26]) * (var[62])); 
# 2772
B_45 = ((5.900000000000000305e-11) * (var[67])); 
# 2773
B_46 = ((5.900000000000000305e-11) * (var[51])); 
# 2774
B_47 = ((rconst[28]) * (var[70])); 
# 2775
B_48 = ((rconst[28]) * (var[69])); 
# 2776
B_49 = ((rconst[29]) * (var[65])); 
# 2777
B_50 = ((rconst[29]) * (var[39])); 
# 2778
B_51 = ((rconst[30]) * (var[67])); 
# 2779
B_52 = ((rconst[30]) * (var[45])); 
# 2780
B_53 = ((7.999999999999999516e-11) * (var[67])); 
# 2781
B_54 = ((7.999999999999999516e-11) * (var[46])); 
# 2782
B_55 = ((rconst[32]) * (var[67])); 
# 2783
B_56 = ((rconst[32]) * (var[49])); 
# 2784
B_57 = ((rconst[33]) * (var[67])); 
# 2785
B_58 = ((rconst[33]) * (var[43])); 
# 2786
B_59 = ((rconst[34]) * (var[68])); 
# 2787
B_60 = ((rconst[34]) * (var[66])); 
# 2788
B_61 = (((2.699999999999999804e-12) * (2)) * (var[72])); 
# 2789
B_62 = (((rconst[36]) * (2)) * (var[72])); 
# 2790
B_63 = ((rconst[37]) * (var[71])); 
# 2791
B_64 = ((rconst[37]) * (var[68])); 
# 2792
B_65 = ((rconst[38]) * (var[72])); 
# 2793
B_66 = ((rconst[38]) * (var[71])); 
# 2794
B_67 = ((rconst[39]) * (var[65])); 
# 2795
B_68 = ((rconst[39]) * (var[57])); 
# 2796
B_69 = ((rconst[40]) * (var[65])); 
# 2797
B_70 = ((rconst[40]) * (var[59])); 
# 2798
B_71 = ((4.899999999999999881e-11) * (var[68])); 
# 2799
B_72 = ((4.899999999999999881e-11) * (var[61])); 
# 2800
B_73 = ((rconst[42]) * (var[72])); 
# 2801
B_74 = ((rconst[42]) * (var[48])); 
# 2802
B_75 = ((rconst[43]) * (var[72])); 
# 2803
B_76 = ((rconst[43]) * (var[55])); 
# 2804
B_77 = (rconst[44]); 
# 2805
B_78 = ((rconst[45]) * (var[68])); 
# 2806
B_79 = ((rconst[45]) * (var[62])); 
# 2807
B_80 = ((rconst[46]) * (var[68])); 
# 2808
B_81 = ((rconst[46]) * (var[51])); 
# 2809
B_82 = ((rconst[47]) * (var[72])); 
# 2810
B_83 = ((rconst[47]) * (var[70])); 
# 2811
B_84 = ((rconst[48]) * (var[72])); 
# 2812
B_85 = ((rconst[48]) * (var[70])); 
# 2813
B_86 = ((rconst[49]) * (var[65])); 
# 2814
B_87 = ((rconst[49]) * (var[32])); 
# 2815
B_88 = ((rconst[50]) * (var[68])); 
# 2816
B_89 = ((rconst[50]) * (var[45])); 
# 2817
B_90 = ((rconst[51]) * (var[68])); 
# 2818
B_91 = ((rconst[51]) * (var[46])); 
# 2819
B_92 = ((rconst[52]) * (var[68])); 
# 2820
B_93 = ((rconst[52]) * (var[49])); 
# 2821
B_94 = ((rconst[53]) * (var[65])); 
# 2822
B_95 = ((rconst[53]) * (var[37])); 
# 2823
B_96 = ((rconst[54]) * (var[65])); 
# 2824
B_97 = ((rconst[54]) * (var[36])); 
# 2825
B_98 = ((3.319999999999999911e-15) * (var[68])); 
# 2826
B_99 = ((3.319999999999999911e-15) * (var[56])); 
# 2827
B_100 = ((1.099999999999999928e-15) * (var[68])); 
# 2828
B_101 = ((1.099999999999999928e-15) * (var[54])); 
# 2829
B_102 = ((rconst[57]) * (var[67])); 
# 2830
B_103 = ((rconst[57]) * (var[59])); 
# 2831
B_104 = ((rconst[58]) * (var[72])); 
# 2832
B_105 = ((rconst[58]) * (var[69])); 
# 2833
B_106 = ((rconst[59]) * (var[72])); 
# 2834
B_107 = ((rconst[59]) * (var[69])); 
# 2835
B_108 = ((rconst[60]) * (var[72])); 
# 2836
B_109 = ((rconst[60]) * (var[69])); 
# 2837
B_110 = ((1.450000000000000001e-11) * (var[67])); 
# 2838
B_111 = ((1.450000000000000001e-11) * (var[56])); 
# 2839
B_112 = ((rconst[62]) * (var[65])); 
# 2840
B_113 = ((rconst[62]) * (var[33])); 
# 2841
B_114 = ((rconst[63]) * (var[65])); 
# 2842
B_115 = ((rconst[63]) * (var[34])); 
# 2843
B_116 = ((rconst[64]) * (var[65])); 
# 2844
B_117 = ((rconst[64]) * (var[35])); 
# 2845
B_118 = ((rconst[65]) * (var[65])); 
# 2846
B_119 = ((rconst[65]) * (var[44])); 
# 2847
B_120 = ((rconst[66]) * (var[65])); 
# 2848
B_121 = ((rconst[66]) * (var[64])); 
# 2849
B_122 = ((rconst[67]) * (var[65])); 
# 2850
B_123 = ((rconst[67]) * (var[64])); 
# 2851
B_124 = ((rconst[68]) * (var[64])); 
# 2852
B_125 = ((rconst[68]) * (var[53])); 
# 2853
B_126 = ((1.000000000000000036e-10) * (var[65])); 
# 2854
B_127 = ((1.000000000000000036e-10) * (var[50])); 
# 2855
B_128 = (rconst[70]); 
# 2856
B_129 = ((2.999999999999999839e-13) * (var[66])); 
# 2857
B_130 = ((2.999999999999999839e-13) * (var[60])); 
# 2858
B_131 = ((5.000000000000000182e-11) * (var[71])); 
# 2859
B_132 = ((5.000000000000000182e-11) * (var[38])); 
# 2860
B_133 = ((3.299999999999999978e-10) * (var[67])); 
# 2861
B_134 = ((3.299999999999999978e-10) * (var[64])); 
# 2862
B_135 = ((rconst[74]) * (var[68])); 
# 2863
B_136 = ((rconst[74]) * (var[64])); 
# 2864
B_137 = ((4.399999999999999932e-13) * (var[72])); 
# 2865
B_138 = ((4.399999999999999932e-13) * (var[64])); 
# 2866
B_139 = (rconst[76]); 
# 2867
(jcb[0]) = (-B_139); 
# 2868
(jcb[1]) = (0); 
# 2869
(jcb[2]) = B_124; 
# 2870
(jcb[3]) = B_125; 
# 2871
(jcb[4]) = (0); 
# 2872
(jcb[5]) = (B_43 + B_78); 
# 2873
(jcb[6]) = B_44; 
# 2874
(jcb[7]) = B_79; 
# 2875
(jcb[8]) = (0); 
# 2876
(jcb[9]) = (B_53 + B_90); 
# 2877
(jcb[10]) = B_54; 
# 2878
(jcb[11]) = B_91; 
# 2879
(jcb[12]) = (0); 
# 2880
(jcb[13]) = B_30; 
# 2881
(jcb[14]) = B_31; 
# 2882
(jcb[15]) = (0); 
# 2883
(jcb[16]) = (B_25 + B_104); 
# 2884
(jcb[17]) = B_105; 
# 2885
(jcb[18]) = (0); 
# 2886
(jcb[19]) = B_69; 
# 2887
(jcb[20]) = B_70; 
# 2888
(jcb[21]) = B_82; 
# 2889
(jcb[22]) = B_65; 
# 2890
(jcb[23]) = (B_66 + B_83); 
# 2891
(jcb[24]) = (0); 
# 2892
(jcb[25]) = B_118; 
# 2893
(jcb[26]) = B_119; 
# 2894
(jcb[27]) = (0); 
# 2895
(jcb[28]) = B_131; 
# 2896
(jcb[29]) = ((0.4000000000000000222) * B_126); 
# 2897
(jcb[30]) = ((0.4000000000000000222) * B_127); 
# 2898
(jcb[31]) = B_132; 
# 2899
(jcb[32]) = ((2) * B_139); 
# 2900
(jcb[33]) = (0); 
# 2901
(jcb[34]) = ((2) * B_139); 
# 2902
(jcb[35]) = (0); 
# 2903
(jcb[36]) = (0); 
# 2904
(jcb[37]) = (((0.6666670000000000096) * B_51) + ((0.6666670000000000096) * B_88)); 
# 2905
(jcb[38]) = ((0.6666670000000000096) * B_52); 
# 2906
(jcb[39]) = ((0.6666670000000000096) * B_89); 
# 2907
(jcb[40]) = (0); 
# 2908
(jcb[41]) = B_6; 
# 2909
(jcb[42]) = B_7; 
# 2910
(jcb[43]) = (0); 
# 2911
(jcb[44]) = B_10; 
# 2912
(jcb[45]) = B_11; 
# 2913
(jcb[46]) = (0); 
# 2914
(jcb[47]) = B_14; 
# 2915
(jcb[48]) = (0); 
# 2916
(jcb[49]) = B_10; 
# 2917
(jcb[50]) = B_11; 
# 2918
(jcb[51]) = (0); 
# 2919
(jcb[52]) = B_15; 
# 2920
(jcb[53]) = B_16; 
# 2921
(jcb[54]) = ((3) * B_139); 
# 2922
(jcb[55]) = (0); 
# 2923
(jcb[56]) = B_15; 
# 2924
(jcb[57]) = B_118; 
# 2925
(jcb[58]) = B_16; 
# 2926
(jcb[59]) = B_124; 
# 2927
(jcb[60]) = B_129; 
# 2928
(jcb[61]) = (B_125 + B_137); 
# 2929
(jcb[62]) = (B_6 + B_119); 
# 2930
(jcb[63]) = ((B_7 + B_10) + B_130); 
# 2931
(jcb[64]) = ((((((((2) * B_23) + ((2) * B_24)) + B_25) + B_47) + B_104) + ((2) * B_106)) + ((2) * B_108)); 
# 2932
(jcb[65]) = (B_48 + B_84); 
# 2933
(jcb[66]) = B_11; 
# 2934
(jcb[67]) = ((((((((2) * B_61) + ((2) * B_62)) + B_85) + B_105) + ((2) * B_107)) + ((2) * B_109)) + B_138); 
# 2935
(jcb[68]) = (0); 
# 2936
(jcb[69]) = B_73; 
# 2937
(jcb[70]) = (B_106 + B_108); 
# 2938
(jcb[71]) = B_84; 
# 2939
(jcb[72]) = (((((((2) * B_61) + ((2) * B_62)) + B_74) + B_85) + B_107) + B_109); 
# 2940
(jcb[73]) = (0); 
# 2941
(jcb[74]) = B_34; 
# 2942
(jcb[75]) = (((B_35 + B_47) + B_106) + B_108); 
# 2943
(jcb[76]) = B_48; 
# 2944
(jcb[77]) = (B_107 + B_109); 
# 2945
(jcb[78]) = (0); 
# 2946
(jcb[79]) = B_6; 
# 2947
(jcb[80]) = (B_7 + B_10); 
# 2948
(jcb[81]) = B_11; 
# 2949
(jcb[82]) = (0); 
# 2950
(jcb[83]) = (B_34 + B_73); 
# 2951
(jcb[84]) = B_35; 
# 2952
(jcb[85]) = B_74; 
# 2953
(jcb[86]) = (0); 
# 2954
(jcb[87]) = B_15; 
# 2955
(jcb[88]) = B_16; 
# 2956
(jcb[89]) = (0); 
# 2957
(jcb[90]) = B_6; 
# 2958
(jcb[91]) = B_7; 
# 2959
(jcb[92]) = (0); 
# 2960
(jcb[93]) = B_86; 
# 2961
(jcb[94]) = B_87; 
# 2962
(jcb[95]) = (0); 
# 2963
(jcb[96]) = ((3) * B_49); 
# 2964
(jcb[97]) = ((3) * B_50); 
# 2965
(jcb[98]) = (0); 
# 2966
(jcb[99]) = ((0.5999999999999999778) * B_126); 
# 2967
(jcb[100]) = B_69; 
# 2968
(jcb[101]) = B_128; 
# 2969
(jcb[102]) = (B_70 + ((0.5999999999999999778) * B_127)); 
# 2970
(jcb[103]) = (0); 
# 2971
(jcb[104]) = B_112; 
# 2972
(jcb[105]) = ((2) * B_114); 
# 2973
(jcb[106]) = B_116; 
# 2974
(jcb[107]) = ((2) * B_96); 
# 2975
(jcb[108]) = ((3) * B_94); 
# 2976
(jcb[109]) = ((((((3) * B_95) + ((2) * B_97)) + B_113) + ((2) * B_115)) + B_117); 
# 2977
(jcb[110]) = (-B_2); 
# 2978
(jcb[111]) = B_0; 
# 2979
(jcb[112]) = (-B_20); 
# 2980
(jcb[113]) = B_19; 
# 2981
(jcb[114]) = (-B_27); 
# 2982
(jcb[115]) = B_26; 
# 2983
(jcb[116]) = (-B_4); 
# 2984
(jcb[117]) = B_8; 
# 2985
(jcb[118]) = B_9; 
# 2986
(jcb[119]) = (-B_86); 
# 2987
(jcb[120]) = (-B_87); 
# 2988
(jcb[121]) = (-B_112); 
# 2989
(jcb[122]) = (-B_113); 
# 2990
(jcb[123]) = (-B_114); 
# 2991
(jcb[124]) = (-B_115); 
# 2992
(jcb[125]) = (-B_116); 
# 2993
(jcb[126]) = (-B_117); 
# 2994
(jcb[127]) = (-B_96); 
# 2995
(jcb[128]) = (-B_97); 
# 2996
(jcb[129]) = (-B_94); 
# 2997
(jcb[130]) = (-B_95); 
# 2998
(jcb[131]) = (-B_131); 
# 2999
(jcb[132]) = B_129; 
# 3000
(jcb[133]) = B_130; 
# 3001
(jcb[134]) = (-B_132); 
# 3002
(jcb[135]) = (-B_49); 
# 3003
(jcb[136]) = (-B_50); 
# 3004
(jcb[137]) = (-B_8); 
# 3005
(jcb[138]) = (-B_9); 
# 3006
(jcb[139]) = ((-B_0) - B_15); 
# 3007
(jcb[140]) = (-B_16); 
# 3008
(jcb[141]) = (-B_41); 
# 3009
(jcb[142]) = (-B_42); 
# 3010
(jcb[143]) = (-B_57); 
# 3011
(jcb[144]) = (-B_58); 
# 3012
(jcb[145]) = (-B_118); 
# 3013
(jcb[146]) = ((0.5999999999999999778) * B_126); 
# 3014
(jcb[147]) = B_128; 
# 3015
(jcb[148]) = ((-B_119) + ((0.5999999999999999778) * B_127)); 
# 3016
(jcb[149]) = ((-B_51) - B_88); 
# 3017
(jcb[150]) = (-B_52); 
# 3018
(jcb[151]) = (-B_89); 
# 3019
(jcb[152]) = ((-B_53) - B_90); 
# 3020
(jcb[153]) = (-B_54); 
# 3021
(jcb[154]) = (-B_91); 
# 3022
(jcb[155]) = ((2) * B_20); 
# 3023
(jcb[156]) = B_86; 
# 3024
(jcb[157]) = B_112; 
# 3025
(jcb[158]) = B_114; 
# 3026
(jcb[159]) = B_116; 
# 3027
(jcb[160]) = B_96; 
# 3028
(jcb[161]) = B_94; 
# 3029
(jcb[162]) = B_49; 
# 3030
(jcb[163]) = B_8; 
# 3031
(jcb[164]) = (-B_15); 
# 3032
(jcb[165]) = ((-B_16) - ((2) * B_19)); 
# 3033
(jcb[166]) = B_17; 
# 3034
(jcb[167]) = B_67; 
# 3035
(jcb[168]) = B_32; 
# 3036
(jcb[169]) = (((((((((((B_9 + B_12) + B_18) + B_33) + B_50) + B_68) + B_87) + B_95) + B_97) + B_113) + B_115) + B_117); 
# 3037
(jcb[170]) = B_13; 
# 3038
(jcb[171]) = ((-B_34) - B_73); 
# 3039
(jcb[172]) = (-B_35); 
# 3040
(jcb[173]) = (-B_74); 
# 3041
(jcb[174]) = ((-B_55) - B_92); 
# 3042
(jcb[175]) = (-B_56); 
# 3043
(jcb[176]) = (-B_93); 
# 3044
(jcb[177]) = (-B_126); 
# 3045
(jcb[178]) = (B_122 + B_137); 
# 3046
(jcb[179]) = (B_123 - B_127); 
# 3047
(jcb[180]) = B_138; 
# 3048
(jcb[181]) = ((-B_45) - B_80); 
# 3049
(jcb[182]) = (-B_46); 
# 3050
(jcb[183]) = (-B_81); 
# 3051
(jcb[184]) = ((-B_17) - B_28); 
# 3052
(jcb[185]) = (-B_18); 
# 3053
(jcb[186]) = (-B_29); 
# 3054
(jcb[187]) = B_14; 
# 3055
(jcb[188]) = (-B_124); 
# 3056
(jcb[189]) = B_71; 
# 3057
(jcb[190]) = B_39; 
# 3058
(jcb[191]) = (-B_125); 
# 3059
(jcb[192]) = B_40; 
# 3060
(jcb[193]) = B_72; 
# 3061
(jcb[194]) = (-B_100); 
# 3062
(jcb[195]) = B_110; 
# 3063
(jcb[196]) = B_39; 
# 3064
(jcb[197]) = (B_40 + B_111); 
# 3065
(jcb[198]) = (-B_101); 
# 3066
(jcb[199]) = B_23; 
# 3067
(jcb[200]) = (B_34 + B_73); 
# 3068
(jcb[201]) = ((-B_36) - B_75); 
# 3069
(jcb[202]) = B_77; 
# 3070
(jcb[203]) = B_38; 
# 3071
(jcb[204]) = (B_35 - B_37); 
# 3072
(jcb[205]) = (B_74 - B_76); 
# 3073
(jcb[206]) = B_100; 
# 3074
(jcb[207]) = ((-B_98) - B_110); 
# 3075
(jcb[208]) = B_102; 
# 3076
(jcb[209]) = (0); 
# 3077
(jcb[210]) = (B_103 - B_111); 
# 3078
(jcb[211]) = ((-B_99) + B_101); 
# 3079
(jcb[212]) = B_108; 
# 3080
(jcb[213]) = B_109; 
# 3081
(jcb[214]) = B_88; 
# 3082
(jcb[215]) = B_90; 
# 3083
(jcb[216]) = B_92; 
# 3084
(jcb[217]) = B_80; 
# 3085
(jcb[218]) = (-B_67); 
# 3086
(jcb[219]) = B_78; 
# 3087
(jcb[220]) = B_135; 
# 3088
(jcb[221]) = (-B_68); 
# 3089
(jcb[222]) = (0); 
# 3090
(jcb[223]) = ((((((B_63 + B_79) + B_81) + B_89) + B_91) + B_93) + B_136); 
# 3091
(jcb[224]) = B_64; 
# 3092
(jcb[225]) = B_41; 
# 3093
(jcb[226]) = B_57; 
# 3094
(jcb[227]) = B_51; 
# 3095
(jcb[228]) = B_53; 
# 3096
(jcb[229]) = B_55; 
# 3097
(jcb[230]) = B_45; 
# 3098
(jcb[231]) = B_28; 
# 3099
(jcb[232]) = (-B_32); 
# 3100
(jcb[233]) = B_43; 
# 3101
(jcb[234]) = B_133; 
# 3102
(jcb[235]) = (-B_33); 
# 3103
(jcb[236]) = ((((((((B_29 + B_42) + B_44) + B_46) + B_52) + B_54) + B_56) + B_58) + B_134); 
# 3104
(jcb[237]) = (0); 
# 3105
(jcb[238]) = (0); 
# 3106
(jcb[239]) = B_98; 
# 3107
(jcb[240]) = ((-B_69) - B_102); 
# 3108
(jcb[241]) = B_71; 
# 3109
(jcb[242]) = (0); 
# 3110
(jcb[243]) = (-B_70); 
# 3111
(jcb[244]) = (-B_103); 
# 3112
(jcb[245]) = (B_72 + B_99); 
# 3113
(jcb[246]) = (0); 
# 3114
(jcb[247]) = B_62; 
# 3115
(jcb[248]) = B_124; 
# 3116
(jcb[249]) = ((-B_128) - B_129); 
# 3117
(jcb[250]) = (0); 
# 3118
(jcb[251]) = (0); 
# 3119
(jcb[252]) = (((B_120 + B_125) + B_133) + B_135); 
# 3120
(jcb[253]) = B_121; 
# 3121
(jcb[254]) = (-B_130); 
# 3122
(jcb[255]) = B_134; 
# 3123
(jcb[256]) = B_136; 
# 3124
(jcb[257]) = B_75; 
# 3125
(jcb[258]) = ((-B_71) - B_77); 
# 3126
(jcb[259]) = (0); 
# 3127
(jcb[260]) = (-B_72); 
# 3128
(jcb[261]) = (0); 
# 3129
(jcb[262]) = B_76; 
# 3130
(jcb[263]) = B_126; 
# 3131
(jcb[264]) = B_45; 
# 3132
(jcb[265]) = B_124; 
# 3133
(jcb[266]) = (0); 
# 3134
(jcb[267]) = ((-B_43) - B_78); 
# 3135
(jcb[268]) = (0); 
# 3136
(jcb[269]) = (((B_120 + B_125) + B_133) + B_135); 
# 3137
(jcb[270]) = (B_121 + B_127); 
# 3138
(jcb[271]) = (((-B_44) + B_46) + B_134); 
# 3139
(jcb[272]) = ((-B_79) + B_136); 
# 3140
(jcb[273]) = B_47; 
# 3141
(jcb[274]) = ((B_48 + B_82) + B_84); 
# 3142
(jcb[275]) = (B_83 + B_85); 
# 3143
(jcb[276]) = B_36; 
# 3144
(jcb[277]) = (0); 
# 3145
(jcb[278]) = ((-B_38) - B_39); 
# 3146
(jcb[279]) = (-B_40); 
# 3147
(jcb[280]) = (0); 
# 3148
(jcb[281]) = B_37; 
# 3149
(jcb[282]) = (0); 
# 3150
(jcb[283]) = (-B_124); 
# 3151
(jcb[284]) = (0); 
# 3152
(jcb[285]) = (0); 
# 3153
(jcb[286]) = ((((((-B_120) - B_122) - B_125) - B_133) - B_135) - B_137); 
# 3154
(jcb[287]) = ((-B_121) - B_123); 
# 3155
(jcb[288]) = (-B_134); 
# 3156
(jcb[289]) = (-B_136); 
# 3157
(jcb[290]) = (0); 
# 3158
(jcb[291]) = (-B_138); 
# 3159
(jcb[292]) = (-B_86); 
# 3160
(jcb[293]) = (-B_112); 
# 3161
(jcb[294]) = (-B_114); 
# 3162
(jcb[295]) = (-B_116); 
# 3163
(jcb[296]) = (-B_96); 
# 3164
(jcb[297]) = (-B_94); 
# 3165
(jcb[298]) = (-B_49); 
# 3166
(jcb[299]) = (-B_8); 
# 3167
(jcb[300]) = ((2) * B_15); 
# 3168
(jcb[301]) = (-B_118); 
# 3169
(jcb[302]) = ((2) * B_16); 
# 3170
(jcb[303]) = (-B_126); 
# 3171
(jcb[304]) = B_45; 
# 3172
(jcb[305]) = (-B_17); 
# 3173
(jcb[306]) = (-B_67); 
# 3174
(jcb[307]) = (-B_32); 
# 3175
(jcb[308]) = (-B_69); 
# 3176
(jcb[309]) = (0); 
# 3177
(jcb[310]) = (0); 
# 3178
(jcb[311]) = (0); 
# 3179
(jcb[312]) = (0); 
# 3180
(jcb[313]) = ((-B_120) - B_122); 
# 3181
(jcb[314]) = ((((((((((((((((((-B_6) - B_9) - B_12) - B_18) - B_33) - B_50) - B_68) - B_70) - B_87) - B_95) - B_97) - B_113) - B_115) - B_117) - B_119) - B_121) - B_123) - B_127); 
# 3182
(jcb[315]) = ((-B_7) + B_10); 
# 3183
(jcb[316]) = B_46; 
# 3184
(jcb[317]) = (0); 
# 3185
(jcb[318]) = (0); 
# 3186
(jcb[319]) = (0); 
# 3187
(jcb[320]) = (B_11 - B_13); 
# 3188
(jcb[321]) = (0); 
# 3189
(jcb[322]) = B_2; 
# 3190
(jcb[323]) = (0); 
# 3191
(jcb[324]) = (0); 
# 3192
(jcb[325]) = (0); 
# 3193
(jcb[326]) = (0); 
# 3194
(jcb[327]) = (0); 
# 3195
(jcb[328]) = (-B_129); 
# 3196
(jcb[329]) = (0); 
# 3197
(jcb[330]) = (0); 
# 3198
(jcb[331]) = (0); 
# 3199
(jcb[332]) = (0); 
# 3200
(jcb[333]) = (-B_6); 
# 3201
(jcb[334]) = (((((-B_7) - B_10) - B_21) - B_59) - B_130); 
# 3202
(jcb[335]) = (-B_22); 
# 3203
(jcb[336]) = (-B_60); 
# 3204
(jcb[337]) = (0); 
# 3205
(jcb[338]) = (0); 
# 3206
(jcb[339]) = (-B_11); 
# 3207
(jcb[340]) = (0); 
# 3208
(jcb[341]) = ((3) * B_49); 
# 3209
(jcb[342]) = (-B_41); 
# 3210
(jcb[343]) = (-B_57); 
# 3211
(jcb[344]) = (-B_51); 
# 3212
(jcb[345]) = (-B_53); 
# 3213
(jcb[346]) = B_34; 
# 3214
(jcb[347]) = (-B_55); 
# 3215
(jcb[348]) = (-B_45); 
# 3216
(jcb[349]) = (-B_28); 
# 3217
(jcb[350]) = B_100; 
# 3218
(jcb[351]) = (B_98 - B_110); 
# 3219
(jcb[352]) = B_32; 
# 3220
(jcb[353]) = (-B_102); 
# 3221
(jcb[354]) = (0); 
# 3222
(jcb[355]) = (-B_43); 
# 3223
(jcb[356]) = (-B_39); 
# 3224
(jcb[357]) = (-B_133); 
# 3225
(jcb[358]) = (B_33 + ((3) * B_50)); 
# 3226
(jcb[359]) = (-B_21); 
# 3227
(jcb[360]) = (((((((((((((-B_22) - B_29) - B_40) - B_42) - B_44) - B_46) - B_52) - B_54) - B_56) - B_58) - B_103) - B_111) - B_134); 
# 3228
(jcb[361]) = (B_99 + B_101); 
# 3229
(jcb[362]) = ((((((2) * B_24) + B_25) + B_35) + B_47) + B_106); 
# 3230
(jcb[363]) = B_48; 
# 3231
(jcb[364]) = (0); 
# 3232
(jcb[365]) = B_107; 
# 3233
(jcb[366]) = B_86; 
# 3234
(jcb[367]) = B_112; 
# 3235
(jcb[368]) = ((2) * B_114); 
# 3236
(jcb[369]) = B_116; 
# 3237
(jcb[370]) = ((2) * B_96); 
# 3238
(jcb[371]) = ((3) * B_94); 
# 3239
(jcb[372]) = (-B_88); 
# 3240
(jcb[373]) = (-B_90); 
# 3241
(jcb[374]) = B_73; 
# 3242
(jcb[375]) = (-B_92); 
# 3243
(jcb[376]) = (-B_80); 
# 3244
(jcb[377]) = (-B_100); 
# 3245
(jcb[378]) = ((-B_98) + B_110); 
# 3246
(jcb[379]) = B_67; 
# 3247
(jcb[380]) = (B_69 + B_102); 
# 3248
(jcb[381]) = (-B_71); 
# 3249
(jcb[382]) = (-B_78); 
# 3250
(jcb[383]) = (0); 
# 3251
(jcb[384]) = ((-B_135) + B_137); 
# 3252
(jcb[385]) = (((((((B_68 + B_70) + B_87) + ((3) * B_95)) + ((2) * B_97)) + B_113) + ((2) * B_115)) + B_117); 
# 3253
(jcb[386]) = (-B_59); 
# 3254
(jcb[387]) = (B_103 + B_111); 
# 3255
(jcb[388]) = (((((((((((-B_60) - B_63) - B_72) - B_79) - B_81) - B_89) - B_91) - B_93) - B_99) - B_101) - B_136); 
# 3256
(jcb[389]) = (B_104 + B_106); 
# 3257
(jcb[390]) = B_84; 
# 3258
(jcb[391]) = (-B_64); 
# 3259
(jcb[392]) = (((((((2) * B_61) + B_74) + B_85) + B_105) + B_107) + B_138); 
# 3260
(jcb[393]) = ((2) * B_27); 
# 3261
(jcb[394]) = (-B_34); 
# 3262
(jcb[395]) = (-B_36); 
# 3263
(jcb[396]) = (0); 
# 3264
(jcb[397]) = B_38; 
# 3265
(jcb[398]) = B_21; 
# 3266
(jcb[399]) = B_22; 
# 3267
(jcb[400]) = (0); 
# 3268
(jcb[401]) = ((((((((((((-2) * B_23) - ((2) * B_24)) - ((2) * B_25)) - ((2) * B_26)) - B_30) - B_35) - B_37) - B_47) - B_104) - B_106) - B_108); 
# 3269
(jcb[402]) = (-B_48); 
# 3270
(jcb[403]) = (-B_31); 
# 3271
(jcb[404]) = (((-B_105) - B_107) - B_109); 
# 3272
(jcb[405]) = B_41; 
# 3273
(jcb[406]) = B_57; 
# 3274
(jcb[407]) = (B_55 + B_92); 
# 3275
(jcb[408]) = ((0.5999999999999999778) * B_126); 
# 3276
(jcb[409]) = B_80; 
# 3277
(jcb[410]) = B_128; 
# 3278
(jcb[411]) = (0); 
# 3279
(jcb[412]) = (0); 
# 3280
(jcb[413]) = (0); 
# 3281
(jcb[414]) = ((0.5999999999999999778) * B_127); 
# 3282
(jcb[415]) = (0); 
# 3283
(jcb[416]) = ((B_42 + B_56) + B_58); 
# 3284
(jcb[417]) = (B_81 + B_93); 
# 3285
(jcb[418]) = (-B_47); 
# 3286
(jcb[419]) = (((-B_48) - B_82) - B_84); 
# 3287
(jcb[420]) = (0); 
# 3288
(jcb[421]) = ((-B_83) - B_85); 
# 3289
(jcb[422]) = B_4; 
# 3290
(jcb[423]) = (-B_131); 
# 3291
(jcb[424]) = (0); 
# 3292
(jcb[425]) = B_118; 
# 3293
(jcb[426]) = ((0.4000000000000000222) * B_126); 
# 3294
(jcb[427]) = (B_17 + B_28); 
# 3295
(jcb[428]) = (0); 
# 3296
(jcb[429]) = (0); 
# 3297
(jcb[430]) = (B_43 + B_78); 
# 3298
(jcb[431]) = (0); 
# 3299
(jcb[432]) = B_122; 
# 3300
(jcb[433]) = (((((B_6 - B_12) + B_18) + B_119) + B_123) + ((0.4000000000000000222) * B_127)); 
# 3301
(jcb[434]) = (B_7 - B_10); 
# 3302
(jcb[435]) = (B_29 + B_44); 
# 3303
(jcb[436]) = ((-B_63) + B_79); 
# 3304
(jcb[437]) = ((-B_30) + B_47); 
# 3305
(jcb[438]) = (B_48 + B_84); 
# 3306
(jcb[439]) = (((((((-B_11) - B_13) - ((2) * B_14)) - B_31) - B_64) - B_65) - B_132); 
# 3307
(jcb[440]) = ((-B_66) + B_85); 
# 3308
(jcb[441]) = (-B_73); 
# 3309
(jcb[442]) = (-B_75); 
# 3310
(jcb[443]) = B_77; 
# 3311
(jcb[444]) = (0); 
# 3312
(jcb[445]) = (-B_137); 
# 3313
(jcb[446]) = (0); 
# 3314
(jcb[447]) = B_59; 
# 3315
(jcb[448]) = (0); 
# 3316
(jcb[449]) = B_60; 
# 3317
(jcb[450]) = (((-B_104) - B_106) - B_108); 
# 3318
(jcb[451]) = ((-B_82) - B_84); 
# 3319
(jcb[452]) = (-B_65); 
# 3320
(jcb[453]) = ((((((((((((-2) * B_61) - ((2) * B_62)) - B_66) - B_74) - B_76) - B_83) - B_85) - B_105) - B_107) - B_109) - B_138); 
# 3321
} 
#endif
# 3323 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) void Fun(double *var, const double *__restrict__ fix, const double *__restrict__ rconst, double *varDot, int &Nfun, const int VL_GLO) {int volatile ___ = 1;(void)var;(void)fix;(void)rconst;(void)varDot;(void)Nfun;(void)VL_GLO;
# 3482
::exit(___);}
#if 0
# 3323
{ 
# 3324
int index = ((__device_builtin_variable_blockIdx.x) * (__device_builtin_variable_blockDim.x)) + (__device_builtin_variable_threadIdx.x); 
# 3326
Nfun++; 
# 3328
double dummy, A_0, A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8, A_9, A_10, A_11, A_12, A_13, A_14, A_15, A_16, A_17, A_18, A_19, A_20, A_21, A_22, A_23, A_24, A_25, A_26, A_27, A_28, A_29, A_30, A_31, A_32, A_33, A_34, A_35, A_36, A_37, A_38, A_39, A_40, A_41, A_42, A_43, A_44, A_45, A_46, A_47, A_48, A_49, A_50, A_51, A_52, A_53, A_54, A_55, A_56, A_57, A_58, A_59, A_60, A_61, A_62, A_63, A_64, A_65, A_66, A_67, A_68, A_69, A_70, A_71, A_72, A_73, A_74, A_75, A_76; 
# 3330
{ 
# 3331
A_0 = (((rconst[0]) * (var[41])) * (fix[0])); 
# 3332
A_1 = (((rconst[1]) * (var[28])) * (fix[0])); 
# 3333
A_2 = (((rconst[2]) * (var[31])) * (fix[0])); 
# 3334
A_3 = (((rconst[3]) * (var[65])) * (var[66])); 
# 3335
A_4 = (((rconst[4]) * (var[40])) * (var[65])); 
# 3336
A_5 = (((rconst[5]) * (var[66])) * (var[71])); 
# 3337
A_6 = (((rconst[6]) * (var[65])) * (var[71])); 
# 3338
A_7 = (((rconst[7]) * (var[71])) * (var[71])); 
# 3339
A_8 = (((rconst[8]) * (var[41])) * (var[47])); 
# 3340
A_9 = (((1.800000000000000004e-12) * (var[52])) * (var[65])); 
# 3341
A_10 = (((rconst[10]) * (var[47])) * (var[47])); 
# 3342
A_11 = ((1000000.0) * (var[29])); 
# 3343
A_12 = (((rconst[12]) * (var[66])) * (var[67])); 
# 3344
A_13 = (((rconst[13]) * (var[69])) * (var[69])); 
# 3345
A_14 = (((rconst[14]) * (var[69])) * (var[69])); 
# 3346
A_15 = (((rconst[15]) * (var[69])) * (var[69])); 
# 3347
A_16 = (((rconst[16]) * (var[69])) * (var[69])); 
# 3348
A_17 = ((rconst[17]) * (var[30])); 
# 3349
A_18 = (((rconst[18]) * (var[52])) * (var[67])); 
# 3350
A_19 = (((rconst[19]) * (var[69])) * (var[71])); 
# 3351
A_20 = (((rconst[20]) * (var[58])) * (var[65])); 
# 3352
A_21 = (((rconst[21]) * (var[48])) * (var[69])); 
# 3353
A_22 = (((rconst[22]) * (var[55])) * (var[69])); 
# 3354
A_23 = ((rconst[23]) * (var[63])); 
# 3355
A_24 = (((rconst[24]) * (var[63])) * (var[67])); 
# 3356
A_25 = (((rconst[25]) * (var[42])) * (var[67])); 
# 3357
A_26 = (((rconst[26]) * (var[62])) * (var[67])); 
# 3358
A_27 = (((5.900000000000000305e-11) * (var[51])) * (var[67])); 
# 3359
A_28 = (((rconst[28]) * (var[69])) * (var[70])); 
# 3360
A_29 = (((rconst[29]) * (var[39])) * (var[65])); 
# 3361
A_30 = (((rconst[30]) * (var[45])) * (var[67])); 
# 3362
A_31 = (((7.999999999999999516e-11) * (var[46])) * (var[67])); 
# 3363
A_32 = (((rconst[32]) * (var[49])) * (var[67])); 
# 3364
A_33 = (((rconst[33]) * (var[43])) * (var[67])); 
# 3365
A_34 = (((rconst[34]) * (var[66])) * (var[68])); 
# 3366
A_35 = (((2.699999999999999804e-12) * (var[72])) * (var[72])); 
# 3367
A_36 = (((rconst[36]) * (var[72])) * (var[72])); 
# 3368
A_37 = (((rconst[37]) * (var[68])) * (var[71])); 
# 3369
A_38 = (((rconst[38]) * (var[71])) * (var[72])); 
# 3370
A_39 = (((rconst[39]) * (var[57])) * (var[65])); 
# 3371
A_40 = (((rconst[40]) * (var[59])) * (var[65])); 
# 3372
A_41 = (((4.899999999999999881e-11) * (var[61])) * (var[68])); 
# 3373
A_42 = (((rconst[42]) * (var[48])) * (var[72])); 
# 3374
A_43 = (((rconst[43]) * (var[55])) * (var[72])); 
# 3375
A_44 = ((rconst[44]) * (var[61])); 
# 3376
A_45 = (((rconst[45]) * (var[62])) * (var[68])); 
# 3377
A_46 = (((rconst[46]) * (var[51])) * (var[68])); 
# 3378
A_47 = (((rconst[47]) * (var[70])) * (var[72])); 
# 3379
A_48 = (((rconst[48]) * (var[70])) * (var[72])); 
# 3380
A_49 = (((rconst[49]) * (var[32])) * (var[65])); 
# 3381
A_50 = (((rconst[50]) * (var[45])) * (var[68])); 
# 3382
A_51 = (((rconst[51]) * (var[46])) * (var[68])); 
# 3383
A_52 = (((rconst[52]) * (var[49])) * (var[68])); 
# 3384
A_53 = (((rconst[53]) * (var[37])) * (var[65])); 
# 3385
A_54 = (((rconst[54]) * (var[36])) * (var[65])); 
# 3386
A_55 = (((3.319999999999999911e-15) * (var[56])) * (var[68])); 
# 3387
A_56 = (((1.099999999999999928e-15) * (var[54])) * (var[68])); 
# 3388
A_57 = (((rconst[57]) * (var[59])) * (var[67])); 
# 3389
A_58 = (((rconst[58]) * (var[69])) * (var[72])); 
# 3390
A_59 = (((rconst[59]) * (var[69])) * (var[72])); 
# 3391
A_60 = (((rconst[60]) * (var[69])) * (var[72])); 
# 3392
A_61 = (((1.450000000000000001e-11) * (var[56])) * (var[67])); 
# 3393
A_62 = (((rconst[62]) * (var[33])) * (var[65])); 
# 3394
A_63 = (((rconst[63]) * (var[34])) * (var[65])); 
# 3395
A_64 = (((rconst[64]) * (var[35])) * (var[65])); 
# 3396
A_65 = (((rconst[65]) * (var[44])) * (var[65])); 
# 3397
A_66 = (((rconst[66]) * (var[64])) * (var[65])); 
# 3398
A_67 = (((rconst[67]) * (var[64])) * (var[65])); 
# 3399
A_68 = (((rconst[68]) * (var[53])) * (var[64])); 
# 3400
A_69 = (((1.000000000000000036e-10) * (var[50])) * (var[65])); 
# 3401
A_70 = ((rconst[70]) * (var[60])); 
# 3402
A_71 = (((2.999999999999999839e-13) * (var[60])) * (var[66])); 
# 3403
A_72 = (((5.000000000000000182e-11) * (var[38])) * (var[71])); 
# 3404
A_73 = (((3.299999999999999978e-10) * (var[64])) * (var[67])); 
# 3405
A_74 = (((rconst[74]) * (var[64])) * (var[68])); 
# 3406
A_75 = (((4.399999999999999932e-13) * (var[64])) * (var[72])); 
# 3407
A_76 = ((rconst[76]) * (var[0])); 
# 3408
(varDot[0]) = (-A_76); 
# 3409
(varDot[1]) = A_68; 
# 3410
(varDot[2]) = (A_26 + A_45); 
# 3411
(varDot[3]) = (A_31 + A_51); 
# 3412
(varDot[4]) = A_19; 
# 3413
(varDot[5]) = (A_15 + A_58); 
# 3414
(varDot[6]) = ((A_38 + A_40) + A_47); 
# 3415
(varDot[7]) = A_65; 
# 3416
(varDot[8]) = (((0.4000000000000000222) * A_69) + A_72); 
# 3417
(varDot[9]) = ((2) * A_76); 
# 3418
(varDot[10]) = ((2) * A_76); 
# 3419
(varDot[11]) = (((0.6666670000000000096) * A_30) + ((0.6666670000000000096) * A_50)); 
# 3420
(varDot[12]) = A_3; 
# 3421
(varDot[13]) = A_5; 
# 3422
(varDot[14]) = A_7; 
# 3423
(varDot[15]) = A_5; 
# 3424
(varDot[16]) = A_8; 
# 3425
(varDot[17]) = (((((((((((((((((A_3 + A_5) + A_8) + ((2) * A_13)) + ((2) * A_14)) + A_15) + A_28) + ((2) * A_35)) + ((2) * A_36)) + A_48) + A_58) + ((2) * A_59)) + ((2) * A_60)) + A_65) + A_68) + A_71) + A_75) + ((3) * A_76)); 
# 3426
(varDot[18]) = (((((((2) * A_35) + ((2) * A_36)) + A_42) + A_48) + A_59) + A_60); 
# 3427
(varDot[19]) = (((A_21 + A_28) + A_59) + A_60); 
# 3428
(varDot[20]) = (A_3 + A_5); 
# 3429
(varDot[21]) = (A_21 + A_42); 
# 3430
(varDot[22]) = A_8; 
# 3431
(varDot[23]) = A_3; 
# 3432
(varDot[24]) = A_49; 
# 3433
(varDot[25]) = ((3) * A_29); 
# 3434
(varDot[26]) = ((A_40 + ((0.5999999999999999778) * A_69)) + A_70); 
# 3435
(varDot[27]) = ((((((3) * A_53) + ((2) * A_54)) + A_62) + ((2) * A_63)) + A_64); 
# 3436
(varDot[28]) = (A_0 - A_1); 
# 3437
(varDot[29]) = (A_10 - A_11); 
# 3438
(varDot[30]) = (A_16 - A_17); 
# 3439
(varDot[31]) = ((-A_2) + A_4); 
# 3440
(varDot[32]) = (-A_49); 
# 3441
(varDot[33]) = (-A_62); 
# 3442
(varDot[34]) = (-A_63); 
# 3443
(varDot[35]) = (-A_64); 
# 3444
(varDot[36]) = (-A_54); 
# 3445
(varDot[37]) = (-A_53); 
# 3446
(varDot[38]) = (A_71 - A_72); 
# 3447
(varDot[39]) = (-A_29); 
# 3448
(varDot[40]) = (-A_4); 
# 3449
(varDot[41]) = ((-A_0) - A_8); 
# 3450
(varDot[42]) = (-A_25); 
# 3451
(varDot[43]) = (-A_33); 
# 3452
(varDot[44]) = (((-A_65) + ((0.5999999999999999778) * A_69)) + A_70); 
# 3453
(varDot[45]) = ((-A_30) - A_50); 
# 3454
(varDot[46]) = ((-A_31) - A_51); 
# 3455
(varDot[47]) = ((((((((((((((A_4 + A_6) - A_8) + A_9) - ((2) * A_10)) + ((2) * A_11)) + A_20) + A_29) + A_39) + A_49) + A_53) + A_54) + A_62) + A_63) + A_64); 
# 3456
(varDot[48]) = ((-A_21) - A_42); 
# 3457
(varDot[49]) = ((-A_32) - A_52); 
# 3458
(varDot[50]) = ((A_67 - A_69) + A_75); 
# 3459
(varDot[51]) = ((-A_27) - A_46); 
# 3460
(varDot[52]) = ((A_7 - A_9) - A_18); 
# 3461
(varDot[53]) = ((A_24 + A_41) - A_68); 
# 3462
(varDot[54]) = (((A_13 + A_24) - A_56) + A_61); 
# 3463
(varDot[55]) = (((((A_21 - A_22) + A_23) + A_42) - A_43) + A_44); 
# 3464
(varDot[56]) = (((((-A_55) + A_56) + A_57) + A_60) - A_61); 
# 3465
(varDot[57]) = (((((((A_37 - A_39) + A_45) + A_46) + A_50) + A_51) + A_52) + A_74); 
# 3466
(varDot[58]) = (((((((((A_18 - A_20) + A_25) + A_26) + A_27) + A_30) + A_31) + A_32) + A_33) + A_73); 
# 3467
(varDot[59]) = ((((A_36 - A_40) + A_41) + A_55) - A_57); 
# 3468
(varDot[60]) = (((((A_66 + A_68) - A_70) - A_71) + A_73) + A_74); 
# 3469
(varDot[61]) = (((-A_41) + A_43) - A_44); 
# 3470
(varDot[62]) = (((((((((((-A_26) + A_27) + A_28) - A_45) + A_47) + A_48) + A_66) + A_68) + A_69) + A_73) + A_74); 
# 3471
(varDot[63]) = ((A_22 - A_23) - A_24); 
# 3472
(varDot[64]) = ((((((-A_66) - A_67) - A_68) - A_73) - A_74) - A_75); 
# 3473
(varDot[65]) = (((((((((((((((((((((-A_3) - A_4) + A_5) - A_6) + ((2) * A_8)) - A_9) - A_20) + A_27) - A_29) - A_39) - A_40) - A_49) - A_53) - A_54) - A_62) - A_63) - A_64) - A_65) - A_66) - A_67) - A_69); 
# 3474
(varDot[66]) = (((((A_1 - A_3) - A_5) - A_12) - A_34) - A_71); 
# 3475
(varDot[67]) = ((((((((((((((((((((((-A_12) + ((2) * A_14)) + A_15) - A_18) + A_20) + A_21) - A_24) - A_25) - A_26) - A_27) + A_28) + ((3) * A_29)) - A_30) - A_31) - A_32) - A_33) + A_55) + A_56) - A_57) + A_59) - A_61) - A_73); 
# 3476
(varDot[68]) = (((((((((((((((((((((((((((-A_34) + ((2) * A_35)) - A_37) + A_39) + A_40) - A_41) + A_42) - A_45) - A_46) + A_48) + A_49) - A_50) - A_51) - A_52) + ((3) * A_53)) + ((2) * A_54)) - A_55) - A_56) + A_57) + A_58) + A_59) + A_61) + A_62) + ((2) * A_63)) + A_64) - A_74) + A_75); 
# 3477
(varDot[69]) = (((((((((((((A_12 - ((2) * A_13)) - ((2) * A_14)) - ((2) * A_15)) - ((2) * A_16)) + ((2) * A_17)) - A_19) - A_21) - A_22) + A_23) - A_28) - A_58) - A_59) - A_60); 
# 3478
(varDot[70]) = (((((((((A_25 - A_28) + A_32) + A_33) + A_46) - A_47) - A_48) + A_52) + ((0.5999999999999999778) * A_69)) + A_70); 
# 3479
(varDot[71]) = (((((((((((((((((A_2 + A_3) - A_5) - A_6) - ((2) * A_7)) + A_9) + A_18) - A_19) + A_26) + A_28) - A_37) - A_38) + A_45) + A_48) + A_65) + A_67) + ((0.4000000000000000222) * A_69)) - A_72); 
# 3480
(varDot[72]) = ((((((((((((A_34 - ((2) * A_35)) - ((2) * A_36)) - A_38) - A_42) - A_43) + A_44) - A_47) - A_48) - A_58) - A_59) - A_60) - A_75); 
# 3481
} 
# 3482
} 
#endif
# 3484 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) void ros_FunTimeDerivative(const double T, double roundoff, double *__restrict__ var, const double *__restrict__ fix, const double *__restrict__ 
# 3485
rconst, double *dFdT, double *Fcn0, int &Nfun, const double *__restrict__ 
# 3486
khet_st, const double *__restrict__ khet_tr, const double *__restrict__ 
# 3487
jx, const int 
# 3488
VL_GLO) 
# 3489
{int volatile ___ = 1;(void)T;(void)roundoff;(void)var;(void)fix;(void)rconst;(void)dFdT;(void)Fcn0;(void)Nfun;(void)khet_st;(void)khet_tr;(void)jx;(void)VL_GLO;
# 3502
::exit(___);}
#if 0
# 3489
{ 
# 3490
int index = ((__device_builtin_variable_blockIdx.x) * (__device_builtin_variable_blockDim.x)) + (__device_builtin_variable_threadIdx.x); 
# 3491
const double DELTAMIN = (9.999999999999999547e-07); 
# 3492
double delta, one_over_delta; 
# 3494
delta = (sqrt(roundoff) * fmax(DELTAMIN, fabs(T))); 
# 3495
one_over_delta = ((1.0) / delta); 
# 3497
Fun(var, fix, rconst, dFdT, Nfun, VL_GLO); 
# 3499
for (int i = 0; i < 73; i++) { 
# 3500
(dFdT[i]) = (((dFdT[i]) - (Fcn0[i])) * one_over_delta); 
# 3501
}  
# 3502
} 
#endif
# 3504 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) static int ros_Integrator(double *__restrict__ var, const double *__restrict__ fix, const double Tstart, const double Tend, double &T, const int 
# 3506
ros_S, const double *__restrict__ ros_M, const double *__restrict__ ros_E, const double *__restrict__ ros_A, const double *__restrict__ ros_C, const double *__restrict__ 
# 3507
ros_Alpha, const double *__restrict__ ros_Gamma, const double ros_ELO, const int *ros_NewF, const int 
# 3509
autonomous, const int vectorTol, const int Max_no_steps, const double 
# 3510
roundoff, const double Hmin, const double Hmax, const double Hstart, double &Hexit, const double 
# 3511
FacMin, const double FacMax, const double FacRej, const double FacSafe, int &
# 3513
Nfun, int &Njac, int &Nstp, int &Nacc, int &Nrej, int &Ndec, int &Nsol, int &Nsng, const double *__restrict__ 
# 3515
rconst, const double *__restrict__ absTol, const double *__restrict__ relTol, double *__restrict__ varNew, double *__restrict__ Fcn0, double *__restrict__ 
# 3516
K, double *__restrict__ dFdT, double *__restrict__ jac0, double *__restrict__ Ghimj, double *__restrict__ varErr, const double *__restrict__ 
# 3518
khet_st, const double *__restrict__ khet_tr, const double *__restrict__ 
# 3519
jx, const int 
# 3521
VL_GLO) 
# 3522
{int volatile ___ = 1;(void)var;(void)fix;(void)Tstart;(void)Tend;(void)T;(void)ros_S;(void)ros_M;(void)ros_E;(void)ros_A;(void)ros_C;(void)ros_Alpha;(void)ros_Gamma;(void)ros_ELO;(void)ros_NewF;(void)autonomous;(void)vectorTol;(void)Max_no_steps;(void)roundoff;(void)Hmin;(void)Hmax;(void)Hstart;(void)Hexit;(void)FacMin;(void)FacMax;(void)FacRej;(void)FacSafe;(void)Nfun;(void)Njac;(void)Nstp;(void)Nacc;(void)Nrej;(void)Ndec;(void)Nsol;(void)Nsng;(void)rconst;(void)absTol;(void)relTol;(void)varNew;(void)Fcn0;(void)K;(void)dFdT;(void)jac0;(void)Ghimj;(void)varErr;(void)khet_st;(void)khet_tr;(void)jx;(void)VL_GLO;
# 3694
::exit(___);}
#if 0
# 3522
{ 
# 3523
int index = ((__device_builtin_variable_blockIdx.x) * (__device_builtin_variable_blockDim.x)) + (__device_builtin_variable_threadIdx.x); 
# 3525
double H, Hnew, HC, HG, Fac; 
# 3526
double Err; 
# 3527
int direction; 
# 3528
int rejectLastH, rejectMoreH; 
# 3529
const double DELTAMIN = (1.000000000000000082e-05); 
# 3532
T = Tstart; 
# 3533
Hexit = (0.0); 
# 3534
H = fmin(Hstart, Hmax); 
# 3535
if (fabs(H) <= ((10.0) * roundoff)) { 
# 3536
H = DELTAMIN; }  
# 3538
if (Tend >= Tstart) 
# 3539
{ 
# 3540
direction = (+1); 
# 3541
} else 
# 3543
{ 
# 3544
direction = (-1); 
# 3545
}  
# 3547
rejectLastH = 0; 
# 3548
rejectMoreH = 0; 
# 3555
while (((direction > 0) && (((T - Tend) + roundoff) <= (0.0))) || ((direction < 0) && (((Tend - T) + roundoff) <= (0.0)))) 
# 3556
{ 
# 3557
if (Nstp > Max_no_steps) { 
# 3558
return -6; }  
# 3560
if (H <= roundoff) { 
# 3562
return -7; 
# 3563
}  
# 3566
Hexit = H; 
# 3567
H = fmin(H, fabs(Tend - T)); 
# 3570
Fun(var, fix, rconst, Fcn0, Nfun, VL_GLO); 
# 3573
if (!autonomous) { 
# 3574
ros_FunTimeDerivative(T, roundoff, var, fix, rconst, dFdT, Fcn0, Nfun, khet_st, khet_tr, jx, VL_GLO); }  
# 3577
Jac_sp(var, fix, rconst, jac0, Njac, VL_GLO); 
# 3581
while (1) 
# 3582
{ 
# 3583
ros_PrepareMatrix(H, direction, ros_Gamma[0], jac0, Ghimj, Nsng, Ndec, VL_GLO); 
# 3586
for (int istage = 0; istage < ros_S; istage++) 
# 3587
{ 
# 3589
if (istage == 0) 
# 3590
{ 
# 3591
for (int i = 0; i < 73; i++) { 
# 3592
(varNew[i]) = (Fcn0[i]); 
# 3593
}  
# 3594
} else { 
# 3595
if (ros_NewF[istage]) 
# 3596
{ 
# 3597
for (int i = 0; i < 73; i++) { 
# 3598
(varNew[i]) = (var[i]); 
# 3599
}  
# 3601
for (int j = 0; j < istage; j++) { 
# 3602
for (int i = 0; i < 73; i++) { 
# 3603
(varNew[i]) = (((K[(j * 73) + i]) * (ros_A[((istage * (istage - 1)) / 2) + j])) + (varNew[i])); 
# 3604
}  
# 3605
}  
# 3606
Fun(varNew, fix, rconst, varNew, Nfun, VL_GLO); 
# 3607
}  }  
# 3609
for (int i = 0; i < 73; i++) { 
# 3610
(K[(istage * 73) + i]) = (varNew[i]); }  
# 3612
for (int j = 0; j < istage; j++) 
# 3613
{ 
# 3614
HC = ((ros_C[((istage * (istage - 1)) / 2) + j]) / (direction * H)); 
# 3615
for (int i = 0; i < 73; i++) { 
# 3616
double tmp = K[(j * 73) + i]; 
# 3617
(K[(istage * 73) + i]) += (tmp * HC); 
# 3618
}  
# 3619
}  
# 3621
if ((!autonomous) && (ros_Gamma[istage])) 
# 3622
{ 
# 3623
HG = ((direction * H) * (ros_Gamma[istage])); 
# 3624
for (int i = 0; i < 73; i++) { 
# 3625
(K[(istage * 73) + i]) += ((dFdT[i]) * HG); 
# 3626
}  
# 3627
}  
# 3629
ros_Solve(Ghimj, K, Nsol, istage, ros_S); 
# 3632
}  
# 3635
for (int i = 0; i < 73; i++) { 
# 3636
double tmpNew = var[i]; 
# 3637
double tmpErr = (0.0); 
# 3639
for (int j = 0; j < ros_S; j++) { 
# 3640
double tmp = K[(j * 73) + i]; 
# 3648
tmpNew += (tmp * (ros_M[j])); 
# 3649
tmpErr += (tmp * (ros_E[j])); 
# 3650
}  
# 3651
(varNew[i]) = tmpNew; 
# 3652
(varErr[i]) = tmpErr; 
# 3653
}  
# 3655
Err = ros_ErrorNorm(var, varNew, varErr, absTol, relTol, vectorTol); 
# 3659
Fac = fmin(FacMax, fmax(FacMin, FacSafe / pow(Err, (1.0) / ros_ELO))); 
# 3660
Hnew = (H * Fac); 
# 3663
Nstp = (Nstp + 1); 
# 3664
if ((Err <= (1.0)) || (H <= Hmin)) 
# 3665
{ 
# 3666
Nacc = (Nacc + 1); 
# 3667
for (int j = 0; j < 73; j++) { 
# 3668
(var[j]) = fmax(varNew[j], (0.0)); }  
# 3670
T = (T + (direction * H)); 
# 3671
Hnew = fmax(Hmin, fmin(Hnew, Hmax)); 
# 3672
if (rejectLastH) { 
# 3673
Hnew = fmin(Hnew, H); }  
# 3674
rejectLastH = 0; 
# 3675
rejectMoreH = 0; 
# 3676
H = Hnew; 
# 3678
break; 
# 3679
} else 
# 3681
{ 
# 3682
if (rejectMoreH) { 
# 3683
Hnew = (H * FacRej); }  
# 3684
rejectMoreH = rejectLastH; 
# 3685
rejectLastH = 1; 
# 3686
H = Hnew; 
# 3687
if (Nacc >= 1) { 
# 3688
Nrej += 1; }  
# 3689
}  
# 3690
}  
# 3691
}  
# 3693
return 0; 
# 3694
} 
#endif
# 3706 "messy_mecca_kpp_acc.cu"
typedef 
# 3696
struct { 
# 3697
double ros_A[15]; 
# 3698
double ros_C[15]; 
# 3699
int ros_NewF[8]; 
# 3700
double ros_M[6]; 
# 3701
double ros_E[6]; 
# 3702
double ros_Alpha[6]; 
# 3703
double ros_Gamma[6]; 
# 3704
double ros_ELO; 
# 3705
int ros_S; 
# 3706
} ros_t; 
# 3711
static ros_t ros[5]; 
# 3790 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) double k_3rd(double temp, double cair, double k0_300K, double n, double kinf_300K, double m, double fc) 
# 3802 "messy_mecca_kpp_acc.cu"
{int volatile ___ = 1;(void)temp;(void)cair;(void)k0_300K;(void)n;(void)kinf_300K;(void)m;(void)fc;
# 3812
::exit(___);}
#if 0
# 3802
{ 
# 3804
double zt_help, k0_T, kinf_T, k_ratio, k_3rd_r; 
# 3806
zt_help = ((300.0) / temp); 
# 3807
k0_T = ((k0_300K * pow(zt_help, n)) * cair); 
# 3808
kinf_T = (kinf_300K * pow(zt_help, m)); 
# 3809
k_ratio = (k0_T / kinf_T); 
# 3810
k_3rd_r = ((k0_T / ((1.0) + k_ratio)) * pow(fc, (1.0) / ((1.0) + pow(log10(k_ratio), 2)))); 
# 3811
return k_3rd_r; 
# 3812
} 
#endif
# 3814 "messy_mecca_kpp_acc.cu"
__attribute__((unused)) double k_3rd_iupac(double temp, double cair, double k0_300K, double n, double kinf_300K, double m, double fc) 
# 3827 "messy_mecca_kpp_acc.cu"
{int volatile ___ = 1;(void)temp;(void)cair;(void)k0_300K;(void)n;(void)kinf_300K;(void)m;(void)fc;
# 3837
::exit(___);}
#if 0
# 3827
{ 
# 3829
double zt_help, k0_T, kinf_T, k_ratio, nu, k_3rd_iupac_r; 
# 3830
zt_help = ((300.0) / temp); 
# 3831
k0_T = ((k0_300K * pow(zt_help, n)) * cair); 
# 3832
kinf_T = (kinf_300K * pow(zt_help, m)); 
# 3833
k_ratio = (k0_T / kinf_T); 
# 3834
nu = ((0.75) - ((1.270000000000000018) * log10(fc))); 
# 3835
k_3rd_iupac_r = ((k0_T / ((1.0) + k_ratio)) * pow(fc, (1.0) / ((1.0) + pow(log10(k_ratio) / nu, 2)))); 
# 3836
return k_3rd_iupac_r; 
# 3837
} 
#endif
# 3842 "messy_mecca_kpp_acc.cu"
double *temp_gpu; 
# 3843
double *press_gpu; 
# 3844
double *cair_gpu; 
# 3847
__attribute__((unused)) void update_rconst(const double *__restrict__ var, const double *__restrict__ 
# 3848
khet_st, const double *__restrict__ khet_tr, const double *__restrict__ 
# 3849
jx, double *__restrict__ rconst, const double *__restrict__ 
# 3850
temp_gpu, const double *__restrict__ 
# 3851
press_gpu, const double *__restrict__ 
# 3852
cair_gpu, const int 
# 3853
VL_GLO) 
# 3854
{int volatile ___ = 1;(void)var;(void)khet_st;(void)khet_tr;(void)jx;(void)rconst;(void)temp_gpu;(void)press_gpu;(void)cair_gpu;(void)VL_GLO;
# 4059
::exit(___);}
#if 0
# 3854
{ 
# 3855
int index = ((__device_builtin_variable_blockIdx.x) * (__device_builtin_variable_blockDim.x)) + (__device_builtin_variable_threadIdx.x); 
# 3859
{ 
# 3860
const double temp_loc = temp_gpu[index]; 
# 3861
const double press_loc = press_gpu[index]; 
# 3862
const double cair_loc = cair_gpu[index]; 
# 3864
double alpha_NO_HO2, beta_NO_HO2, k0_NO_HO2, k2d_NO_HO2, k1d_NO_HO2, k2w_NO_HO2, k1w_NO_HO2, k_PrO2_HO2, k_PrO2_NO, k_PrO2_CH3O2, k_HO2_HO2, k_NO3_NO2, k_NO2_HO2, k_HNO3_OH, k_CH3O2, k_CH3OOH_OH, k_CH3CO3_NO2, k_PAN_M, k_ClO_ClO, k_BrO_NO2, k_I_NO2, k_DMS_OH, G7402a_yield, k_O3s, beta_null_CH3NO3, beta_inf_CH3NO3, beta_CH3NO3, k_NO2_CH3O2, k_G4138, k_G9408, KRO2NO, KRO2HO2, KAPHO2, KAPNO, KRO2NO3, KNO3AL, J_IC3H7NO3, J_ACETOL, RO2; 
# 3866
alpha_NO_HO2 = ((((var[47]) * (6.599999999999999967e-27)) * temp_loc) * exp((3700.0) / temp_loc)); 
# 3867
beta_NO_HO2 = max(((((530.0) / temp_loc) + (press_loc * (4.800399999999999813e-06))) - (1.729999999999999982)) * (0.01000000000000000021), (0.0)); 
# 3868
k0_NO_HO2 = ((3.500000000000000031e-12) * exp((250.0) / temp_loc)); 
# 3869
k2d_NO_HO2 = ((beta_NO_HO2 * k0_NO_HO2) / ((1.0) + beta_NO_HO2)); 
# 3870
k1d_NO_HO2 = (k0_NO_HO2 - k2d_NO_HO2); 
# 3871
k2w_NO_HO2 = (((beta_NO_HO2 * k0_NO_HO2) * ((1.0) + ((42.0) * alpha_NO_HO2))) / (((1.0) + alpha_NO_HO2) * ((1.0) + beta_NO_HO2))); 
# 3872
k1w_NO_HO2 = (k0_NO_HO2 - k2w_NO_HO2); 
# 3873
k_PrO2_HO2 = ((1.899999999999999982e-13) * exp((1300.0) / temp_loc)); 
# 3874
k_PrO2_NO = ((2.699999999999999804e-12) * exp((360.0) / temp_loc)); 
# 3875
k_PrO2_CH3O2 = ((9.459999999999999575e-14) * exp((431.0) / temp_loc)); 
# 3876
k_HO2_HO2 = ((((1.500000000000000071e-12) * exp((19.0) / temp_loc)) + (((1.700000000000000027e-33) * exp((1000.0) / temp_loc)) * cair_loc)) * ((1.0) + (((1.400000000000000021e-21) * exp((2200.0) / temp_loc)) * (var[47])))); 
# 3877
k_NO3_NO2 = k_3rd(temp_loc, cair_loc, (2.000000000000000167e-30), (4.400000000000000355), (1.400000000000000093e-12), (0.6999999999999999556), (0.5999999999999999778)); 
# 3878
k_NO2_HO2 = k_3rd(temp_loc, cair_loc, (2.000000000000000167e-31), (3.399999999999999911), (2.900000000000000164e-12), (1.100000000000000089), (0.5999999999999999778)); 
# 3879
k_HNO3_OH = (((2.399999999999999871e-14) * exp((460.0) / temp_loc)) + ((1.0) / (((1.0) / (((6.499999999999999851e-34) * exp((1335.0) / temp_loc)) * cair_loc)) + ((1.0) / ((2.700000000000000116e-17) * exp((2199.0) / temp_loc)))))); 
# 3880
k_CH3O2 = ((1.030000000000000029e-13) * exp((365.0) / temp_loc)); 
# 3881
k_CH3OOH_OH = ((5.299999999999999631e-12) * exp((190.0) / temp_loc)); 
# 3882
k_CH3CO3_NO2 = k_3rd(temp_loc, cair_loc, (9.700000000000000248e-29), (5.599999999999999645), (9.29999999999999955e-12), (1.5), (0.5999999999999999778)); 
# 3883
k_PAN_M = (k_CH3CO3_NO2 / ((8.999999999999999629e-29) * exp((14000.0) / temp_loc))); 
# 3884
k_ClO_ClO = k_3rd_iupac(temp_loc, cair_loc, (2.000000000000000112e-32), (4.0), (9.999999999999999395e-12), (0.0), (0.4500000000000000111)); 
# 3885
k_BrO_NO2 = k_3rd_iupac(temp_loc, cair_loc, (4.699999999999999866e-31), (3.100000000000000089), (1.799999999999999923e-11), (0.0), (0.4000000000000000222)); 
# 3886
k_I_NO2 = k_3rd_iupac(temp_loc, cair_loc, (2.999999999999999812e-31), (1.0), (6.600000000000000473e-11), (0.0), (0.6300000000000000044)); 
# 3887
k_DMS_OH = ((((1.000000000000000062e-09) * exp((5820.0) / temp_loc)) * (var[73])) / ((1.00000000000000002e+30) + (((5.0) * exp((6280.0) / temp_loc)) * (var[73])))); 
# 3888
G7402a_yield = ((0.8000000000000000444) / (1.100000000000000089)); 
# 3889
k_O3s = (((((1.700000000000000026e-12) * exp((-(940.0)) / temp_loc)) * (var[65])) + (((9.999999999999999988e-15) * exp((-(490.0)) / temp_loc)) * (var[71]))) + ((((jx[(2 * VL_GLO) + index]) * (2.199999999999999899e-10)) * (var[47])) / (((((3.199999999999999936e-11) * exp((70.0) / temp_loc)) * (var[73])) + (((1.799999999999999923e-11) * exp((110.0) / temp_loc)) * (var[-1]))) + ((2.199999999999999899e-10) * (var[47]))))); 
# 3890
beta_null_CH3NO3 = ((0.002949999999999999931) + (((5.149999999999999627e-22) * cair_loc) * pow(temp_loc / (298), (7.400000000000000355)))); 
# 3891
beta_inf_CH3NO3 = (0.02199999999999999872); 
# 3892
beta_CH3NO3 = ((beta_null_CH3NO3 * beta_inf_CH3NO3) / (beta_null_CH3NO3 + beta_inf_CH3NO3)); 
# 3893
k_NO2_CH3O2 = k_3rd(temp_loc, cair_loc, (1.000000000000000083e-30), (4.799999999999999822), (7.200000000000000017e-12), (2.100000000000000089), (0.5999999999999999778)); 
# 3894
k_G4138 = (4.249999999999999864e-12); 
# 3895
k_G9408 = (3.660000000000000157e-11); 
# 3896
KRO2NO = ((2.540000000000000082e-12) * exp((360.0) / temp_loc)); 
# 3897
KRO2HO2 = ((2.910000000000000222e-13) * exp((1300.0) / temp_loc)); 
# 3898
KAPHO2 = ((4.299999999999999853e-13) * exp((1040.0) / temp_loc)); 
# 3899
KAPNO = ((8.099999999999999817e-12) * exp((270.0) / temp_loc)); 
# 3900
KRO2NO3 = (2.499999999999999849e-12); 
# 3901
KNO3AL = ((1.400000000000000093e-12) * exp((-(1900.0)) / temp_loc)); 
# 3902
J_IC3H7NO3 = ((3.700000000000000178) * (jx[(10 * VL_GLO) + index])); 
# 3903
J_ACETOL = (((0.6500000000000000222) * (0.1100000000000000006)) * (jx[(14 * VL_GLO) + index])); 
# 3904
RO2 = (0.0); 
# 3905
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3906
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3907
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3908
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3909
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3910
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3911
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3912
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3913
if (70 > 0) { RO2 = (RO2 + (var[70])); }  
# 3914
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3915
if (3 > 0) { RO2 = (RO2 + (var[3])); }  
# 3916
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3917
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3918
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3919
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3920
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3921
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3922
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3923
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3924
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3925
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3926
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3927
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3928
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3929
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3930
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3931
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3932
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3933
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3934
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3935
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3936
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3937
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3938
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3939
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3940
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3941
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3942
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3943
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3944
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3945
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3946
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3947
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3948
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3949
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3950
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3951
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3952
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3953
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3954
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3955
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3956
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3957
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3958
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3959
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3960
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3961
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3962
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3963
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3964
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3965
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3966
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3967
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3968
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3969
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3970
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3971
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3972
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3973
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3974
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3975
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3976
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3977
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3978
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3979
if ((-1) > 0) { RO2 = (RO2 + (var[-1])); }  
# 3981
(rconst[0]) = ((3.300000000000000237e-11) * exp((55.0) / temp_loc)); 
# 3982
(rconst[1]) = (((5.999999999999999994e-34) * pow(temp_loc / (300.0), -(2.399999999999999911))) * cair_loc); 
# 3983
(rconst[2]) = k_3rd(temp_loc, cair_loc, (4.399999999999999973e-32), (1.300000000000000044), (7.49999999999999995e-11), -(0.2000000000000000111), (0.5999999999999999778)); 
# 3984
(rconst[3]) = ((1.700000000000000026e-12) * exp((-(940.0)) / temp_loc)); 
# 3985
(rconst[4]) = ((2.800000000000000186e-12) * exp((-(1800.0)) / temp_loc)); 
# 3986
(rconst[5]) = ((9.999999999999999988e-15) * exp((-(490.0)) / temp_loc)); 
# 3987
(rconst[6]) = ((4.800000000000000227e-11) * exp((250.0) / temp_loc)); 
# 3988
(rconst[7]) = k_HO2_HO2; 
# 3989
(rconst[8]) = ((1.630000000000000058e-10) * exp((60.0) / temp_loc)); 
# 3990
(rconst[10]) = (((((6.521000000000000363e-26) * temp_loc) * exp((1851.089999999999918) / temp_loc)) * exp((-(0.005104850000000000186)) * temp_loc)) * (1000000.0)); 
# 3991
(rconst[12]) = ((2.800000000000000024e-11) * exp((-(250.0)) / temp_loc)); 
# 3992
(rconst[13]) = ((9.999999999999999799e-13) * exp((-(1590.0)) / temp_loc)); 
# 3993
(rconst[14]) = ((2.99999999999999998e-11) * exp((-(2450.0)) / temp_loc)); 
# 3994
(rconst[15]) = ((3.500000000000000233e-13) * exp((-(1370.0)) / temp_loc)); 
# 3995
(rconst[16]) = k_ClO_ClO; 
# 3996
(rconst[17]) = (k_ClO_ClO / ((1.720000000000000009e-27) * exp((8649.0) / temp_loc))); 
# 3997
(rconst[18]) = ((1.100000000000000079e-11) * exp((-(980.0)) / temp_loc)); 
# 3998
(rconst[19]) = ((2.199999999999999915e-12) * exp((340.0) / temp_loc)); 
# 3999
(rconst[20]) = ((1.700000000000000026e-12) * exp((-(230.0)) / temp_loc)); 
# 4000
(rconst[21]) = ((6.200000000000000239e-12) * exp((295.0) / temp_loc)); 
# 4001
(rconst[22]) = k_3rd_iupac(temp_loc, cair_loc, (1.60000000000000009e-31), (3.399999999999999911), (7.000000000000000384e-11), (0.0), (0.4000000000000000222)); 
# 4002
(rconst[23]) = (((6.917999999999999875e-07) * exp((-(10909.0)) / temp_loc)) * cair_loc); 
# 4003
(rconst[24]) = ((6.200000000000000239e-12) * exp((145.0) / temp_loc)); 
# 4004
(rconst[25]) = ((6.60000000000000015e-12) * exp((-(1240.0)) / temp_loc)); 
# 4005
(rconst[26]) = ((8.100000000000000463e-11) * exp((-(34.0)) / temp_loc)); 
# 4006
(rconst[28]) = ((3.300000000000000075e-12) * exp((-(115.0)) / temp_loc)); 
# 4007
(rconst[29]) = ((1.64000000000000008e-12) * exp((-(1520.0)) / temp_loc)); 
# 4008
(rconst[30]) = k_3rd_iupac(temp_loc, cair_loc, (1.849999999999999874e-29), (3.299999999999999822), (5.99999999999999996e-10), (0.0), (0.4000000000000000222)); 
# 4009
(rconst[32]) = k_3rd_iupac(temp_loc, cair_loc, (6.099999999999999808e-30), (3.0), (2.000000000000000073e-10), (0.0), (0.5999999999999999778)); 
# 4010
(rconst[33]) = ((8.299999999999999772e-11) * exp((-(100.0)) / temp_loc)); 
# 4011
(rconst[34]) = ((1.699999999999999946e-11) * exp((-(800.0)) / temp_loc)); 
# 4012
(rconst[36]) = ((2.900000000000000265e-14) * exp((840.0) / temp_loc)); 
# 4013
(rconst[37]) = ((7.699999999999999906e-12) * exp((-(450.0)) / temp_loc)); 
# 4014
(rconst[38]) = ((4.499999999999999809e-12) * exp((500.0) / temp_loc)); 
# 4015
(rconst[39]) = ((6.700000000000000128e-12) * exp((155.0) / temp_loc)); 
# 4016
(rconst[40]) = ((1.999999999999999879e-11) * exp((240.0) / temp_loc)); 
# 4017
(rconst[42]) = ((8.699999999999999684e-12) * exp((260.0) / temp_loc)); 
# 4018
(rconst[43]) = k_BrO_NO2; 
# 4019
(rconst[44]) = (k_BrO_NO2 / ((((((5.44000000000000024e-09) * exp((14192.0) / temp_loc)) * (1000000.0)) * (8.314462100000000078)) * temp_loc) / ((101325.0) * (6.022141289999999686e+23)))); 
# 4020
(rconst[45]) = ((7.699999999999999906e-12) * exp((-(580.0)) / temp_loc)); 
# 4021
(rconst[46]) = ((2.599999999999999827e-12) * exp((-(1600.0)) / temp_loc)); 
# 4022
(rconst[47]) = (G7402a_yield * (5.70000000000000035e-12)); 
# 4023
(rconst[48]) = (((1.0) - G7402a_yield) * (5.70000000000000035e-12)); 
# 4024
(rconst[49]) = ((2.349999999999999882e-12) * exp((-(1300.0)) / temp_loc)); 
# 4025
(rconst[50]) = (((2.800000000000000186e-13) * exp((224.0) / temp_loc)) / ((1.0) + (((1.130000000000000029e+24) * exp((-(3200.0)) / temp_loc)) / (var[73])))); 
# 4026
(rconst[51]) = ((1.799999999999999923e-11) * exp((-(460.0)) / temp_loc)); 
# 4027
(rconst[52]) = ((6.350000000000000237e-15) * exp((440.0) / temp_loc)); 
# 4028
(rconst[53]) = ((1.349999999999999902e-12) * exp((-(600.0)) / temp_loc)); 
# 4029
(rconst[54]) = ((1.99999999999999996e-12) * exp((-(840.0)) / temp_loc)); 
# 4030
(rconst[57]) = ((2.300000000000000071e-10) * exp((135.0) / temp_loc)); 
# 4031
(rconst[58]) = ((1.600000000000000049e-12) * exp((430.0) / temp_loc)); 
# 4032
(rconst[59]) = ((2.900000000000000164e-12) * exp((220.0) / temp_loc)); 
# 4033
(rconst[60]) = ((5.79999999999999952e-13) * exp((170.0) / temp_loc)); 
# 4034
(rconst[62]) = ((1.99999999999999996e-12) * exp((-(840.0)) / temp_loc)); 
# 4035
(rconst[63]) = ((1.99999999999999996e-12) * exp((-(840.0)) / temp_loc)); 
# 4036
(rconst[64]) = ((2.399999999999999871e-12) * exp((-(920.0)) / temp_loc)); 
# 4037
(rconst[65]) = k_3rd(temp_loc, cair_loc, (3.299999999999999925e-31), (4.299999999999999822), (1.600000000000000049e-12), (0.0), (0.5999999999999999778)); 
# 4038
(rconst[66]) = ((1.130000000000000072e-11) * exp((-(253.0)) / temp_loc)); 
# 4039
(rconst[67]) = k_DMS_OH; 
# 4040
(rconst[68]) = ((1.899999999999999982e-13) * exp((520.0) / temp_loc)); 
# 4041
(rconst[70]) = ((18000000000000.0) * exp((-(8661.0)) / temp_loc)); 
# 4042
(rconst[74]) = ((8.99999999999999994e-11) * exp((-(2386.0)) / temp_loc)); 
# 4043
(rconst[76]) = (khet_tr[(0 * VL_GLO) + index]); 
# 4044
(rconst[10 - 1]) = (1.800000000000000004e-12); 
# 4045
(rconst[12 - 1]) = (1000000.0); 
# 4046
(rconst[28 - 1]) = (5.900000000000000305e-11); 
# 4047
(rconst[32 - 1]) = (7.999999999999999516e-11); 
# 4048
(rconst[36 - 1]) = (2.699999999999999804e-12); 
# 4049
(rconst[42 - 1]) = (4.899999999999999881e-11); 
# 4050
(rconst[56 - 1]) = (3.319999999999999911e-15); 
# 4051
(rconst[57 - 1]) = (1.099999999999999928e-15); 
# 4052
(rconst[62 - 1]) = (1.450000000000000001e-11); 
# 4053
(rconst[70 - 1]) = (1.000000000000000036e-10); 
# 4054
(rconst[72 - 1]) = (2.999999999999999839e-13); 
# 4055
(rconst[73 - 1]) = (5.000000000000000182e-11); 
# 4056
(rconst[74 - 1]) = (3.299999999999999978e-10); 
# 4057
(rconst[76 - 1]) = (4.399999999999999932e-13); 
# 4058
} 
# 4059
} 
#endif
# 4063 "messy_mecca_kpp_acc.cu"
void Rosenbrock(double *__restrict__ conc, const double Tstart, const double Tend, double *__restrict__ rstatus, int *__restrict__ istatus, const int 
# 4065
autonomous, const int vectorTol, const int UplimTol, const int method, const int Max_no_steps, double *__restrict__ 
# 4066
d_jac0, double *__restrict__ d_Ghimj, double *__restrict__ d_varNew, double *__restrict__ d_K, double *__restrict__ d_varErr, double *__restrict__ d_dFdT, double *__restrict__ d_Fcn0, double *__restrict__ d_var, double *__restrict__ d_fix, double *__restrict__ d_rconst, const double 
# 4067
Hmin, const double Hmax, const double Hstart, const double FacMin, const double FacMax, const double FacRej, const double FacSafe, const double roundoff, const double *__restrict__ 
# 4069
absTol, const double *__restrict__ relTol, const double *__restrict__ 
# 4071
khet_st, const double *__restrict__ khet_tr, const double *__restrict__ 
# 4072
jx, const double *__restrict__ 
# 4074
temp_gpu, const double *__restrict__ 
# 4075
press_gpu, const double *__restrict__ 
# 4076
cair_gpu, const int 
# 4078
VL_GLO) ;
#if 0
# 4079
{ 
# 4080
int index = ((__device_builtin_variable_blockIdx.x) * (__device_builtin_variable_blockDim.x)) + (__device_builtin_variable_threadIdx.x); 
# 4089 "messy_mecca_kpp_acc.cu"
double *Ghimj = &(d_Ghimj[index * 455]); 
# 4090
double *K = &(d_K[(index * 73) * 6]); 
# 4091
double *varNew = &(d_varNew[index * 73]); 
# 4092
double *Fcn0 = &(d_Fcn0[index * 73]); 
# 4093
double *dFdT = &(d_dFdT[index * 73]); 
# 4094
double *jac0 = &(d_jac0[index * 455]); 
# 4095
double *varErr = &(d_varErr[index * 73]); 
# 4096
double *var = &(d_var[index * 74]); 
# 4097
double *fix = &(d_fix[index * 1]); 
# 4098
double *rconst = &(d_rconst[index * 77]); 
# 4102
if (index < VL_GLO) 
# 4103
{ 
# 4105
int Nfun, Njac, Nstp, Nacc, Nrej, Ndec, Nsol, Nsng; 
# 4106
double Texit, Hexit; 
# 4108
Nfun = 0; 
# 4109
Njac = 0; 
# 4110
Nstp = 0; 
# 4111
Nacc = 0; 
# 4112
Nrej = 0; 
# 4113
Ndec = 0; 
# 4114
Nsol = 0; 
# 4115
Nsng = 0; 
# 4118
const double *ros_A = &(((ros[method - 1]).ros_A)[0]); 
# 4119
const double *ros_C = &(((ros[method - 1]).ros_C)[0]); 
# 4120
const double *ros_M = &(((ros[method - 1]).ros_M)[0]); 
# 4121
const double *ros_E = &(((ros[method - 1]).ros_E)[0]); 
# 4122
const double *ros_Alpha = &(((ros[method - 1]).ros_Alpha)[0]); 
# 4123
const double *ros_Gamma = &(((ros[method - 1]).ros_Gamma)[0]); 
# 4124
const int *ros_NewF = &(((ros[method - 1]).ros_NewF)[0]); 
# 4125
const int ros_S = (ros[method - 1]).ros_S; 
# 4126
const double ros_ELO = (ros[method - 1]).ros_ELO; 
# 4140 "messy_mecca_kpp_acc.cu"
for (int i = 0; i < 74; i++) { 
# 4141
(var[i]) = (conc[(i * VL_GLO) + index]); }  
# 4143
for (int i = 0; i < 1; i++) { 
# 4144
(fix[i]) = (conc[((73 + i) * VL_GLO) + index]); }  
# 4147
update_rconst(var, khet_st, khet_tr, jx, rconst, temp_gpu, press_gpu, cair_gpu, VL_GLO); 
# 4149
ros_Integrator(var, fix, Tstart, Tend, Texit, ros_S, ros_M, ros_E, ros_A, ros_C, ros_Alpha, ros_Gamma, ros_ELO, ros_NewF, autonomous, vectorTol, Max_no_steps, roundoff, Hmin, Hmax, Hstart, Hexit, FacMin, FacMax, FacRej, FacSafe, Nfun, Njac, Nstp, Nacc, Nrej, Ndec, Nsol, Nsng, rconst, absTol, relTol, varNew, Fcn0, K, dFdT, jac0, Ghimj, varErr, khet_st, khet_tr, jx, VL_GLO); 
# 4167
for (int i = 0; i < 73; i++) { 
# 4168
(conc[(i * VL_GLO) + index]) = (var[i]); }  
# 4172
(istatus[(0 * VL_GLO) + index]) = Nfun; 
# 4173
(istatus[(1 * VL_GLO) + index]) = Njac; 
# 4174
(istatus[(2 * VL_GLO) + index]) = Nstp; 
# 4175
(istatus[(3 * VL_GLO) + index]) = Nacc; 
# 4176
(istatus[(4 * VL_GLO) + index]) = Nrej; 
# 4177
(istatus[(5 * VL_GLO) + index]) = Ndec; 
# 4178
(istatus[(6 * VL_GLO) + index]) = Nsol; 
# 4179
(istatus[(7 * VL_GLO) + index]) = Nsng; 
# 4181
(rstatus[(0 * VL_GLO) + index]) = Texit; 
# 4182
(rstatus[(1 * VL_GLO) + index]) = Hexit; 
# 4183
}  
# 4184
} 
#endif
# 4190 "messy_mecca_kpp_acc.cu"
void reduce_istatus_1(int *istatus, int4 *tmp_out_1, int4 *tmp_out_2, int VL_GLO, int *xNacc, int *xNrej) ;
#if 0
# 4191
{ 
# 4192
int index = ((__device_builtin_variable_blockIdx.x) * (__device_builtin_variable_blockDim.x)) + (__device_builtin_variable_threadIdx.x); 
# 4193
int idx_1 = __device_builtin_variable_threadIdx.x; 
# 4194
int global_size = (__device_builtin_variable_blockDim.x) * (__device_builtin_variable_gridDim.x); 
# 4196
int foo; 
# 4198
int4 accumulator_1 = make_int4(0, 0, 0, 0); 
# 4199
int4 accumulator_2 = make_int4(0, 0, 0, 0); 
# 4200
while (index < VL_GLO) 
# 4201
{ 
# 4202
(accumulator_1.x) += (istatus[(0 * VL_GLO) + index]); 
# 4203
(accumulator_1.y) += (istatus[(1 * VL_GLO) + index]); 
# 4204
(accumulator_1.z) += (istatus[(2 * VL_GLO) + index]); 
# 4206
foo = (istatus[(3 * VL_GLO) + index]); 
# 4207
(xNacc[index]) = foo; 
# 4208
(accumulator_1.w) += foo; 
# 4209
foo = (istatus[(4 * VL_GLO) + index]); 
# 4210
(xNrej[index]) = foo; 
# 4211
(accumulator_2.x) += foo; 
# 4212
(accumulator_2.y) += (istatus[(5 * VL_GLO) + index]); 
# 4213
(accumulator_2.z) += (istatus[(6 * VL_GLO) + index]); 
# 4214
(accumulator_2.w) += (istatus[(7 * VL_GLO) + index]); 
# 4215
index += global_size; 
# 4216
}  
# 4218
__attribute__((unused)) static int4 buffer_1[64]; 
# 4219
__attribute__((unused)) static int4 buffer_2[64]; 
# 4221
(buffer_1[idx_1]) = accumulator_1; 
# 4222
(buffer_2[idx_1]) = accumulator_2; 
# 4223
__syncthreads(); 
# 4225
int idx_2, active_threads = __device_builtin_variable_blockDim.x; 
# 4226
int4 tmp_1, tmp_2; 
# 4227
while (active_threads != 1) 
# 4228
{ 
# 4229
active_threads /= 2; 
# 4230
if (idx_1 < active_threads) 
# 4231
{ 
# 4232
idx_2 = (idx_1 + active_threads); 
# 4234
tmp_1 = (buffer_1[idx_1]); 
# 4235
tmp_2 = (buffer_1[idx_2]); 
# 4237
(tmp_1.x) += (tmp_2.x); 
# 4238
(tmp_1.y) += (tmp_2.y); 
# 4239
(tmp_1.z) += (tmp_2.z); 
# 4240
(tmp_1.w) += (tmp_2.w); 
# 4242
(buffer_1[idx_1]) = tmp_1; 
# 4245
tmp_1 = (buffer_2[idx_1]); 
# 4246
tmp_2 = (buffer_2[idx_2]); 
# 4248
(tmp_1.x) += (tmp_2.x); 
# 4249
(tmp_1.y) += (tmp_2.y); 
# 4250
(tmp_1.z) += (tmp_2.z); 
# 4251
(tmp_1.w) += (tmp_2.w); 
# 4253
(buffer_2[idx_1]) = tmp_1; 
# 4255
}  
# 4256
__syncthreads(); 
# 4257
}  
# 4258
if (idx_1 == 0) 
# 4259
{ 
# 4260
(tmp_out_1[__device_builtin_variable_blockIdx.x]) = (buffer_1[0]); 
# 4261
(tmp_out_2[__device_builtin_variable_blockIdx.x]) = (buffer_2[0]); 
# 4262
}  
# 4263
} 
#endif
# 4265 "messy_mecca_kpp_acc.cu"
void reduce_istatus_2(int4 *tmp_out_1, int4 *tmp_out_2, int *out) ;
#if 0
# 4266
{ 
# 4267
int idx_1 = __device_builtin_variable_threadIdx.x; 
# 4269
__attribute__((unused)) static int4 buffer_1[32]; 
# 4270
__attribute__((unused)) static int4 buffer_2[32]; 
# 4272
(buffer_1[idx_1]) = (tmp_out_1[idx_1]); 
# 4273
(buffer_2[idx_1]) = (tmp_out_2[idx_1]); 
# 4274
__syncthreads(); 
# 4276
int idx_2, active_threads = __device_builtin_variable_blockDim.x; 
# 4277
int4 tmp_1, tmp_2; 
# 4278
while (active_threads != 1) 
# 4279
{ 
# 4280
active_threads /= 2; 
# 4281
if (idx_1 < active_threads) 
# 4282
{ 
# 4283
idx_2 = (idx_1 + active_threads); 
# 4285
tmp_1 = (buffer_1[idx_1]); 
# 4286
tmp_2 = (buffer_1[idx_2]); 
# 4288
(tmp_1.x) += (tmp_2.x); 
# 4289
(tmp_1.y) += (tmp_2.y); 
# 4290
(tmp_1.z) += (tmp_2.z); 
# 4291
(tmp_1.w) += (tmp_2.w); 
# 4293
(buffer_1[idx_1]) = tmp_1; 
# 4296
tmp_1 = (buffer_2[idx_1]); 
# 4297
tmp_2 = (buffer_2[idx_2]); 
# 4299
(tmp_1.x) += (tmp_2.x); 
# 4300
(tmp_1.y) += (tmp_2.y); 
# 4301
(tmp_1.z) += (tmp_2.z); 
# 4302
(tmp_1.w) += (tmp_2.w); 
# 4304
(buffer_2[idx_1]) = tmp_1; 
# 4306
}  
# 4307
__syncthreads(); 
# 4308
}  
# 4309
if (idx_1 == 0) 
# 4310
{ 
# 4311
tmp_1 = (buffer_1[0]); 
# 4312
tmp_2 = (buffer_2[0]); 
# 4313
(out[0]) = (tmp_1.x); 
# 4314
(out[1]) = (tmp_1.y); 
# 4315
(out[2]) = (tmp_1.z); 
# 4316
(out[3]) = (tmp_1.w); 
# 4317
(out[4]) = (tmp_2.x); 
# 4318
(out[5]) = (tmp_2.y); 
# 4319
(out[6]) = (tmp_2.z); 
# 4320
(out[7]) = (tmp_2.w); 
# 4321
}  
# 4322
} 
#endif
# 4325 "messy_mecca_kpp_acc.cu"
enum { TRUE = 1, FALSE = 0}; 
# 4326
double *d_conc, *d_temp, *d_press, *d_cair, *d_khet_st, *d_khet_tr, *d_jx, *d_jac0, *d_Ghimj, *d_varNew, *d_K, *d_varErr, *d_dFdT, *d_Fcn0, *d_var, *d_fix, *d_rconst; 
# 4327
int initialized = (FALSE); 
# 4330
double *d_rstatus, *d_absTol, *d_relTol; 
# 4331
int *d_istatus, *d_istatus_rd, *d_xNacc, *d_xNrej; 
# 4332
int4 *d_tmp_out_1, *d_tmp_out_2; 
# 4335
void init_first_time(int pe, int VL_GLO, int size_khet_st, int size_khet_tr, int size_jx) { 
# 4338
int deviceCount, device; 
# 4339
cudaGetDeviceCount(&deviceCount); 
# 4340
device = (pe % deviceCount); 
# 4341
cudaSetDevice(device); 
# 4343
printf("PE[%d]: selected %d of total %d\n", pe, device, deviceCount); 
# 4344
cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); 
# 4346
cudaMalloc((void **)(&d_conc), (sizeof(double) * VL_GLO) * (74)); 
# 4347
cudaMalloc((void **)(&d_khet_st), (sizeof(double) * VL_GLO) * size_khet_st); 
# 4348
cudaMalloc((void **)(&d_khet_tr), (sizeof(double) * VL_GLO) * size_khet_tr); 
# 4349
cudaMalloc((void **)(&d_jx), (sizeof(double) * VL_GLO) * size_jx); 
# 4351
cudaMalloc((void **)(&d_rstatus), (sizeof(double) * VL_GLO) * (2)); 
# 4352
cudaMalloc((void **)(&d_istatus), (sizeof(int) * VL_GLO) * (8)); 
# 4353
cudaMalloc((void **)(&d_absTol), sizeof(double) * (73)); 
# 4354
cudaMalloc((void **)(&d_relTol), sizeof(double) * (73)); 
# 4357
cudaMalloc((void **)(&temp_gpu), sizeof(double) * VL_GLO); 
# 4358
cudaMalloc((void **)(&press_gpu), sizeof(double) * VL_GLO); 
# 4359
cudaMalloc((void **)(&cair_gpu), sizeof(double) * VL_GLO); 
# 4362
cudaMalloc((void **)(&d_istatus_rd), sizeof(int) * (8)); 
# 4363
cudaMalloc((void **)(&d_tmp_out_1), sizeof(int4) * (64)); 
# 4364
cudaMalloc((void **)(&d_tmp_out_2), sizeof(int4) * (64)); 
# 4365
cudaMalloc((void **)(&d_xNacc), sizeof(int) * VL_GLO); 
# 4366
cudaMalloc((void **)(&d_xNrej), sizeof(int) * VL_GLO); 
# 4369
cudaMalloc((void **)(&d_jac0), (sizeof(double) * VL_GLO) * (455)); 
# 4370
cudaMalloc((void **)(&d_Ghimj), (sizeof(double) * VL_GLO) * (455)); 
# 4371
cudaMalloc((void **)(&d_varNew), (sizeof(double) * VL_GLO) * (73)); 
# 4372
cudaMalloc((void **)(&d_Fcn0), (sizeof(double) * VL_GLO) * (73)); 
# 4373
cudaMalloc((void **)(&d_dFdT), (sizeof(double) * VL_GLO) * (73)); 
# 4375
cudaMalloc((void **)(&d_K), ((sizeof(double) * VL_GLO) * (73)) * (6)); 
# 4376
cudaMalloc((void **)(&d_varErr), (sizeof(double) * VL_GLO) * (73)); 
# 4377
cudaMalloc((void **)(&d_var), (sizeof(double) * VL_GLO) * (74)); 
# 4378
cudaMalloc((void **)(&d_fix), (sizeof(double) * VL_GLO) * (1)); 
# 4379
cudaMalloc((void **)(&d_rconst), (sizeof(double) * VL_GLO) * (77)); 
# 4381
initialized = (TRUE); 
# 4382
} 
# 4387
extern "C" void finalize_cuda() { 
# 4389
cudaFree(d_conc); 
# 4390
cudaFree(d_temp); 
# 4391
cudaFree(d_press); 
# 4392
cudaFree(d_cair); 
# 4393
cudaFree(d_khet_st); 
# 4394
cudaFree(d_khet_tr); 
# 4395
cudaFree(d_jx); 
# 4396
cudaFree(d_rstatus); 
# 4397
cudaFree(d_istatus); 
# 4398
cudaFree(d_absTol); 
# 4399
cudaFree(d_relTol); 
# 4400
cudaFree(d_istatus_rd); 
# 4401
cudaFree(d_tmp_out_1); 
# 4402
cudaFree(d_tmp_out_2); 
# 4403
cudaFree(d_xNacc); 
# 4404
cudaFree(d_xNrej); 
# 4405
cudaFree(temp_gpu); 
# 4406
cudaFree(press_gpu); 
# 4407
cudaFree(cair_gpu); 
# 4409
cudaFree(d_jac0); 
# 4410
cudaFree(d_Ghimj); 
# 4411
cudaFree(d_varNew); 
# 4412
cudaFree(d_Fcn0); 
# 4413
cudaFree(d_dFdT); 
# 4414
cudaFree(d_K); 
# 4415
cudaFree(d_varErr); 
# 4416
cudaFree(d_var); 
# 4417
cudaFree(d_fix); 
# 4418
cudaFree(d_rconst); 
# 4420
} 
# 4424
extern "C" void kpp_integrate_cuda_(int *pe_p, int *sizes, double *time_step_len_p, double *conc, double *temp, double *press, double *cair, double *
# 4425
khet_st, double *khet_tr, double *jx, double *absTol, double *relTol, int *ierr, int *istatus, int *
# 4426
xNacc, int *xNrej, double *rndoff, int *icntrl = 0, double *rcntrl = 0) 
# 4453 "messy_mecca_kpp_acc.cu"
{ 
# 4455
const double DELTAMIN = (1.000000000000000082e-05); 
# 4459
int VL_GLO = sizes[0]; 
# 4460
int size_khet_st = sizes[1]; 
# 4461
int size_khet_tr = sizes[2]; 
# 4462
int size_jx = sizes[3]; 
# 4463
double roundoff = *rndoff; 
# 4465
double Tstart, Tend; 
# 4466
Tstart = (0.0); 
# 4467
Tend = (*time_step_len_p); 
# 4468
int pe = *pe_p; 
# 4471
int autonomous, vectorTol, UplimTol, method, Max_no_steps; 
# 4472
double Hmin, Hmax, Hstart, FacMin, FacMax, FacRej, FacSafe; 
# 4475
if (rcntrl == (__null)) 
# 4476
{ 
# 4477
rcntrl = (new double [7]); 
# 4478
for (int i = 0; i < 7; i++) { 
# 4479
(rcntrl[i]) = (0.0); }  
# 4480
}  
# 4481
if (icntrl == (__null)) 
# 4482
{ 
# 4483
icntrl = (new int [4]); 
# 4484
for (int i = 0; i < 4; i++) { 
# 4485
(icntrl[i]) = 0; }  
# 4486
}  
# 4489
if (initialized == (FALSE)) { init_first_time(pe, VL_GLO, size_khet_st, size_khet_tr, size_jx); }  
# 4492
cudaMemcpy(d_conc, conc, (sizeof(double) * VL_GLO) * (74), cudaMemcpyHostToDevice); 
# 4494
cudaMemcpy(temp_gpu, temp, sizeof(double) * VL_GLO, cudaMemcpyHostToDevice); 
# 4495
cudaMemcpy(press_gpu, press, sizeof(double) * VL_GLO, cudaMemcpyHostToDevice); 
# 4496
cudaMemcpy(cair_gpu, cair, sizeof(double) * VL_GLO, cudaMemcpyHostToDevice); 
# 4498
cudaMemcpy(d_khet_st, khet_st, (sizeof(double) * VL_GLO) * size_khet_st, cudaMemcpyHostToDevice); 
# 4499
cudaMemcpy(d_khet_tr, khet_tr, (sizeof(double) * VL_GLO) * size_khet_tr, cudaMemcpyHostToDevice); 
# 4500
cudaMemcpy(d_jx, jx, (sizeof(double) * VL_GLO) * size_jx, cudaMemcpyHostToDevice); 
# 4503
cudaMemcpy(d_absTol, absTol, sizeof(double) * (73), cudaMemcpyHostToDevice); 
# 4504
cudaMemcpy(d_relTol, relTol, sizeof(double) * (73), cudaMemcpyHostToDevice); 
# 4508
int block_size, grid_size; 
# 4510
block_size = 64; 
# 4511
grid_size = (((VL_GLO + block_size) - 1) / block_size); 
# 4512
dim3 dimBlock(block_size); 
# 4513
dim3 dimGrid(grid_size); 
# 4519
; 
# 4524
int ierr_tmp = 0; 
# 4525
{ 
# 4527
autonomous = (!((icntrl[0]) == 0)); 
# 4531
if ((icntrl[1]) == 0) 
# 4532
{ 
# 4533
vectorTol = 1; 
# 4534
UplimTol = 73; 
# 4535
} else 
# 4537
{ 
# 4538
vectorTol = 0; 
# 4539
UplimTol = 1; 
# 4540
}  
# 4543
if ((icntrl[2]) == 0) 
# 4544
{ 
# 4545
method = 4; 
# 4546
} else { 
# 4547
if (((icntrl[2]) >= 1) && ((icntrl[2]) <= 5)) 
# 4548
{ 
# 4549
method = (icntrl[2]); 
# 4550
} else 
# 4552
{ 
# 4553
printf("User-selected Rosenbrock method: icntrl[2]=%d\n", method); 
# 4554
ierr_tmp = (-2); 
# 4555
}  }  
# 4557
if ((icntrl[3]) == 0) 
# 4558
{ 
# 4559
Max_no_steps = 100000; 
# 4560
} else { 
# 4561
if ((icntrl[3]) > 0) 
# 4562
{ 
# 4563
Max_no_steps = (icntrl[3]); 
# 4564
} else 
# 4566
{ 
# 4567
printf("User-selected max no. of steps: icntrl[3]=%d\n", icntrl[3]); 
# 4568
ierr_tmp = (-1); 
# 4569
}  }  
# 4571
roundoff = machine_eps_flt(); 
# 4574
if ((rcntrl[0]) == (0.0)) 
# 4575
{ 
# 4576
Hmin = (0.0); 
# 4577
} else { 
# 4578
if ((rcntrl[0]) > (0.0)) 
# 4579
{ 
# 4580
Hmin = (rcntrl[0]); 
# 4581
} else 
# 4583
{ 
# 4584
printf("User-selected Hmin: rcntrl[0]=%f\n", rcntrl[0]); 
# 4585
ierr_tmp = (-3); 
# 4586
}  }  
# 4588
if ((rcntrl[1]) == (0.0)) 
# 4589
{ 
# 4590
Hmax = fabs(Tend - Tstart); 
# 4591
} else { 
# 4592
if ((rcntrl[1]) > (0.0)) 
# 4593
{ 
# 4594
Hmax = fmin(fabs(rcntrl[1]), fabs(Tend - Tstart)); 
# 4595
} else 
# 4597
{ 
# 4598
printf("User-selected Hmax: rcntrl[1]=%f\n", rcntrl[1]); 
# 4599
ierr_tmp = (-3); 
# 4600
}  }  
# 4602
if ((rcntrl[2]) == (0.0)) 
# 4603
{ 
# 4604
Hstart = fmax(Hmin, DELTAMIN); 
# 4605
} else { 
# 4606
if ((rcntrl[2]) > (0.0)) 
# 4607
{ 
# 4608
Hstart = fmin(fabs(rcntrl[2]), fabs(Tend - Tstart)); 
# 4609
} else 
# 4611
{ 
# 4612
printf("User-selected Hstart: rcntrl[2]=%f\n", rcntrl[2]); 
# 4613
ierr_tmp = (-3); 
# 4614
}  }  
# 4616
if ((rcntrl[3]) == (0.0)) 
# 4617
{ 
# 4618
FacMin = (0.2000000000000000111); 
# 4619
} else { 
# 4620
if ((rcntrl[3]) > (0.0)) 
# 4621
{ 
# 4622
FacMin = (rcntrl[3]); 
# 4623
} else 
# 4625
{ 
# 4626
printf("User-selected FacMin: rcntrl[3]=%f\n", rcntrl[3]); 
# 4627
ierr_tmp = (-4); 
# 4628
}  }  
# 4629
if ((rcntrl[4]) == (0.0)) 
# 4630
{ 
# 4631
FacMax = (6.0); 
# 4632
} else { 
# 4633
if ((rcntrl[4]) > (0.0)) 
# 4634
{ 
# 4635
FacMax = (rcntrl[4]); 
# 4636
} else 
# 4638
{ 
# 4639
printf("User-selected FacMax: rcntrl[4]=%f\n", rcntrl[4]); 
# 4640
ierr_tmp = (-4); 
# 4641
}  }  
# 4643
if ((rcntrl[5]) == (0.0)) 
# 4644
{ 
# 4645
FacRej = (0.1000000000000000056); 
# 4646
} else { 
# 4647
if ((rcntrl[5]) > (0.0)) 
# 4648
{ 
# 4649
FacRej = (rcntrl[5]); 
# 4650
} else 
# 4652
{ 
# 4653
printf("User-selected FacRej: rcntrl[5]=%f\n", rcntrl[5]); 
# 4654
ierr_tmp = (-4); 
# 4655
}  }  
# 4657
if ((rcntrl[6]) == (0.0)) 
# 4658
{ 
# 4659
FacSafe = (0.9000000000000000222); 
# 4660
} else { 
# 4661
if ((rcntrl[6]) > (0.0)) 
# 4662
{ 
# 4663
FacSafe = (rcntrl[6]); 
# 4664
} else 
# 4666
{ 
# 4667
printf("User-selected FacSafe: rcntrl[6]=%f\n", rcntrl[6]); 
# 4668
ierr_tmp = (-4); 
# 4669
}  }  
# 4671
for (int i = 0; i < UplimTol; i++) 
# 4672
{ 
# 4673
if ((((absTol[i]) <= (0.0)) || ((relTol[i]) <= ((10.0) * roundoff))) || ((relTol[i]) >= (1.0))) 
# 4674
{ 
# 4675
printf("CCC absTol(%d) = %f \n", i, absTol[i]); 
# 4676
printf("CCC relTol(%d) = %f \n", i, relTol[i]); 
# 4677
ierr_tmp = (-5); 
# 4678
}  
# 4679
}  
# 4680
} 
# 4683
(__cudaPushCallConfiguration(dimGrid, dimBlock)) ? (void)0 : Rosenbrock(d_conc, Tstart, Tend, d_rstatus, d_istatus, autonomous, vectorTol, UplimTol, method, Max_no_steps, d_jac0, d_Ghimj, d_varNew, d_K, d_varErr, d_dFdT, d_Fcn0, d_var, d_fix, d_rconst, Hmin, Hmax, Hstart, FacMin, FacMax, FacRej, FacSafe, roundoff, d_absTol, d_relTol, d_khet_st, d_khet_tr, d_jx, temp_gpu, press_gpu, cair_gpu, VL_GLO); 
# 4695
; 
# 4698
(__cudaPushCallConfiguration(32, 64)) ? (void)0 : reduce_istatus_1(d_istatus, d_tmp_out_1, d_tmp_out_2, VL_GLO, d_xNacc, d_xNrej); 
# 4701
; 
# 4703
(__cudaPushCallConfiguration(1, 32)) ? (void)0 : reduce_istatus_2(d_tmp_out_1, d_tmp_out_2, d_istatus_rd); 
# 4705
; 
# 4708
cudaMemcpy(conc, d_conc, (sizeof(double) * VL_GLO) * (73), cudaMemcpyDeviceToHost); 
# 4709
cudaMemcpy(xNacc, d_xNacc, sizeof(int) * VL_GLO, cudaMemcpyDeviceToHost); 
# 4710
cudaMemcpy(xNrej, d_xNrej, sizeof(int) * VL_GLO, cudaMemcpyDeviceToHost); 
# 4715
} 
# 38 "/usr/include/sys/time.h" 3
extern "C" {
# 56 "/usr/include/sys/time.h" 3
struct timezone { 
# 58
int tz_minuteswest; 
# 59
int tz_dsttime; 
# 60
}; 
# 62
typedef struct timezone *__restrict__ __timezone_ptr_t; 
# 72 "/usr/include/sys/time.h" 3
extern int gettimeofday(timeval *__restrict__ __tv, __timezone_ptr_t __tz) throw()
# 73
 __attribute((__nonnull__(1))); 
# 78
extern int settimeofday(const timeval * __tv, const struct timezone * __tz) throw(); 
# 86
extern int adjtime(const timeval * __delta, timeval * __olddelta) throw(); 
# 92
enum __itimer_which { 
# 95
ITIMER_REAL, 
# 98
ITIMER_VIRTUAL, 
# 102
ITIMER_PROF
# 104
}; 
# 108
struct itimerval { 
# 111
timeval it_interval; 
# 113
timeval it_value; 
# 114
}; 
# 121
typedef int __itimer_which_t; 
# 126
extern int getitimer(__itimer_which_t __which, itimerval * __value) throw(); 
# 132
extern int setitimer(__itimer_which_t __which, const itimerval *__restrict__ __new, itimerval *__restrict__ __old) throw(); 
# 139
extern int utimes(const char * __file, const timeval  __tvp[2]) throw()
# 140
 __attribute((__nonnull__(1))); 
# 144
extern int lutimes(const char * __file, const timeval  __tvp[2]) throw()
# 145
 __attribute((__nonnull__(1))); 
# 148
extern int futimes(int __fd, const timeval  __tvp[2]) throw(); 
# 155
extern int futimesat(int __fd, const char * __file, const timeval  __tvp[2]) throw(); 
# 190 "/usr/include/sys/time.h" 3
}
# 4737 "messy_mecca_kpp_acc.cu"
double conc[5760 * 74]; 
# 4738
double temp[5760]; 
# 4739
double press[5760]; 
# 4740
double cair[5760]; 
# 4741
double jx[5760 * 74]; 
# 4744
int xNacc[5760]; 
# 4745
int xNrej[5760]; 
# 4747
double conc_cell[74] = {(0.0), (0.0), (1.130030837133365053e-06), (2161.176818259260017), (0.0001469481417859824128), (0.0002894067546497780248), (0.0), (0.0), (6.377486492629031622e-31), (0.0002774602114035594155), (9.159068418074057955e-22), (1.681545841334170886e-30), (6.587848965925120834e-36), (4.057130203198297654e-31), (7.556675262619906408e-06), (5.625822089563362005e-06), (7.248546508346979966e-10), (7.771754415762507499e-39), (1.672965892516880913e-32), (5.778276640099592545e-29), (2.169623196599309996e-31), (4.449685524913890094e-29), (9.236991853178720775e-28), (1.73125484793541291e-09), (6.419363370200839028e-28), (4.035724058634079279e-29), (6234.08726448301968), (25802.77881328489821), (1.33974252411334005), (11.15141769464590027), (8.023966161170008037e-32), (1.405402576145367075e-30), (2.416365419045455886e-29), (3.763980220765518692e-33), (0.0003687747273615521091), (4.4006958058575549e-30), (8.096351349854846844e-09), (1.605777396541510016e-08), (8.424266813161654464e-05), (1.275728897910597132e-29), (36780.60690670069744), (44.28021855848810162), (5.485594561042763652e-10), (3.418234885986840192e-32), (1.808885697309332159e-08), (2.295321288609201868e-30), (7.186736555958002736e-32), (667193926.5490679741), (9.443976722997097911e-30), (2.065479750965849984e-30), (658798139.7173529863), (5013220.829272099771), (6.594652607797343386e-13), (4.779051920325237223e-33), (0.2413303920517579915), (2.657031589287186106e-30), (1.166890334972386016e-14), (337.0697822316579959), (126494.9772056910006), (891.1969152016109774), (222.5573672438320045), (1.224516246698130084), (4845.027548231059882), (535329.6161963680061), (0.03077774956209535992), (989833722.9372060299), (38527.62914324420126), (1.857293910861109019e-07), (5035616002.440179825), (26824247.31079050153), (211466.2391751630057), (60638129767802.70312), (225227339137553.0), (87651408241.11650085)}; 
# 4824
double abstol[74] = {(0.0)}; 
# 4828
double reltol[74] = {(0.0)}; 
# 4833
double khet_st[5760 * 74] = {(0.0)}; 
# 4837
double khet_tr[5760 * 74] = {(0.0)}; 
# 4859 "messy_mecca_kpp_acc.cu"
int main(int argc, char **argv) { 
# 4861
int n = 1; 
# 4865
int istatus; 
# 4866
int ierr; 
# 4867
int i, j; 
# 4869
int sizes[4] = {5760, 74, 74, 115}; 
# 4870
int icntrl[20] = {0, 0, 2, 0}; 
# 4872
double roundoff = (2.220446049250313081e-16); 
# 4873
double timestep = (720.0); 
# 4875
for (i = 0; i < 5760; i++) { 
# 4876
for (j = 0; j < 74; j++) { 
# 4877
(conc[(i * 74) + j]) = (conc_cell[j]); 
# 4878
}  
# 4879
(temp[i]) = (240.9959719722450018); 
# 4880
(press[i]) = (0.9945912361145019531); 
# 4881
(cair[i]) = (298914285136738.0); 
# 4883
(khet_tr[(i * 4) + 0]) = (7.408371201503456121e-08); 
# 4884
(khet_tr[(i * 4) + 1]) = (4.849455570110967549e-07); 
# 4885
(khet_tr[(i * 4) + 2]) = (0.0); 
# 4886
(khet_tr[(i * 4) + 3]) = (2.718003287797324843e-07); 
# 4887
}  
# 4890
for (i = 0; i < 74; i++) { 
# 4891
(abstol[i]) = (10.0); 
# 4892
(reltol[i]) = (0.5); 
# 4893
}  
# 4896
cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); 
# 4898
for (i = 0; i < 5760; i++) { for (j = 0; j < 74; j++) { (conc[(j * 5760) + i]) = (conc_cell[j]); }  }  ; 
# 4900
kpp_integrate_cuda_(&n, sizes, &timestep, conc, temp, press, cair, khet_st, khet_tr, jx, abstol, reltol, &ierr, &istatus, xNacc, xNrej, &roundoff, icntrl); 
# 4909 "messy_mecca_kpp_acc.cu"
timeval start, end; 
# 4911
if (argc == 2) { 
# 4912
(icntrl[2]) = atoi(argv[1]); 
# 4913
for (i = 0; i < 5760; i++) { for (j = 0; j < 74; j++) { (conc[(j * 5760) + i]) = (conc_cell[j]); }  }  
# 4914
gettimeofday(&start, __null); 
# 4915
kpp_integrate_cuda_(&n, sizes, &timestep, conc, temp, press, cair, khet_st, khet_tr, jx, abstol, reltol, &ierr, &istatus, xNacc, xNrej, &roundoff, icntrl); 
# 4916
gettimeofday(&end, __null); 
# 4917
printf("%d: %ld (ms)\n", icntrl[2], (((end.tv_sec) * (1000)) + ((end.tv_usec) / (1000))) - (((start.tv_sec) * (1000)) + ((start.tv_usec) / (1000)))); 
# 4918
printf("Results:"); for (j = 0; j < 74; j++) { printf("   %.12e  ", conc[j * 5760]); }  printf("\n"); ; 
# 4920
return 0; 
# 4921
}  
# 4925
(icntrl[2]) = 1; 
# 4927
restart:; 
# 4931
for (i = 0; i < 5760; i++) { for (j = 0; j < 74; j++) { (conc[(j * 5760) + i]) = (conc_cell[j]); }  }  ; 
# 4932
gettimeofday(&start, __null); 
# 4934
kpp_integrate_cuda_(&n, sizes, &timestep, conc, temp, press, cair, khet_st, khet_tr, jx, abstol, reltol, &ierr, &istatus, xNacc, xNrej, &roundoff, icntrl); 
# 4936
gettimeofday(&end, __null); 
# 4938
printf("Results:"); for (j = 0; j < 74; j++) { printf("   %.12e  ", conc[j * 5760]); }  printf("\n"); ; 
# 4940
printf("%d: %ld (ms)\n", icntrl[2], (((end.tv_sec) * (1000)) + ((end.tv_usec) / (1000))) - (((start.tv_sec) * (1000)) + ((start.tv_usec) / (1000)))); 
# 4942
(icntrl[2])++; 
# 4943
if ((icntrl[2]) > 5) { return 0; }  
# 4944
goto restart; 
# 4950
} 

# 1 "messy_mecca_kpp_acc.cudafe1.stub.c"
#define _NV_ANON_NAMESPACE _GLOBAL__N__27_messy_mecca_kpp_acc_cpp1_ii_781dcaca
# 1 "messy_mecca_kpp_acc.cudafe1.stub.c"
#include "messy_mecca_kpp_acc.cudafe1.stub.c"
# 1 "messy_mecca_kpp_acc.cudafe1.stub.c"
#undef _NV_ANON_NAMESPACE
