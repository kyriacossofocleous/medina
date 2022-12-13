#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "messy_mecca_kpp_acc.fatbin.c"
extern void __device_stub__Z10RosenbrockPdddS_PiiiiiiS_S_S_S_S_S_S_S_S_S_ddddddddPKdS2_S2_S2_S2_S2_S2_S2_i(double *__restrict__, const double, const double, double *__restrict__, int *__restrict__, const int, const int, const int, const int, const int, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, const double, const double, const double, const double, const double, const double, const double, const double, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const int);
extern void __device_stub__Z16reduce_istatus_1PiP4int4S1_iS_S_(int *, struct int4 *, struct int4 *, int, int *, int *);
extern void __device_stub__Z16reduce_istatus_2P4int4S0_Pi(struct int4 *, struct int4 *, int *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z10RosenbrockPdddS_PiiiiiiS_S_S_S_S_S_S_S_S_S_ddddddddPKdS2_S2_S2_S2_S2_S2_S2_i(double *__restrict__ __par0, const double __par1, const double __par2, double *__restrict__ __par3, int *__restrict__ __par4, const int __par5, const int __par6, const int __par7, const int __par8, const int __par9, double *__restrict__ __par10, double *__restrict__ __par11, double *__restrict__ __par12, double *__restrict__ __par13, double *__restrict__ __par14, double *__restrict__ __par15, double *__restrict__ __par16, double *__restrict__ __par17, double *__restrict__ __par18, double *__restrict__ __par19, const double __par20, const double __par21, const double __par22, const double __par23, const double __par24, const double __par25, const double __par26, const double __par27, const double *__restrict__ __par28, const double *__restrict__ __par29, const double *__restrict__ __par30, const double *__restrict__ __par31, const double *__restrict__ __par32, const double *__restrict__ __par33, const double *__restrict__ __par34, const double *__restrict__ __par35, const int __par36){ double *__T4;
 double *__T5;
 int *__T6;
 double *__T7;
 double *__T8;
 double *__T9;
 double *__T10;
 double *__T11;
 double *__T12;
 double *__T13;
 double *__T14;
 double *__T15;
 double *__T16;
 const double *__T17;
 const double *__T18;
 const double *__T19;
 const double *__T20;
 const double *__T21;
 const double *__T22;
 const double *__T23;
 const double *__T24;
__cudaLaunchPrologue(37);__T4 = __par0;__cudaSetupArgSimple(__T4, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__T5 = __par3;__cudaSetupArgSimple(__T5, 24UL);__T6 = __par4;__cudaSetupArgSimple(__T6, 32UL);__cudaSetupArgSimple(__par5, 40UL);__cudaSetupArgSimple(__par6, 44UL);__cudaSetupArgSimple(__par7, 48UL);__cudaSetupArgSimple(__par8, 52UL);__cudaSetupArgSimple(__par9, 56UL);__T7 = __par10;__cudaSetupArgSimple(__T7, 64UL);__T8 = __par11;__cudaSetupArgSimple(__T8, 72UL);__T9 = __par12;__cudaSetupArgSimple(__T9, 80UL);__T10 = __par13;__cudaSetupArgSimple(__T10, 88UL);__T11 = __par14;__cudaSetupArgSimple(__T11, 96UL);__T12 = __par15;__cudaSetupArgSimple(__T12, 104UL);__T13 = __par16;__cudaSetupArgSimple(__T13, 112UL);__T14 = __par17;__cudaSetupArgSimple(__T14, 120UL);__T15 = __par18;__cudaSetupArgSimple(__T15, 128UL);__T16 = __par19;__cudaSetupArgSimple(__T16, 136UL);__cudaSetupArgSimple(__par20, 144UL);__cudaSetupArgSimple(__par21, 152UL);__cudaSetupArgSimple(__par22, 160UL);__cudaSetupArgSimple(__par23, 168UL);__cudaSetupArgSimple(__par24, 176UL);__cudaSetupArgSimple(__par25, 184UL);__cudaSetupArgSimple(__par26, 192UL);__cudaSetupArgSimple(__par27, 200UL);__T17 = __par28;__cudaSetupArgSimple(__T17, 208UL);__T18 = __par29;__cudaSetupArgSimple(__T18, 216UL);__T19 = __par30;__cudaSetupArgSimple(__T19, 224UL);__T20 = __par31;__cudaSetupArgSimple(__T20, 232UL);__T21 = __par32;__cudaSetupArgSimple(__T21, 240UL);__T22 = __par33;__cudaSetupArgSimple(__T22, 248UL);__T23 = __par34;__cudaSetupArgSimple(__T23, 256UL);__T24 = __par35;__cudaSetupArgSimple(__T24, 264UL);__cudaSetupArgSimple(__par36, 272UL);__cudaLaunch(((char *)((void ( *)(double *__restrict__, const double, const double, double *__restrict__, int *__restrict__, const int, const int, const int, const int, const int, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, const double, const double, const double, const double, const double, const double, const double, const double, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const int))Rosenbrock)));}
# 4063 "messy_mecca_kpp_acc.cu"
void Rosenbrock( double *__restrict__ __cuda_0,const double __cuda_1,const double __cuda_2,double *__restrict__ __cuda_3,int *__restrict__ __cuda_4,const int __cuda_5,const int __cuda_6,const int __cuda_7,const int __cuda_8,const int __cuda_9,double *__restrict__ __cuda_10,double *__restrict__ __cuda_11,double *__restrict__ __cuda_12,double *__restrict__ __cuda_13,double *__restrict__ __cuda_14,double *__restrict__ __cuda_15,double *__restrict__ __cuda_16,double *__restrict__ __cuda_17,double *__restrict__ __cuda_18,double *__restrict__ __cuda_19,const double __cuda_20,const double __cuda_21,const double __cuda_22,const double __cuda_23,const double __cuda_24,const double __cuda_25,const double __cuda_26,const double __cuda_27,const double *__restrict__ __cuda_28,const double *__restrict__ __cuda_29,const double *__restrict__ __cuda_30,const double *__restrict__ __cuda_31,const double *__restrict__ __cuda_32,const double *__restrict__ __cuda_33,const double *__restrict__ __cuda_34,const double *__restrict__ __cuda_35,const int __cuda_36)
# 4079 "messy_mecca_kpp_acc.cu"
{__device_stub__Z10RosenbrockPdddS_PiiiiiiS_S_S_S_S_S_S_S_S_S_ddddddddPKdS2_S2_S2_S2_S2_S2_S2_i( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10,__cuda_11,__cuda_12,__cuda_13,__cuda_14,__cuda_15,__cuda_16,__cuda_17,__cuda_18,__cuda_19,__cuda_20,__cuda_21,__cuda_22,__cuda_23,__cuda_24,__cuda_25,__cuda_26,__cuda_27,__cuda_28,__cuda_29,__cuda_30,__cuda_31,__cuda_32,__cuda_33,__cuda_34,__cuda_35,__cuda_36);
# 4184 "messy_mecca_kpp_acc.cu"
}
# 1 "./temp_files/messy_mecca_kpp_acc.cudafe1.stub.c"
void __device_stub__Z16reduce_istatus_1PiP4int4S1_iS_S_( int *__par0,  struct int4 *__par1,  struct int4 *__par2,  int __par3,  int *__par4,  int *__par5) {  __cudaLaunchPrologue(6); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 40UL); __cudaLaunch(((char *)((void ( *)(int *, struct int4 *, struct int4 *, int, int *, int *))reduce_istatus_1))); }
# 4190 "messy_mecca_kpp_acc.cu"
void reduce_istatus_1( int *__cuda_0,struct int4 *__cuda_1,struct int4 *__cuda_2,int __cuda_3,int *__cuda_4,int *__cuda_5)
# 4191 "messy_mecca_kpp_acc.cu"
{__device_stub__Z16reduce_istatus_1PiP4int4S1_iS_S_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 4263 "messy_mecca_kpp_acc.cu"
}
# 1 "./temp_files/messy_mecca_kpp_acc.cudafe1.stub.c"
void __device_stub__Z16reduce_istatus_2P4int4S0_Pi( struct int4 *__par0,  struct int4 *__par1,  int *__par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(struct int4 *, struct int4 *, int *))reduce_istatus_2))); }
# 4265 "messy_mecca_kpp_acc.cu"
void reduce_istatus_2( struct int4 *__cuda_0,struct int4 *__cuda_1,int *__cuda_2)
# 4266 "messy_mecca_kpp_acc.cu"
{__device_stub__Z16reduce_istatus_2P4int4S0_Pi( __cuda_0,__cuda_1,__cuda_2);
# 4322 "messy_mecca_kpp_acc.cu"
}
# 1 "./temp_files/messy_mecca_kpp_acc.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T27) {  __nv_dummy_param_ref(__T27); __nv_save_fatbinhandle_for_managed_rt(__T27); __cudaRegisterEntry(__T27, ((void ( *)(struct int4 *, struct int4 *, int *))reduce_istatus_2), _Z16reduce_istatus_2P4int4S0_Pi, (-1)); __cudaRegisterEntry(__T27, ((void ( *)(int *, struct int4 *, struct int4 *, int, int *, int *))reduce_istatus_1), _Z16reduce_istatus_1PiP4int4S1_iS_S_, (-1)); __cudaRegisterEntry(__T27, ((void ( *)(double *__restrict__, const double, const double, double *__restrict__, int *__restrict__, const int, const int, const int, const int, const int, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, double *__restrict__, const double, const double, const double, const double, const double, const double, const double, const double, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const double *__restrict__, const int))Rosenbrock), _Z10RosenbrockPdddS_PiiiiiiS_S_S_S_S_S_S_S_S_S_ddddddddPKdS2_S2_S2_S2_S2_S2_S2_i, (-1)); __cudaRegisterVariable(__T27, __shadow_var(ros,::ros), 0, 2400UL, 1, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
