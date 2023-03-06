/*
 * ppp_cuda.cuh
 *
 *      Author: David Lawrie
 *      for cuda error checking functions used by both go_fish and by sfs
 */

#ifndef MYCUDAHELPER_CUH_
#define MYCUDAHELPER_CUH_

#include <cuda_runtime.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;
#include <memory>
#include "../_internal/ppp_types.hpp"

/* ----- cuda vector math functions ----- */

using uint = unsigned int;
using ulong = unsigned long;
//IMPORTANT DO NOT PUT INTO NAMESPACE OR ADL WILL NOT BE ABLE TO FIND CUDA's ::makefloat4 WITHOUT THE :: AS :: WITH A BLANK IN FRONT DENOTES A NO-NAMESPACE FUNCTION
//IF IN NAMESPACE ADL WILL SEE THE BELOW make_float4 IN NAMESPACE AND ASSUME THAT IS THE ONLY ONE AVAILABLE (SAME FOR make_uint4)
__device__ __forceinline__ float4 make_float4(float val){ return make_float4(val,val,val,val); }

__device__ __forceinline__ float4 operator*(float lhs, uint4 rhs){ return make_float4(lhs*rhs.x,lhs*rhs.y,lhs*rhs.z,lhs*rhs.w); }
__device__ __forceinline__ float4 operator*(float lhs, float4 rhs){ return make_float4(lhs*rhs.x,lhs*rhs.y,lhs*rhs.z,lhs*rhs.w); }
__device__ __forceinline__ float4 operator-(float lhs, float4 rhs){ return make_float4((lhs-rhs.x),(lhs-rhs.y),(lhs-rhs.z),(lhs-rhs.w)); }
__device__ __forceinline__ float4 operator*(float4 lhs, float4 rhs){ return make_float4(lhs.x*rhs.x,lhs.y*rhs.y,lhs.z*rhs.z,lhs.w*rhs.w); }
__device__ __forceinline__ float4 operator/(float4 lhs, float4 rhs){ return make_float4(lhs.x/rhs.x,lhs.y/rhs.y,lhs.z/rhs.z,lhs.w/rhs.w); }
__device__ __forceinline__ float4 operator+(float4 lhs, float4 rhs){ return make_float4(lhs.x+rhs.x,lhs.y+rhs.y,lhs.z+rhs.z,lhs.w+rhs.w); }
__device__ __forceinline__ float4 operator+(float4 lhs, float rhs){ return make_float4(lhs.x+rhs,lhs.y+rhs,lhs.z+rhs,lhs.w+rhs); }
__device__ __forceinline__ float4 operator/(float4 lhs, float rhs){ return make_float4(lhs.x/rhs,lhs.y/rhs,lhs.z/rhs,lhs.w/rhs); }
__device__ __forceinline__ void operator+=(float4 &lhs, float4 rhs){ lhs.x+=rhs.x;lhs.y+=rhs.y;lhs.z+=rhs.z;lhs.w+=rhs.w; }

__device__ __forceinline__ uint4 make_uint4(uint val){ return make_uint4(val,val,val,val); }

/* ----- end cuda vector math functions ----- */

/* ----- cuda error checking ----- */
#ifndef __CUDA_DEBUG_SYNC__
constexpr bool cuda_debug_sync = false;
#else
constexpr bool cuda_debug_sync = true;
#endif

#define cudaCheckErrors(expr1,expr2,expr3) { cudaError_t e = expr1; int g = expr2; int p = expr3; if (e != cudaSuccess) { fprintf(stderr,"error %d %s\tfile %s\tline %d\tgeneration %d\t population %d\n", e, cudaGetErrorString(e),__FILE__,__LINE__, g,p); exit(1); } }
#define cudaCheckErrorsAsync(expr1,expr2,expr3) { cudaCheckErrors(expr1,expr2,expr3); if(cuda_debug_sync){ cudaCheckErrors(cudaDeviceSynchronize(),expr2,expr3); } }
/* ----- end cuda error checking ----- */

namespace ppp{

/*rounds float to nearest unsigned int*/
__host__ __device__ __forceinline__ uint fround_u(float num){ return (num+0.5f); }

/*round_clamping_float :=> max must be <= MAX_UINT-1*/
__host__ __device__ __forceinline__ uint rfclamp_u(float j, float max){ return fround_u(fmaxf(0.f,fminf(j,max))); }

/*clamps unsigned int to rounded max float value*/
__host__ __device__ __forceinline__ uint clamp_u(uint j, float max){ return min(j,fround_u(max)); }

/* ----- cuda device setting ----- */
__forceinline__ cudaDeviceProp set_cuda_device(int & cuda_device){
	int cudaDeviceCount;
	cudaCheckErrorsAsync(cudaGetDeviceCount(&cudaDeviceCount),-1,-1);
	if(cuda_device >= 0 && cuda_device < cudaDeviceCount){ cudaCheckErrors(cudaSetDevice(cuda_device),-1,-1); } //unless user specifies, driver auto-magically selects free GPU to run on
	int myDevice;
	cudaCheckErrorsAsync(cudaGetDevice(&myDevice),-1,-1);
	cudaDeviceProp devProp;
	cudaCheckErrors(cudaGetDeviceProperties(&devProp, myDevice),-1,-1);
	cuda_device = myDevice;
	return devProp;
}
/* ----- end cuda device setting ----- */

/* ----- unique_device_ptr wrapper ----- */
template<typename T>
struct delete_device_pointer{
	inline void operator()(T * ptr) noexcept{ if(ptr != nullptr){ cudaFree(ptr); ptr = nullptr; } }
};

template<typename T>
inline T* safe_cuda_malloc(ulong num_items, int generation, int population){
	if(num_items == 0){ return nullptr; }
	T* ptr;
	cudaError_t error = cudaMalloc((void**) &ptr, num_items*sizeof(T));
	if(error != cudaSuccess) {
		delete_device_pointer<T> deleter;
		deleter(ptr); //just in case the error returned is from a previous async command and cudaMalloc successfully allocated memory
		cudaCheckErrors(error,generation,population);
	}
	return ptr;
}

template<>
inline void* safe_cuda_malloc(ulong num_items, int generation, int population){
	if(num_items == 0){ return nullptr; }
	void* ptr;
	cudaError_t error = cudaMalloc((void**) &ptr, num_items);
	if(error != cudaSuccess) {
		delete_device_pointer<void> deleter;
		deleter(ptr); //just in case the error returned is from a previous async command and cudaMalloc successfully allocated memory
		cudaCheckErrors(error,generation,population);
	}
	return ptr;
}

template<typename T>
using unique_device_ptr = std::unique_ptr<T, delete_device_pointer<T>>;

template<typename T>
inline unique_device_ptr<T> make_unique_device_ptr(ulong num_items, int generation, int population){ return unique_device_ptr<T>(safe_cuda_malloc<T>(num_items, generation, population)); }

template<typename T>
inline void reset_device_ptr(unique_device_ptr<T> & ptr, ulong num_items, int generation, int population){ ptr.reset(safe_cuda_malloc<T>(num_items, generation, population)); }
/* ----- end unique_device_ptr wrapper ----- */

/* ----- registered host memory wrapper ----- */
template<typename T>
inline T* safe_cuda_host_register(T* ptr, ulong num_items, int generation, int population){
	cudaError_t error = cudaHostRegister(ptr, num_items*sizeof(T),cudaHostRegisterPortable);
	if(error != cudaSuccess) {
		cudaHostUnregister(ptr); //just in case the error returned is from a previous async command and cudaHostRegister successfully pinned memory, if it didn't this might also return an error, but I don't check it or throw that error, just the original
		cudaCheckErrors(error,generation,population);
	}
	return ptr;
}

template<typename T>
struct unregister_host_pointer{
	inline void operator()(T * ptr) noexcept{ if(ptr != nullptr){ cudaHostUnregister(ptr); } } //does *NOT* free or delete pointer
};

template<typename T>
using registered_host_ptr = std::unique_ptr<T, unregister_host_pointer<T>>;

//pins host memory so all cuda contexts see the memory as pinned
template<typename T>
inline registered_host_ptr<T> register_host_ptr(T* ptr, ulong num_items, int generation, int population){ return registered_host_ptr<T>(safe_cuda_host_register(ptr, num_items, generation, population)); }
/* ----- end registered host memory wrapper ----- */

/* ----- raii obj array ----- */
template<typename T, typename Obj_Creator, typename Obj_Destroyer>
class raii_cuobj_array{
	ppp::unique_host_span<T> obj_array;
	Obj_Creator my_creator;
	Obj_Destroyer my_destroyer;

public:
	inline raii_cuobj_array(): my_creator{}, my_destroyer{} { }

	inline raii_cuobj_array(std::ptrdiff_t num_items): obj_array(num_items), my_creator{}, my_destroyer{} {
		for(int i = 0; i < num_items; i++){
			cudaError_t error =  my_creator(&obj_array[i]);
			if(error != cudaSuccess) {
				for(int j = 0; j <= i; j++){ my_destroyer(obj_array[j]); } //makes sure to clean up all obj arrays if cudaCheckErrors throws
				cudaCheckErrors(error,-1,i);
			}
		}
	}

	inline raii_cuobj_array(raii_cuobj_array &) = delete; //not copyable
	inline raii_cuobj_array(raii_cuobj_array &&) = delete; //not moveable
	inline raii_cuobj_array& operator=(const raii_cuobj_array &) = delete; //not copyable
	inline raii_cuobj_array& operator=(raii_cuobj_array &&) = delete; //not moveable

	inline T& operator[](std::ptrdiff_t index){ return obj_array[index]; } //throw error if i > my_num_items
	inline const T& operator[](std::ptrdiff_t index) const{ return obj_array[index]; } //throw error if i > my_num_items

	inline ~raii_cuobj_array() noexcept { for(int i = 0; i < obj_array.size(); i++){ my_destroyer(obj_array[i]); } }
};

struct stream_creator{ inline cudaError_t operator()(cudaStream_t* stream){ return cudaStreamCreateWithFlags(stream,cudaStreamNonBlocking); } };

struct stream_destroyer{ inline void operator()(cudaStream_t& stream){ cudaStreamDestroy(stream); } };

struct event_creator{ inline cudaError_t operator()(cudaEvent_t* event){ return cudaEventCreateWithFlags(event,cudaEventDisableTiming); } };

struct event_destroyer{ inline void operator()(cudaEvent_t& event){ cudaEventDestroy(event); } };

using raii_stream_array = raii_cuobj_array<cudaStream_t,stream_creator,stream_destroyer>;

using raii_event_array = raii_cuobj_array<cudaEvent_t,event_creator,event_destroyer>;
/* ----- end raii obj array ----- */

} /* ----- end namespace PPP ----- */

#endif /* MYCUDAHELPER_CUH_ */
