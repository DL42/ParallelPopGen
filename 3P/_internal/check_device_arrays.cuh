__global__ static void print_Device_array_uint(uint * array, int num){

	for(int i = 0; i < num; i++){
		//if(i%1000 == 0){ printf("\n"); }
		printf("%d: %d\t",i,array[i]);
	}
}

__global__ static void sum_Device_array_bit(uint * array, int num){
//	int sum = 0;
	for(int i = 0; i < num; i++){
		//if(i%1000 == 0){ printf("\n"); }
		uint n = array[i];
		while (n) {
		    if (n & 1)
		    	sum+=1;
		    n >>= 1;
		}
		printf("%d\t",__popc(array[i]));
	}
}

__global__ static void sum_Device_array_uint(uint * array, int num){
	int j = 0;
	for(int i = 0; i < num; i++){
		j += array[i];
	}
	printf("%d",j);
}

__global__ static void sum_Device_array_float(float * array, int start, int end){
	double j = 0;
	for(int i = start; i < end; i++){
		j += array[i];
	}
	printf("%lf\n",j);
}

__global__ static void compareDevicearray(int * array1, int * array2, int array_length){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < array_length; id+= blockDim.x*gridDim.x){
		if(array1[id] != array2[id]){ printf("%d,%d,%d\t",id,array1[id],array2[id]); }
	}
}

__global__ static void copyDevicearray(int * array1, int * array2, int array_length){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < array_length; id+= blockDim.x*gridDim.x){ array1[id] = array2[id]; }
}

__global__ static void compareDevicearray(float * array1, float * array2, int array_length){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < array_length; id+= blockDim.x*gridDim.x){
		if(array1[id] != array2[id]){ printf("%d,%f,%f\t",id,array1[id],array2[id]); return; }
	}
}

__global__ static void copyDevicearray(float * array1, float * array2, int array_length){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < array_length; id+= blockDim.x*gridDim.x){ array1[id] = array2[id]; }
}

__global__ static void print_Device_array_float(float * array, int num){
	printf("%5.10e\n",array[num]);
}
