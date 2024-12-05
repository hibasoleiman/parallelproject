#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>


#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

__global__ void CONVOLUTE_VALID(double *input, double *mask, double *output, int height, int width, int channels)											\
{																						\
	extern __shared__ double N_ds[];

     int tile_height_o = blockDim.y - LENGTH_KERNEL +1;
    int tile_width_o = blockDim.x - LENGTH_KERNEL +1;
    int BLOCK_WIDTH = tile_width_o + (LENGTH_KERNEL - 1);

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row_o = blockIdx.y * tile_height_o + ty;
    int col_o = blockIdx.x * tile_width_o + tx;

    int row_i = row_o ;
    int col_i = col_o ;

    for (int c=0; c< channels; ++c){

        if(row_i < height && col_i < width)
            N_ds[ty * blockDim.x + tx] = input[(row_i * width + col_i) * channels + c];
        else
            N_ds[ty * blockDim.x + tx] = 0; //out of bounds

        __syncthreads();

        float Pvalue = 0;
        if(ty < tile_height_o && tx < tile_width_o && row_o + LENGTH_KERNEL<= height && col_o + LENGTH_KERNEL<= width) {

            for(int k = 0; k < LENGTH_KERNEL; ++k)
                for(int l = 0; l < LENGTH_KERNEL; ++l)
                    Pvalue += N_ds[(ty + k) * blockDim.x + tx + l] * mask[k * LENGTH_KERNEL + l];  
        
            output[(row_o * width + col_o) * channels + c] = Pvalue;
        }

        __syncthreads();
    }
}

    


void ConvoluteValidHost(double *h_input,double *h_mask,double *h_output,int height, int width, int channels              
) {
    
    double *d_input;
    double *d_mask;
    double *d_output;

  
    size_t input_size = channels * height * width * sizeof(double);
    size_t mask_size = LENGTH_KERNEL * LENGTH_KERNEL * sizeof(double);
    size_t output_size = channels * height * width * sizeof(double);

    
    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_mask, mask_size);
    cudaMalloc((void**)&d_output, output_size);

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, mask_size, cudaMemcpyHostToDevice);

    int tile_height_o = blockDim.y - LENGTH_KERNEL +1;
    int tile_width_o = blockDim.x - LENGTH_KERNEL +1;
    int BLOCK_WIDTH = tile_width_o + (LENGTH_KERNEL - 1);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
    
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    
    size_t shared_mem_size = blockDim.x * blockDim.y * sizeof(float);

  
    CONVOLUTE_VALID<<<gridDim, blockDim, shared_mem_size>>>(
        d_input, d_mask,d_output, height, width, channels);

    
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

   
    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);
}


__global__ void convolution_Full(double *input,double *mask, double *output,int height, int width, int channels) {
    
    extern __shared__ double N_ds[];

    int tile_height_o = blockDim.y - LENGTH_KERNEL +1;
    int tile_width_o = blockDim.x - LENGTH_KERNEL +1;


    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int row_o = blockIdx.y * tile_height_o + ty;
    int col_o = blockIdx.x * tile_width_o + tx;

    int row_i = row_o - (LENGTH_KERNEL / 2);
    int col_i = col_o - (LENGTH_KERNEL / 2);

    for(int c = 0; c < channels; ++c){
        
        if(row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
            N_ds[ty * blockDim.x + tx] = input[(row_i * width + col_i) * channels + c];
        else
            N_ds[ty * blockDim.x + tx] = 0;

        __syncthreads();

        float Pvalue = 0;
        if(ty < tile_height_o && tx < tile_width_o && row_o < height && col_o < width) {

            for(int k = 0; k < LENGTH_KERNEL; ++k)
                for(int l = 0; l <LENGTH_KERNEL; ++l)
                    Pvalue += N_ds[(ty + k) * blockDim.x + tx + l] * mask[k *LENGTH_KERNEL + l];  

            if(row_o<height && col_o<width){
                output[(row_o * width + col_o) * channels + c] = Pvalue;
            }
        }

        __syncthreads();
    }
}

void host_convolution_full(double *h_input,double *h_mask,double *h_output, int height, int width, int channels) {
    
    size_t input_size = height * width * channels * sizeof(double);
    size_t mask_size = LENGTH_KERNEL * LENGTH_KERNEL * sizeof(double);
    size_t output_size = height * width * channels * sizeof(double);


    double *d_input, *d_mask, *d_output;

    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_mask, mask_size);
    cudaMalloc((void**)&d_output, output_size);

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, mask_size, cudaMemcpyHostToDevice);
    
    int tile_height_o = blockDim.y - LENGTH_KERNEL +1;
    int tile_width_o= blockDim.x - LENGTH_KERNEL + 1;
    int BLOCK_WIDTH = tile_width_o + (LENGTH_KERNEL - 1);

    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH); 
    dim3 gridDim((width + tile_width_o - 1) / tile_width_o,
                 (height + tile_height_o - 1) / tile_height_o);

    // Calculate shared memory size
    size_t shared_mem_size = blockDim.x * blockDim.y * sizeof(float);

    // Launch kernel
    convolution_Full<<<gridDim, blockDim, shared_mem_size>>>(
        d_input, d_mask, d_output, height, width, channels);

    // Synchronize and check for errors
    cudaDeviceSynchronize();

    // Copy output back to host
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);
}


#define CONVOLUTION_FORWARD(input,output,weight,bias,action, height, width, channels)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			ConvoluteValidHost(input[x], weight[x][y], output[y],height,width, channels );					\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			host_convolution_full(outerror[y],  weight[x][y], inerror[x], height, width, channels);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTEVALIDHost(input[x], wd[x][y], outerror[y], height,width, channels );						\
}


#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];							\
	}																							\
}

__global__ void  DOT_PRODUCT_FORWARD(double *input,double *output, double *weight,double *bias,int input_cols, int input_rows, double(*action)(double))				\
{					
  int Tile_Width = blockDim.x;
   extern __shared__ double Sinput[];													\
   extern __shared__ double Sweight[];	
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int tx= threadIdx.x;												\
    int ty= threadIdx.y;
    
    int row = by*blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    double outputval = 0;
    												\
	for (int p = 0; p < (input_cols + Tile_Width -1) / Tile_Width; ++p){
        Sinput[ty * Tile_Width + tx] = row < input_rows && p * Tile_Width + tx < input_cols ?
                                        input[row * input_cols + p * Tile_Width + tx] : 0;
                                        
        Sweight[ty * Tile_Width + tx] = p * Tile_Width + ty < LENGTH_KERNEL && col < LENGTH_KERNEL ?
                                        weight[(p * Tile_Width + ty) * LENGTH_KERNEL + col] : 0;
        __syncthreads();

        for(int k=0;k<Tile_Width;++k){
            outputval += Sinput[ty * Tile_Width +k ] * Sweight[k*Tile_Width + tx];

        }    

        __syncthreads();                           
    }								\
	if(row<input_rows && col< weight_cols){
        outputval += bias[col];
        outputval = action(outputval);
        output[row*weight_cols + col] = outputval;
    }	
    		\
}

void dotProductForwardHost(double* h_input, double* h_output, double* h_weight, double* h_bias,int width, int height, double (*action)(double)) 
{
     int Tile_Width = blockDim.x;
    size_t input_size = width * height * sizeof(double);
    size_t weight_size = LENGTH_KERNEL *LENGTH_KERNEL * sizeof(double);
    size_t output_size = width * LENGTH_KERNEL * sizeof(double);
    size_t bias_size = LENGTH_KERNEL * sizeof(double);

    
    double *d_input, *d_weight, *d_bias, *d_output;

    
    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_weight, weight_size);
    cudaMalloc((void**)&d_bias, bias_size);
    cudaMalloc((void**)&d_output, output_size);

   
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bias_size, cudaMemcpyHostToDevice);

  
    dim3 blockDim(Tile_Width, Tile_Width, 1);
    dim3 gridDim((height + Tile_Width - 1) / Tile_Width,
                 (width + Tile_Width - 1) / Tile_Width, 1);

    // Shared memory size
    size_t shared_mem_size = 2 * Tile_Width * Tile_Width * sizeof(double);

    // Launch kernel
    DOT_PRODUCT_FORWARD<<<gridDim, blockDim, shared_mem_size>>>(
        d_input, d_output, d_weight, d_bias, action);

    // Synchronize and check for errors
    cudaDeviceSynchronize();
 

    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}


__global__ void  DOT_PRODUCT_BACKWARD(double *input,double *inerror,double *outerror,double *weight,double *wd,double *bd, int input_Len, int outputLen, double (*actiongrad)(double))	\
{     
    
     int Tile_Width = blockDim.x;
    extern __shared__ double Sweight[];
     extern __shared__ double Souterror[];
    extern  __shared__ double Sinput[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int tx= threadIdx.x;												\
    int ty= threadIdx.y;
    
    int row = by*blockDim.y + ty;
    int col = bx * blockDim.x + tx;	
    double inerrorval = 0;

    for(int p=0;p < (input_cols + Tile_Width-1)/Tile_Width; ++p){
        Sinput[ty * Tile_Width + tx] = row < input_Len && p * Tile_Width + tx < input_Len ?
                                        input[row * input_Len + p * Tile_Width + tx] : 0;
                                        
        Sweight[ty * Tile_Width + tx] = p * Tile_Width + ty < weight_rows && col < weight_cols ?
                                        weight[(p * Tile_Width + ty) * weight_cols + col] : 0;
        
        Souterror[ty * Tile_Width + tx] =row < output_rows && p * Tile_Width + tx < output_cols ?
                                        outerror[row * output_cols + p * Tile_Width + tx] : 0;
        __syncthreads();
        
        for(int k=0;k<Tile_Width;++k){
            inerrorval += Souterror[ty * Tile_Width +k ] * Sweight[k*Tile_Width + tx];

        }
        __syncthreads();
        if(Row<inputLen){
            inerror[Row] = inerror_val*actiongrad(input[Row]);

        }

        if(Col<outputLen){
            atomicAdd(&bd[Col], outerror[Col]);
        }

        if (Row < inputLen && Col < outputLen) {
        wd[Row * outputLen + Col] += input[Row] * outerror[Col];
    }
        

    }												\
	
}
void dotProductBackwardHost(
    double* h_input, double* h_inerror, double* h_outerror, double* h_weight,
    double* h_wd, double* h_bd,
    int input_rows, int input_cols, int output_rows, int output_cols, double (*actiongrad)(double)) 
{
    
    size_t input_size = input_rows * input_cols * sizeof(double);
    size_t outerror_size = output_rows * output_cols * sizeof(double);
    size_t weight_size = input_cols * output_cols * sizeof(double);
    size_t inerror_size = input_rows * input_cols * sizeof(double);
    size_t wd_size = input_cols * output_cols * sizeof(double);
    size_t bd_size = output_cols * sizeof(double);

    

    // Device pointers
    double *d_input, *d_inerror, *d_outerror, *d_weight, *d_wd, *d_bd;

    // Allocate device memory
    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_inerror, inerror_size);
    cudaMalloc((void**)&d_outerror, outerror_size);
    cudaMalloc((void**)&d_weight, weight_size);
    cudaMalloc((void**)&d_wd, wd_size);
    cudaMalloc((void**)&d_bd, bd_size);

    // Initialize wd and bd to zero on the device
    cudaMemset(d_wd, 0, wd_size);
    cudaMemset(d_bd, 0, bd_size);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_outerror, h_outerror, outerror_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(Tile_Width, Tile_Width, 1);
    dim3 gridDim((input_cols + Tile_Width - 1) / Tile_Width,
                 (input_rows + Tile_Width - 1) / Tile_Width, 1);

    // Shared memory size
    size_t shared_mem_size = (3 * Tile_Width * Tile_Width) * sizeof(double);

    // Launch kernel
    DOT_PRODUCT_BACKWARD<<<gridDim, blockDim, shared_mem_size>>>(
        d_input, d_inerror, d_outerror, d_weight, d_wd, d_bd, input_rows, output_cols, actiongrad);

    // Synchronize and check for errors
    cudaDeviceSynchronize();
  

    // Copy results back to the host
    cudaMemcpy(h_inerror, d_inerror, inerror_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_wd, d_wd, wd_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bd, d_bd, bd_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_inerror);
    cudaFree(d_outerror);
    cudaFree(d_weight);
    cudaFree(d_wd);
    cudaFree(d_bd);
}


double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
	CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action, LENGTH_FEATURE0, LENGTH_FEATURE1, INPUT);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action,LENGTH_FEATURE2, LENGTH_FEATURE3, LAYER1);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action, LENGTH_FEATURE4, LENGTH_FEATURE5, INPUT);
	dotProductForwardHost(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, LENGTH_FEATURE5*LENGTH_FEATURES5*LAYER5,OUTPUT,actiongrad);
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}

static inline void load_input(Feature *features, image input)
{
	double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}
}

static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
	double inner = 0;
	for (int i = 0; i < count; ++i)
	{
		double res = 0;
		for (int j = 0; j < count; ++j)
		{
			res += exp(input[j] - input[i]);
		}
		loss[i] = 1. / res;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;
	}
}

static void load_target(Feature *features, Feature *errors, int label)
{
	double *output = (double *)features->output;
	double *error = (double *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
{
	double *output = (double *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	double maxvalue = *output;
	for (uint8 i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
		}
	}
	return result;
}

static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}


void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize)
{
	double buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;

	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };
		load_input(&features, inputs[i]);
		forward(lenet, &features, relu);
		load_target(&features, &errors, labels[i]);
		backward(lenet, &deltas, &errors, &features, relugrad);
		
		{
			FOREACH(j, GETCOUNT(LeNet5))
				buffer[j] += ((double *)&deltas)[j];
		}
	}
	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += k * buffer[i];
}

void Train(LeNet5 *lenet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	load_target(&features, &errors, label);
	backward(lenet, &deltas, &errors, &features, relugrad);
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
}

uint8 Predict(LeNet5 *lenet, image input,uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	return get_result(&features, count);
}

void Initial(LeNet5 *lenet)
{
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = f64rand());
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
	for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
} 