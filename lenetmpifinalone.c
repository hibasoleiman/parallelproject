#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <mpi.h>

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)


#define CONVOLUTE_VALID(input, output, weight) {                                 \
    int rank, size;                                                             \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                        \
    MPI_Comm_size(MPI_COMM_WORLD, &size);                                        \
                                                                                \
    int rows_per_process = GETLENGTH(output) / size;                            \
    int start_row = rank * rows_per_process;                                    \
    int end_row = (rank + 1) * rows_per_process;                                \
                                                                                \
    /* Last process takes remaining rows */                                     \
    if (rank == size - 1) {                                                      \
        end_row = GETLENGTH(output);                                             \
    }                                                                           \
                                                                                \
    /* Perform convolution for the assigned rows */                              \
    for (int o0 = start_row; o0 < end_row; o0++) {                              \
        for (int o1 = 0; o1 < GETLENGTH(*(output)); o1++) {                      \
            for (int w0 = 0; w0 < GETLENGTH(weight); w0++) {                     \
                for (int w1 = 0; w1 < GETLENGTH(*(weight)); w1++) {             \
                    (output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1]; \
                }                                                               \
            }                                                                   \
        }                                                                       \
    }                                                                           \
                                                                                \
    /* Gather the results from all processes to rank 0 */                        \
    MPI_Gather((output) + start_row, rows_per_process * GETLENGTH(*(output)),    \
               MPI_DOUBLE, (output), rows_per_process * GETLENGTH(*(output)),    \
               MPI_DOUBLE, 0, MPI_COMM_WORLD);                                  \
}


#define CONVOLUTE_FULL(input, output, weight) \
{ \
    int rank, size; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    MPI_Comm_size(MPI_COMM_WORLD, &size); \
    \
    /* Divide the input into blocks of rows per process */ \
    int local_rows = GETLENGTH(input) / size; \
    int start_row = rank * local_rows; \
    int end_row = (rank == size - 1) ? GETLENGTH(input) : start_row + local_rows; \
    \
    /* Perform the convolution operation on the local segment */ \
    for (int i0 = start_row; i0 < end_row; i0++) { \
        for (int i1 = 0; i1 < GETLENGTH(*(input)); i1++) { \
            for (int w0 = 0; w0 < GETLENGTH(weight); w0++) { \
                for (int w1 = 0; w1 < GETLENGTH(*(weight)); w1++) { \
                    (output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1]; \
                } \
            } \
        } \
    } \
    \
    /* Gather the results back to the root process */ \
    MPI_Gather(output + start_row, local_rows * GETLENGTH(*(output)), MPI_DOUBLE, \
               output, local_rows * GETLENGTH(*(output)), MPI_DOUBLE, \
               0, MPI_COMM_WORLD); \
}


#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
}


#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH((input)) / GETLENGTH((output));								\
	const int len1 = GETLENGTH((input)) / GETLENGTH((output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH((output)))															\
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
	const int len0 = GETLENGTH((inerror)) / GETLENGTH((outerror));							\
	const int len1 = GETLENGTH((inerror)) / GETLENGTH((outerror));							\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH((outerror)))														\
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

#define DOT_PRODUCT_FORWARD_MPI(input, output, weight, bias, action) \
{ \
    /* Get rank and size within the function */ \
    int rank, size; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    MPI_Comm_size(MPI_COMM_WORLD, &size); \
    \
    int total_output_size = GETLENGTH(output); \
    int local_output_size = total_output_size / size; \
    int start_idx = rank * local_output_size; \
    int end_idx = (rank == size - 1) ? total_output_size : start_idx + local_output_size; \
    \
    /* Local pointer for output to reduce redundant memory accesses */ \
    double *local_output = (double *)output + start_idx; \
    \
    /* Perform the dot product computation for each output element */ \
    for (int y = start_idx; y < end_idx; ++y) { \
        local_output[y - start_idx] = 0.0; /* Initialize output element to 0 */ \
        for (int x = 0; x < GETLENGTH(weight); ++x) { \
            local_output[y - start_idx] += ((double *)input)[x] * weight[x][y]; \
        } \
        local_output[y - start_idx] = action(local_output[y - start_idx] + bias[y]); \
    } \
    \
    /* Gather the computed output values from all processes to the root (rank 0) */ \
    MPI_Gather(local_output, local_output_size, MPI_DOUBLE, \
               output, local_output_size, MPI_DOUBLE, \
               0, MPI_COMM_WORLD); \
}



#define DOT_PRODUCT_BACKWARD_MPI(input, inerror, outerror, weight, wd, bd, actiongrad)	\
{																					\
    /* Declare rank and size inside the function */									\
    int rank, size;																\
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);											\
    MPI_Comm_size(MPI_COMM_WORLD, &size);											\
																					\
    /* Get dimensions of weight and error arrays */								\
    int weight_rows = GETLENGTH(weight);											\
    int weight_cols = GETLENGTH(*weight);											\
    int error_size = GETLENGTH(outerror);											\
    int input_size = GETCOUNT(inerror);											\
																					\
    /* Divide workload for each rank (splitting rows of weight matrix) */			\
    int local_rows = weight_rows / size;											\
    int start_row = rank * local_rows;												\
    int end_row = (rank == size - 1) ? weight_rows : start_row + local_rows;		\
																					\
    /* Step 1: Parallel computation of inerror */									\
    for (int x = start_row; x < end_row; ++x) {									\
        for (int y = 0; y < weight_cols; ++y) {									\
            ((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];		\
        }																			\
    }																				\
																					\
    /* Apply the gradient of the activation function for each input element */		\
    for (int i = rank; i < input_size; i += size) {								\
        ((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
    }																				\
																					\
    /* Step 2: Parallel computation of bias gradients (bd) */						\
    double *local_bd = (double *)calloc(error_size, sizeof(double));				\
    for (int j = rank; j < error_size; j += size) {								\
        local_bd[j] += ((double *)outerror)[j];									\
    }																				\
																					\
    /* Reduce bias gradients to the root (rank 0) */								\
    MPI_Allreduce(MPI_IN_PLACE, local_bd, error_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); \
    if (rank == 0) {																\
        for (int j = 0; j < error_size; ++j) {										\
            bd[j] += local_bd[j];													\
        }																			\
    }																				\
    free(local_bd);																\
																					\
    /* Step 3: Parallel computation of weight gradients (wd) */						\
    double *local_wd = (double *)calloc(weight_rows * weight_cols, sizeof(double));\
    for (int x = start_row; x < end_row; ++x) {									\
        for (int y = 0; y < weight_cols; ++y) {									\
            local_wd[x * weight_cols + y] += ((double *)input)[x] * ((double *)outerror)[y]; \
        }																			\
    }																				\
																					\
    /* Reduce weight gradients across all processes */								\
    MPI_Allreduce(MPI_IN_PLACE, local_wd, weight_rows * weight_cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); \
    if (rank == 0) {																\
        for (int x = 0; x < weight_rows; ++x) {									\
            for (int y = 0; y < weight_cols; ++y) {								\
                wd[x][y] += local_wd[x * weight_cols + y];						\
            }																		\
        }																			\
    }																				\
    free(local_wd);																\
}


double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}
double locallayer_times[6] = {0};  // Array to store cumulative times for layers 1 to 6

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
        clock_t start, end;

       // Layer 1 (Convolution + Activation)
       start = clock();
	CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
	end = clock();
    locallayer_times[0] += ((double)(end - start)) / CLOCKS_PER_SEC;
    
         // Layer 2 (Subsampling)
    start = clock();
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	  end = clock();
    locallayer_times[1] += ((double)(end - start)) / CLOCKS_PER_SEC;

         // Layer 3 (Convolution + Activation)
    start = clock();
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	end = clock();
    locallayer_times[2] += ((double)(end - start)) / CLOCKS_PER_SEC;

        // Layer 4 (Subsampling)
    start = clock();
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	end = clock();
    locallayer_times[3] += ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Layer 5 (Convolution + Activation)
    start = clock();
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	end = clock();
    locallayer_times[4] += ((double)(end - start)) / CLOCKS_PER_SEC;
    
     // Layer 6 (Dot Product + Activation)
    start = clock();
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
	end = clock();
    locallayer_times[5] += ((double)(end - start)) / CLOCKS_PER_SEC;
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{

clock_t start, end;

    // Layer 6 Backward (Dot Product)
    start = clock();
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	end = clock();
    locallayer_times[5] += ((double)(end - start)) / CLOCKS_PER_SEC;
    
      // Layer 5 Backward (Convolution)
    start = clock();
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	end = clock();
    locallayer_times[4] += ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Layer 4 Backward (Subsampling)
    start = clock();
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	 end = clock();
    locallayer_times[3] += ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Layer 3 Backward (Convolution)
    start = clock();
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	end = clock();
    locallayer_times[2] += ((double)(end - start)) / CLOCKS_PER_SEC;
    
     // Layer 2 Backward (Subsampling)
    start = clock();
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	end = clock();
    locallayer_times[1] += ((double)(end - start)) / CLOCKS_PER_SEC;
    
     // Layer 1 Backward (Convolution)
    start = clock();
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
	 end = clock();
    locallayer_times[0] += ((double)(end - start)) / CLOCKS_PER_SEC;
}

static inline void load_input(Feature *features, image input)
{
	double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(input) / sizeof(*input))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(input) / sizeof(*input))
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Split work among processes (batchSize divided by number of processes)
    int start = (batchSize / size) * rank;
    int end = (batchSize / size) * (rank + 1);
    if (rank == size - 1) {
        end = batchSize;  // Ensure the last process takes the remainder
    }

    for (int i = start; i < end; ++i)
    {
        Feature features = { 0 };
        Feature errors = { 0 };
        LeNet5 deltas = { 0 };

        // Load input and perform forward/backward passes
        load_input(&features, inputs[i]);
        forward(lenet, &features, relu);
        load_target(&features, &errors, labels[i]);
        backward(lenet, &deltas, &errors, &features, relugrad);

        // Accumulate the deltas (gradients)
        for (int j = 0; j < GETCOUNT(LeNet5); ++j)
            buffer[j] += ((double *)&deltas)[j];
    }

    // Gather results from all processes (sum of gradients)
    double global_buffer[GETCOUNT(LeNet5)] = { 0 };
    MPI_Reduce(buffer, global_buffer, GETCOUNT(LeNet5), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Only the root process (rank 0) updates the weights
    if (rank == 0) {
        double k = ALPHA / batchSize;
        for (int i = 0; i < GETCOUNT(LeNet5); ++i)
            ((double *)lenet)[i] += k * global_buffer[i];
    }
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
void print_cumulative_times()
{
    printf("\nCumulative mpiparallel Times for Each Layer:\n");
    for (int i = 0; i < 6; i++)
    {
        printf("Layer %d: %f seconds\n", i + 1, locallayer_times[i]);
    }
}
