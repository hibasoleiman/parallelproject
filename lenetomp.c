#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#define FILE_TRAIN_IMAGE		"train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000


int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data)*count, 1, fp_image);
	fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
	for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
	{
		TrainBatch(lenet, train_data + i, train_label + i, batch_size);
		if (i * 100 / total_size > percent)
			printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
	}
}

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label,int total_size)
{
	int right = 0, percent = 0;
	for (int i = 0; i < total_size; ++i)
	{
		uint8 l = test_label[i];
		int p = Predict(lenet, test_data[i], 10);
		right += l == p;
		if (i * 100 / total_size > percent)
			printf("test:%2d%%\n", percent = i * 100 / total_size);
	}
	return right;
}

int save(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if (!fp) return 1;
	fwrite(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}

int load(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}



void foo()
{
	image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
	uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
	if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(train_data);
		free(train_label);
		system("pause");
	}
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(test_data);
		free(test_label);
		system("pause");
	}


	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	if (load(lenet, LENET_FILE))
		Initial(lenet);
	clock_t start = clock();
	int batches[] = { 300 };
	for (int i = 0; i < sizeof(batches) / sizeof(*batches);++i)
		training(lenet, train_data, train_label, batches[i],COUNT_TRAIN);
	int right = testing(lenet, test_data, test_label, COUNT_TEST);
	printf("%d/%d\n", right, COUNT_TEST);
	printf("Time:%u\n", (unsigned)((clock() - start)/ CLOCKS_PER_SEC));
	print_cumulative_times();
	omp_set_num_threads(3);
	//save(lenet, LENET_FILE);
	free(lenet);
	free(train_data);
	free(train_label);
	free(test_data);
	free(test_label);
	system("pause");
}

int main()
{
 
	foo();
	
	return 0;
}
#include "lenet.h"
#include <memory.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

#define GETLENGTH(array) (sizeof(array) / sizeof(*(array)))
#define GETCOUNT(array)  (sizeof(array) / sizeof(double))
#define FOREACH(i, count) for (int i = 0; i < count; ++i)

#define CONVOLUTE_VALID(input, output, weight) \
{ \
    FOREACH(o0, GETLENGTH(output)) \
        FOREACH(o1, GETLENGTH(*(output))) \
            FOREACH(w0, GETLENGTH(weight)) \
                FOREACH(w1, GETLENGTH(*(weight))) \
                    (output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1]; \
}

#define CONVOLUTE_FULL(input, output, weight) \
{ \
    FOREACH(i0, GETLENGTH(input)) \
        FOREACH(i1, GETLENGTH(*(input))) \
            FOREACH(w0, GETLENGTH(weight)) \
                FOREACH(w1, GETLENGTH(*(weight))) \
                    (output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1]; \
}

#define CONVOLUTION_FORWARD(input, output, weight, bias, action) \
{ \
    _Pragma("omp parallel for collapse(2) schedule(dynamic)") \
    for (int x = 0; x < GETLENGTH(weight); ++x) \
        for (int y = 0; y < GETLENGTH(*weight); ++y) \
            CONVOLUTE_VALID(input[x], output[y], weight[x][y]); \
\
    _Pragma("omp parallel for") \
    FOREACH(j, GETLENGTH(output)) \
        FOREACH(i, GETCOUNT(output[j])) \
            ((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]); \
}

#define CONVOLUTION_BACKWARD(input, inerror, outerror, weight, wd, bd, actiongrad) \
{ \
    _Pragma("omp parallel for collapse(2) schedule(dynamic)") \
    for (int x = 0; x < GETLENGTH(weight); ++x) \
        for (int y = 0; y < GETLENGTH(*weight); ++y) \
            CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]); \
    _Pragma("omp parallel for") \
    FOREACH(i, GETCOUNT(inerror)) \
        ((double *)inerror)[i] *= actiongrad(((double *)input)[i]); \
    _Pragma("omp parallel for") \
    FOREACH(j, GETLENGTH(outerror)) \
        FOREACH(i, GETCOUNT(outerror[j])) \
            bd[j] += ((double *)outerror[j])[i]; \
    _Pragma("omp parallel for collapse(2) schedule(dynamic)") \
    for (int x = 0; x < GETLENGTH(weight); ++x) \
        for (int y = 0; y < GETLENGTH(*weight); ++y) \
            CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]); \
}

#define SUBSAMP_MAX_FORWARD(input, output) \
{ \
    const int len0 = GETLENGTH((input)) / GETLENGTH((output)); \
    const int len1 = GETLENGTH((input)) / GETLENGTH((output)); \
    _Pragma("omp parallel for collapse(3) schedule(dynamic)") \
    FOREACH(i, GETLENGTH(output)) \
    FOREACH(o0, GETLENGTH(*(output))) \
    FOREACH(o1, GETLENGTH((output))) \
    { \
        int x0 = 0, x1 = 0, ismax; \
        FOREACH(l0, len0) \
            FOREACH(l1, len1) \
        { \
            ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1]; \
            x0 += ismax * (l0 - x0); \
            x1 += ismax * (l1 - x1); \
        } \
        output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1]; \
    } \
}

#define SUBSAMP_MAX_BACKWARD(input, inerror, outerror) \
{ \
    const int len0 = GETLENGTH((inerror)) / GETLENGTH((outerror)); \
    const int len1 = GETLENGTH((inerror)) / GETLENGTH((outerror)); \
    _Pragma("omp parallel for collapse(3) schedule(dynamic)") \
    FOREACH(i, GETLENGTH(outerror)) \
    FOREACH(o0, GETLENGTH(*(outerror))) \
    FOREACH(o1, GETLENGTH((outerror))) \
    { \
        int x0 = 0, x1 = 0, ismax; \
        FOREACH(l0, len0) \
            FOREACH(l1, len1) \
        { \
            ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1]; \
            x0 += ismax * (l0 - x0); \
            x1 += ismax * (l1 - x1); \
        } \
        inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1]; \
    } \
}

#define DOT_PRODUCT_FORWARD(input, output, weight, bias, action) \
{ \
    _Pragma("omp parallel for collapse(2) schedule(dynamic)") \
    for (int x = 0; x < GETLENGTH(weight); ++x) \
        for (int y = 0; y < GETLENGTH(*weight); ++y) \
            ((double *)output)[y] += ((double *)input)[x] * weight[x][y]; \
    _Pragma("omp parallel for") \
    FOREACH(j, GETLENGTH(bias)) \
        ((double *)output)[j] = action(((double *)output)[j] + bias[j]); \
}

#define DOT_PRODUCT_BACKWARD(input, inerror, outerror, weight, wd, bd, actiongrad) \
{ \
    _Pragma("omp parallel for collapse(2) schedule(dynamic)") \
    for (int x = 0; x < GETLENGTH(weight); ++x) \
        for (int y = 0; y < GETLENGTH(*weight); ++y) \
            ((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y]; \
    _Pragma("omp parallel for") \
    FOREACH(i, GETCOUNT(inerror)) \
        ((double *)inerror)[i] *= actiongrad(((double *)input)[i]); \
    _Pragma("omp parallel for") \
    FOREACH(j, GETLENGTH(outerror)) \
        bd[j] += ((double *)outerror)[j]; \
    _Pragma("omp parallel for collapse(2) schedule(dynamic)") \
    for (int x = 0; x < GETLENGTH(weight); ++x) \
        for (int y = 0; y < GETLENGTH(*weight); ++y) \
            wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y]; \
}


double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}
double times[6] = {0};  

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{

        clock_t start, end;
        
       start = omp_get_wtime();
       CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
       end = omp_get_wtime();
       times[0] += (end - start);

     
      start = omp_get_wtime();
      SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
      end = omp_get_wtime();
      times[1] += (end - start);


 start = omp_get_wtime();
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	 end = omp_get_wtime();
    times[2] += (end - start);
     

   start = omp_get_wtime();
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	 end = omp_get_wtime();
      times[3] += (end - start);
   

    start = omp_get_wtime();
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	 end = omp_get_wtime();
    layer_times[4] += (end - start);
    

    start = omp_get_wtime();
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
	 end = omp_get_wtime();
     times[5] += (end - start);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
 
clock_t start, end;
       
       start = omp_get_wtime();
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	 end = omp_get_wtime();
       times[5] += (end - start);

          start = omp_get_wtime();
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	 end = omp_get_wtime();
       times[4] += (end - start);
    
       
       start = omp_get_wtime();
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	  end = omp_get_wtime();
        times[3] += (end - start);
        
       start = omp_get_wtime();
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	 end = omp_get_wtime();
       times[2] += (end - start);
    
      
    start = omp_get_wtime();
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	 end = omp_get_wtime();
      times[1] += (end - start);
      

      start = omp_get_wtime();
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
	 end = omp_get_wtime();
      times[0] += (end - start);

}
static inline void load_input(Feature *features, image input)
{

        #pragma omp parallel
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
}
static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)

{
 #pragma omp parallel
    {
	double inner = 0;
	 #pragma omp parallel for reduction(-:inner)

	for (int i = 0; i < count; ++i)
	{
		double res = 0;
		#pragma omp parallel for reduction(+:res)

		for (int j = 0; j < count; ++j)
		{
			res += exp(input[j] - input[i]);
		}
		loss[i] = 1. / res;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	#pragma omp parallel for

	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;
	}
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
#pragma omp parallel
    {

	double buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };
		load_input(&features, inputs[i]);
		forward(lenet, &features, relu);
		load_target(&features, &errors, labels[i]);
		backward(lenet, &deltas, &errors, &features, relugrad);
		#pragma omp critical
		{
			FOREACH(j, GETCOUNT(LeNet5))
				buffer[j] += ((double *)&deltas)[j];
		}
	}
	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += k * buffer[i];
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
    printf("\nCumulative Sequential Times for Each Layer:\n");
    for (int i = 0; i < 6; i++)
    {
        printf("Layer %d: %f seconds\n", i + 1, times[i]);
    }
}