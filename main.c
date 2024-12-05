#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>  // Include MPI header

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

// void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size, int rank, int size)
// {
//     for (int i = rank, percent = 0; i <= total_size - batch_size; i += size)
//     {
//         TrainBatch(lenet, train_data + i, train_label + i, batch_size);
//         if (i * 100 / total_size > percent)
//             printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
//     }
// }

// int testing(LeNet5 *lenet, image *test_data, uint8 *test_label, int total_size, int rank, int size)
// {
//     int right = 0, percent = 0;
//     for (int i = rank; i < total_size; i += size)
//     {
//         uint8 l = test_label[i];
//         int p = Predict(lenet, test_data[i], 10);
//         right += l == p;
//         if (i * 100 / total_size > percent)
//             printf("test:%2d%%\n", percent = i * 100 / total_size);
//     }
//     int total_right = 0;
//     MPI_Reduce(&right, &total_right, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);  // Reduce to root process
//     return total_right;
// }

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
    for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
    {
        TrainBatch(lenet, train_data + i, train_label + i, batch_size);
        if (i * 100 / total_size > percent)
            printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
    }
}

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label, int total_size)
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


void foo(int rank, int size)
{
    // image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
    // uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
    // image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
    // uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));

    // if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
    // {
    //     printf("ERROR!!!\nDataset File Not Find! Please Copy Dataset to the Folder Included the exe\n");
    //     free(train_data);
    //     free(train_label);
    //     system("pause");
    // }
    // if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
    // {
    //     printf("ERROR!!!\nDataset File Not Find! Please Copy Dataset to the Folder Included the exe\n");
    //     free(test_data);
    //     free(test_label);
    //     system("pause");
    // }

    // LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
    // if (load(lenet, LENET_FILE))
    //     Initial(lenet);
    
    // clock_t start = clock();
    // int batches[] = { 300 };

    // // Parallelize the training across processes
    // for (int i = 0; i < sizeof(batches) / sizeof(*batches); ++i)
    //     training(lenet, train_data, train_label, batches[i], COUNT_TRAIN, rank, size);

    // // Parallelize the testing across processes
    // int right = testing(lenet, test_data, test_label, COUNT_TEST, rank, size);

    // if (rank == 0)
    // {
    //     printf("%d/%d\n", right, COUNT_TEST);
    //     printf("Time: %u\n", (unsigned)(clock() - start));
    //     // Optionally save the model after training
    //     //save(lenet, LENET_FILE);
    // }

    // free(lenet);
    // free(train_data);
    // free(train_label);
    // free(test_data);
    // free(test_label);
    // system("pause");
    image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
    uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
    image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
    uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
    if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
    {
        if (rank == 0) // Only print errors from rank 0
            printf("ERROR!!!\nDataset File Not Found! Please Copy Dataset to the Folder Including the exe\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
    {
        if (rank == 0)
            printf("ERROR!!!\nDataset File Not Found! Please Copy Dataset to the Folder Including the exe\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
    if (load(lenet, LENET_FILE))
        Initial(lenet);

    if (rank == 0) // Print timing information only from rank 0
        printf("Training started...\n");

    clock_t start = clock();
    int batches[] = { 300 };
    for (int i = 0; i < sizeof(batches) / sizeof(*batches); ++i)
        training(lenet, train_data, train_label, batches[i], COUNT_TRAIN);
    int right = testing(lenet, test_data, test_label, COUNT_TEST);

    if (rank == 0)
    {
        printf("%d/%d\n", right, COUNT_TEST);
        printf("Time:%u\n", (unsigned)((clock() - start) / CLOCKS_PER_SEC));
        // save(lenet, LENET_FILE);
    }

    free(lenet);
    free(train_data);
    free(train_label);
    free(test_data);
    free(test_label);
}


// int main(int argc, char *argv[])
// {
//     // Initialize MPI
//     MPI_Init(&argc, &argv); 

//     // Get the rank of the current process
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the current process
//     MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the number of processes

//     // Call the training/testing function
//     foo(rank, size); 

//     // Finalize MPI
//     MPI_Finalize();  // Finalize MPI
    
//     return 0;
// }

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);  // Initialize MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the number of processes

    // Record start time
    double start_time = MPI_Wtime();

    // Call your function that includes parallel computation
    foo(rank, size);

    // Record end time
    double end_time = MPI_Wtime();

    // Calculate elapsed time
    double elapsed_time = end_time - start_time;
    if (rank == 0) {  // Only print from the root process (rank 0)
        printf("Overall execution time: %f seconds\n", elapsed_time);
    }

    MPI_Finalize();  // Finalize MPI
    return 0;
}