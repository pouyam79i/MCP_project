// Libs
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// *** difinitions ***
#define MAX_WORDS 6000000 // file size 
#define MAX_KEYS 32

// *** file address ***
static const char FILE_KEYWORDS[] = "../data/keywords.txt";
static const char FILE_DATASET[] =  "../data/large.txt";

// *** app required functions ***
int str_length(char str[]) {
    // initializing count variable (stores the length of the string)
    int count; 
    // incrementing the count till the end of the string
    for (count = 0; str[count] != '\0'; ++count);
    // returning the character count of the string
    return count; 
}
void read_word_by_word (FILE *f, char **text, unsigned int len) {
    char word[1024];
    unsigned int i, t_idx = 0;
    /* assumes no word exceeds length of 1023 */
    while (fscanf(f, " %1023s", word) == 1) {
        text[i][0] = ' '; 
        text[i][1] = ' '; 
        text[i][2] = ' '; 
        text[i][3] = ' '; 
        text[i][4] = '1'; 
        int l = str_length(word);
        for(i=0; i<l && i < 4; i++){
            text[t_idx][i] = word[i];
        }
        if(l == 4){
            text[t_idx][4] = '0';
        }
        t_idx++;
        if (t_idx == len)
            break;
    }
}

// *** cpu functions ***
void findFreqWithCPU(
    char **text, 
    unsigned int length,    // length of text
    char **k_words,
    unsigned int k_num,     // number of key words
    unsigned int *matches
) 
{
    int l, i;
	unsigned int word;
	for (l=0; l<length; l++)
	{
        char *word = text[l];
        for (i=0; i<k_num; i++){
            bool is_a_match = (word[4]=='0') && (word[3]==k_words[i][3]) && (word[2]==k_words[i][2]) && (word[1]==k_words[i][1]) && (word[0]==k_words[i][0]);
            if (is_a_match){
                matches[i] += 1;
            }
        }
	}
}

// *** cuda functions ***
cudaError_t findFreqWithGPU(    
    char **text, 
    unsigned int length,    // length of text
    char **k_words,
    unsigned int k_num,     // number of key words
    unsigned int *matches
    );
// kernel function
__global__ void kernel( 
    char **text, 
    unsigned int length,    // length of text
    char **k_words,
    unsigned int k_num,     // number of key words
    unsigned int *matches
)
{
    
}


void printResult(char** k_words, unsigned int *matches, unsigned int len) {
    for (int i = 0; i < len; i++) {
        if (k_words[i][0] == ' ') { continue; }
        printf("%c%c%c%c:%d\n", k_words[i][0], k_words[i][1], k_words[i][2], k_words[i][3], matches[i]);
    }
}



int main()
{

    // *** Preprocessing

    // alocate space
    unsigned int *matches = (unsigned int*)calloc(MAX_KEYS, sizeof(unsigned int));
    char **text    = (char**) calloc(MAX_WORDS, sizeof(char*));
    char **k_words = (char**) calloc(MAX_KEYS, sizeof(char*));
    int i;
    for ( i = 0; i < MAX_WORDS; i++ )
    {
        text[i] = (char*) calloc(5, sizeof(char));
    }
    for ( i = 0; i < MAX_KEYS; i++ )
    {
        k_words[i] = (char*) calloc(5, sizeof(char));
    }

    // read from text files:
    FILE *fp;
    // read from keywords
    fp = fopen(FILE_KEYWORDS,"r");
	if (!fp)
	{	printf("Unable to open keyword file.\n");	exit(0);}
    read_word_by_word(fp, k_words, MAX_KEYS);
	fclose(fp);
    // read from text
    fp = fopen(FILE_DATASET,"r");
	if (!fp)
	{	printf("Unable to open dataset file.\n");	exit(0);}
    read_word_by_word(fp, text, MAX_WORDS);
	fclose(fp);

    // *** CPU execution
	//const clock_t begin_time = clock();
	findFreqWithCPU(text, MAX_WORDS, k_words, MAX_KEYS, matches);
	//float runTime = (float)( clock() - begin_time ) /  CLOCKS_PER_SEC;
	//printf("CPU runtime: %fs\n", runTime);
    printResult(k_words, matches, MAX_KEYS);

    // *** GPU execution



    // ****** RUN TIME REF:

    // // Add vectors in parallel.
    // cudaError_t cudaStatus = findFreqWithGPU();
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "addWithCuda failed!");
    //     return 1;
    // }

    // // cudaDeviceReset must be called before exiting in order for profiling and
    // // tracing tools such as Nsight and Visual Profiler to show complete traces.
    // cudaStatus = cudaDeviceReset();
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "cudaDeviceReset failed!");
    //     return 1;
    // }

    return 0;
}




// // Helper function for using CUDA to add vectors in parallel.
// cudaError_t findFreqWithGPU(
// char **text, 
// unsigned int length,    // length of text
// char **k_words,
// unsigned int k_num,     // number of key words
// unsigned int *matches
// )
// {
//     int *dev_a = 0;
//     int *dev_b = 0;
//     int *dev_c = 0;
//     cudaError_t cudaStatus;

//     // Choose which GPU to run on, change this on a multi-GPU system.
//     cudaStatus = cudaSetDevice(0);
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//         goto Error;
//     }

//     // Allocate GPU buffers for three vectors (two input, one output)    .
//     cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaMalloc failed!");
//         goto Error;
//     }

//     cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaMalloc failed!");
//         goto Error;
//     }

//     cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaMalloc failed!");
//         goto Error;
//     }

//     // Copy input vectors from host memory to GPU buffers.
//     cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaMemcpy failed!");
//         goto Error;
//     }

//     cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaMemcpy failed!");
//         goto Error;
//     }

//     // Launch a kernel on the GPU with one thread for each element.
//     kernel<<<1, size>>>(dev_c, dev_a, dev_b);

//     // Check for any errors launching the kernel
//     cudaStatus = cudaGetLastError();
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//         goto Error;
//     }
    
//     // cudaDeviceSynchronize waits for the kernel to finish, and returns
//     // any errors encountered during the launch.
//     cudaStatus = cudaDeviceSynchronize();
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//         goto Error;
//     }

//     // Copy output vector from GPU buffer to host memory.
//     cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaMemcpy failed!");
//         goto Error;
//     }

// Error:
//     cudaFree(dev_c);
//     cudaFree(dev_a);
//     cudaFree(dev_b);
    
//     return cudaStatus;
// }
