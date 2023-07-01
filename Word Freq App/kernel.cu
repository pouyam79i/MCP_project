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
        text[t_idx][0] = ' ';
        text[t_idx][1] = ' ';
        text[t_idx][2] = ' ';
        text[t_idx][3] = ' ';
        text[t_idx][4] = '1';
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
	char *word;
	for (l=0; l<length; l++)
	{
        char *word = text[l];
        for (i=0; i<k_num; i++){
            bool is_a_match = (word[4]=='0') && (word[3]==k_words[i][3]) && (word[2]==k_words[i][2]) && (word[1]==k_words[i][1]) && (word[0]==k_words[i][0]);
            if (is_a_match){
                matches[i] += 1;
                //break;
            }
        }
	}
}

// *** cuda functions ***
cudaError_t findFreqWithGPU(    
    char *text, 
    unsigned int length,    // length of text
    char *k_words,
    unsigned int k_num,     // number of key words
    unsigned int *matches
    );
// kernel function
__global__ void kernel( 
    char *text, 
    unsigned int length,     // length of text
    char *k_words,
    unsigned int k_num,      // number of key words
    unsigned int *matches,
    unsigned int chunck_size // number of words (5-byte check)
)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    idx *= chunck_size * 5;
    int l, i;
	char w0, w1, w2, w3, w4;
	for (l= idx; l<idx+chunck_size*5; l=l+5)
	{
        w0 = text[l];
        w1 = text[l+1];
        w2 = text[l+2];
        w3 = text[l+3];
        w4 = text[l+4];
        for (i=0; i<k_num; i=i+5){
            if ((w4 == '0') && (w3 == k_words[i + 3]) && (w2 == k_words[i + 2]) && (w1 == k_words[i + 1]) && (w0 == k_words[i + 0])){
                atomicAdd(&(matches[i/5]), 1);
                //break;
            }
        }
	}
}


void printResult(char** k_words, unsigned int *matches, unsigned int len) {
    for (int i = 0; i < len; i++) {
        if (k_words[i][0] == ' ') { continue; }
        printf("%c%c%c%c:%d\n", k_words[i][0], k_words[i][1], k_words[i][2], k_words[i][3], matches[i]);
    }
    printf("\n\n\n");
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
	const clock_t begin_time = clock();
	findFreqWithCPU(text, MAX_WORDS, k_words, MAX_KEYS, matches);
	float runTime = (float)( clock() - begin_time ) /  CLOCKS_PER_SEC;
	printf("CPU runtime: %fs\n", runTime);
    printf("CPU result:\n");
    printResult(k_words, matches, MAX_KEYS);

    // *** GPU execution
    char* text_1d = (char*)calloc(MAX_WORDS*5, sizeof(char));
    char* k_words_1d = (char*)calloc(MAX_KEYS*5, sizeof(char));
    for (int i = 0; i < MAX_WORDS * 5; i = i + 5)
    {
        text_1d[i] = text[i / 5][0];
        text_1d[i + 1] = text[i / 5][1];
        text_1d[i + 2] = text[i / 5][2];
        text_1d[i + 3] = text[i / 5][3];
        text_1d[i + 4] = text[i / 5][4];
    }

    for (int i = 0; i < MAX_KEYS * 5; i = i + 5)
    {
        k_words_1d[i] = k_words[i / 5][0];
        k_words_1d[i + 1] = k_words[i / 5][1];
        k_words_1d[i + 2] = k_words[i / 5][2];
        k_words_1d[i + 3] = k_words[i / 5][3];
        k_words_1d[i + 4] = k_words[i / 5][4];
    }

    for (int i = 0; i < MAX_KEYS; i++)
    {
        matches[i] = 0;
    }

    cudaError_t cudaStatus = findFreqWithGPU(text_1d, MAX_WORDS, k_words_1d, MAX_KEYS, matches);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    printf("GPU result:\n");
    printResult(k_words, matches, MAX_KEYS);
    // cudaDeviceReset must be called before exiting in order for profiling and
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


    return 0;
}


 // Helper function for using CUDA to add vectors in parallel.
 cudaError_t findFreqWithGPU(
 char *text, 
 unsigned int length,    // length of text
 char *k_words,
 unsigned int k_num,     // number of key words
 unsigned int *matches
 )
 {

     char *dev_text;
     char *dev_k_words;
     unsigned int *dev_matches = 0;
     cudaError_t cudaStatus;

     // Choose which GPU to run on, change this on a multi-GPU system.
     cudaStatus = cudaSetDevice(0);
     if (cudaStatus != cudaSuccess) {
         fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
         goto Error;
     }
      

     // Allocate memory for text list.
     cudaStatus = cudaMalloc((void**)&dev_text,5 * MAX_WORDS * sizeof(char));
     if (cudaStatus != cudaSuccess) {
         fprintf(stderr, "cudaMalloc failed!");
         goto Error;
     }


     // Allocate memory for key word list.
     cudaStatus = cudaMalloc((void**)&dev_k_words, 5 * MAX_KEYS * sizeof(char));
     if (cudaStatus != cudaSuccess) {
         fprintf(stderr, "cudaMalloc failed!");
         goto Error;
     }

     // allocate memory for matches
     cudaStatus = cudaMalloc((void**)&dev_matches, MAX_KEYS * sizeof(unsigned int));
     if (cudaStatus != cudaSuccess) {
         fprintf(stderr, "cudaMalloc failed!");
         goto Error;
     }

    // cp data to gpu
    cudaStatus = cudaMemcpy(dev_text, text, 5 * MAX_WORDS * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_k_words, k_words, 5 * MAX_KEYS * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_matches, matches, MAX_KEYS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


     dim3 DimGrid(25, 1, 1);
     dim3 DimBlock(128, 1, 1);
     int chunck_size = MAX_WORDS / 3200; // 1875
     
     const clock_t begin_time = clock();

     // Launch a kernel on the GPU with one thread for each element.
     kernel<<<DimGrid, DimBlock>>>(
         dev_text,
         length*5,    // length of text
         dev_k_words,
         k_num*5,     // number of key words
         dev_matches,
         chunck_size
         );

     // Check for any errors launching the kernel
     cudaStatus = cudaGetLastError();
     if (cudaStatus != cudaSuccess) {
         fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
         goto Error;
     }
    
     // cudaDeviceSynchronize waits for the kernel to finish, and returns
     // any errors encountered during the launch.
     cudaStatus = cudaDeviceSynchronize();
     if (cudaStatus != cudaSuccess) {
         fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
         goto Error;
     }

     float runTime = (float)( clock() - begin_time ) /  CLOCKS_PER_SEC;
     printf("GPU runtime: %fs\n", runTime);

     // Copy output vector from GPU buffer to host memory.
     cudaStatus = cudaMemcpy(matches, dev_matches, MAX_KEYS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
     if (cudaStatus != cudaSuccess) {
         fprintf(stderr, "cudaMemcpy failed!");
         goto Error;
     }

 Error:
     cudaFree(dev_text);
     cudaFree(dev_k_words);
     cudaFree(dev_matches);
    
     return cudaStatus;
 }

