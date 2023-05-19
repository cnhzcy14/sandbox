#include "gaussian.h"
#include "args.h"
#include <stdbool.h>
#include <sys/time.h>
#include <stdio.h>
int main ( int argc, char *argv[] )
{
    double units;
    uint32_t gaussianSize;
    float gaussianSigma;
    char* imgname;//the name of the image file, taken by the arguments
    struct timeval tpend1, tpend2;
    long usec1 = 0;

    //read in the program's arguments
    if(readArguments(argc,argv,&imgname,&gaussianSize,&gaussianSigma)==false)
        return -1;
//    gettimeofday(&tpend1, 0);

//    //perform CPU blurring
//    if(pna_blur_cpu(imgname,gaussianSize,gaussianSigma)==false)//time it
//        return -2;
//    gettimeofday(&tpend2, 0);
//    usec1 = 1000 * (tpend2.tv_sec - tpend1.tv_sec) + (tpend2.tv_usec - tpend1.tv_usec) / 1000;
//    printf("pna_blur_cpu use time=%ld ms \n",usec1);
        
    //perform GPU blurring and then read the timer
    gettimeofday(&tpend1, 0);
    if(pna_blur_gpu(imgname,gaussianSize,gaussianSigma)==false)
        return -3;
    gettimeofday(&tpend2, 0);
    usec1 = 1000 * (tpend2.tv_sec - tpend1.tv_sec) + (tpend2.tv_usec - tpend1.tv_usec) / 1000;
    printf("pna_blur_gpu only use time=%ld ms \n",usec1); 

    return 0;
}
