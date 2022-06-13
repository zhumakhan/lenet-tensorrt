#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INSIZE 28

typedef struct mnist_data{
	float data[INSIZE][INSIZE];
	unsigned int label;
}mnist_data;


const char *train_label_filename="data/train-labels.idx1-ubyte";
const char *train_image_filename="data/train-images.idx3-ubyte";

const char *test_label_filename="data/t10k-labels.idx1-ubyte";
const char *test_image_filename="data/t10k-images.idx3-ubyte";

//https://bytes.com/topic/c/answers/577180-converting-big-endian-little-endian
static unsigned int swap_endian(unsigned int b){
	return (b>>24) | ((b>>8) & 0x0000ff00) | ((b<<8) & 0x00ff0000) | (b<<24);
}

static unsigned int mnist_bin_to_int(char* tmp){
	return (*tmp) & 0x000000ff;
}

static int mnist_load(const char *image_filename, const char *label_filename, mnist_data **data_set, unsigned int *count);

static int mnist_load(const char *image_filename, const char *label_filename, mnist_data **data_set, unsigned int *count){
	FILE *image_file = fopen(image_filename,"rb");
	FILE *label_file = fopen(label_filename,"rb");

	unsigned int image_magic, image_number, image_row, image_column;
	unsigned int label_magic, label_number;
	
	fread((void*)&image_magic,4,1,image_file);
	fread((void*)&image_number,4,1,image_file);
	fread((void*)&image_row,4,1,image_file);
	fread((void*)&image_column,4,1,image_file);
	
	image_magic =  swap_endian(image_magic);
	image_number =  swap_endian(image_number);
	image_row =  swap_endian(image_row);
	image_column =  swap_endian(image_column);

	fread((void*)&label_magic,4,1,label_file);
	fread((void*)&label_number,4,1,label_file);
	
	label_magic =  swap_endian(label_magic);
	label_number =  swap_endian(label_number);


	printf("image magic number = %d (should be 2051)\n",image_magic);
	printf("label magic number = %d (should be 2049)\n",label_magic);
	printf("image total number = %d (should be 10000)\n",image_number);
	printf("label total number = %d (should be 10000)\n",label_number);
	printf("rows = %d cols = %d (both should be 28)\n",image_row,image_column);
	
	(*data_set) = (mnist_data*)malloc(sizeof(mnist_data) * image_number);

	char image[INSIZE*INSIZE];
	char *labels = (char*)malloc(image_number * sizeof(char));
	
	if(image_number != fread((void*)labels,1,image_number,label_file)){
		printf("Error while reading labels");
	}
	
	*count = 0;
	int read_size;
	for(int k=0;k<image_number;k++){
		
		read_size = fread((void*)&image,1,INSIZE*INSIZE,image_file);
		if(read_size == INSIZE*INSIZE)
			(*count)++;
		else{
			printf("Error while reading image");
			break;
		}
		for(int i=0;i<image_row;i++){
			for(int j=0;j<image_column;j++){
				(*data_set)[k].data[i][j]=((float)mnist_bin_to_int(&image[i*INSIZE+j]))/255;
				(*data_set)[k].label = mnist_bin_to_int(&labels[k]);
			}
		}
	}
	free(labels);
	fclose(image_file);
	fclose(label_file);
	return 0;
}