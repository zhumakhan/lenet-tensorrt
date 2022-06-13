#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <iostream>
#include <cstring>

#include "slenet_params.h"
#include "read.h"



using namespace nvinfer1;

class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        // if (severity <= Severity::kWARNING)
        // std::cout << msg << std::endl;
    }
} logger;


int main(){
    IBuilder * builder = createInferBuilder(logger);
    std::cout<<"Float16 support: "<<builder->platformHasFastFp16()<<std::endl;
    std::cout<<"Int8 support: "<<builder->platformHasFastInt8()<<std::endl;
    
    INetworkDefinition * network = builder->createNetworkV2(0);
    IBuilderConfig * config = builder->createBuilderConfig();
    
    // config->setFlag(BuilderFlag::kDEBUG);
    builder->setMaxBatchSize(32);
    // config->setFlag(BuilderFlag::kFP16);
    
    ITensor * input = network->addInput("in", DataType::kFLOAT, Dims3{1,28,28});

    
    //conv layer
    Weights conv_w{DataType::kFLOAT, c1_weight, 6*25};
    Weights conv_b{DataType::kFLOAT, c1_bias, 6};

    IConvolutionLayer * conv_layer = network->addConvolutionNd(
        *input,
        6,
        Dims2{5,5},
        conv_w,
        conv_b
    );
    conv_layer->setStride(DimsHW{1,1});
    conv_layer->setPadding(DimsHW{0,0});
    conv_layer->setNbGroups(1);
    conv_layer->setName("Conv0");

    input = conv_layer->getOutput(0);
    input->setName("Conv0_out");

    //sigmoid activation
    IActivationLayer * act_layer = network->addActivation(*input, ActivationType::kSIGMOID);
    act_layer->setName("Act0");
    input = act_layer->getOutput(0);
    input->setName("Act0_out");


    //subsampling layer

    float pool_weights[6][16];
    float pool_biases[6];
    for(int i = 0; i < 6; i++){
        std::memcpy(pool_weights[i], s2_weight[0], 16);
        std::memcpy(&pool_biases[i], &s2_bias[0],1);
    }
    Weights pool_w{DataType::kFLOAT, pool_weights, 6*16};
    Weights pool_b{DataType::kFLOAT, pool_biases, 6};

    IConvolutionLayer * pool_layer = network->addConvolutionNd(
        *input,
        6,
        Dims2{4,4},
        pool_w,
        pool_b
    );
    pool_layer->setStride(DimsHW{4,4});
    pool_layer->setPadding(DimsHW{0,0});
    pool_layer->setNbGroups(6);
    pool_layer->setName("Pool0");

    input = pool_layer->getOutput(0);
    input->setName("Pool0_out");

    //sigmoid activation
    act_layer = network->addActivation(*input, ActivationType::kSIGMOID);
    act_layer->setName("Act1");
    input = act_layer->getOutput(0);
    input->setName("Act1_out");

    //fully connected layer
    Weights fc_w {DataType::kFLOAT, f3_weight, 10*6*6*6};
    Weights fc_b {DataType::kFLOAT, f3_bias, 10};

    IFullyConnectedLayer * fc_layer = network->addFullyConnected(*input, 10, fc_w, fc_b);
    fc_layer->setName("Fc0");

    input = fc_layer->getOutput(0);
    input->setName("out");
    
    network->markOutput(*input);


    //build an inference engine from the network
    // IHostMemory * serialized_engine = builder->buildSerializedNetwork(*network, *config);
    ICudaEngine * engine = builder->buildEngineWithConfig(*network, *config);
    IExecutionContext * context = engine->createExecutionContext();
    
    std::cout<<"input/output numbers: "<<engine->getNbBindings()<<std::endl;
    int buf_input_idx = engine->getBindingIndex("in");
    int buf_output_idx = engine->getBindingIndex("out");
    std::cout<<"Number of layers in network: "<<network->getNbLayers()<<std::endl;
    std::cout<<"Number of layers in engine: "<<engine->getNbLayers()<<std::endl;
    
    //Memory management
    void * buffer[2];//binding memory space of input and output
    float output[10];

    cudaMalloc(&buffer[0], 1*28*28*sizeof(float));
    cudaMalloc(&buffer[1], 10*sizeof(float));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    mnist_data* data_set;
	unsigned int count=0;
	int read=mnist_load(test_image_filename,test_label_filename,&data_set,&count);
	
	if(read == 0){
		printf("test_cnt = %d (should be 10000)\n\n",count);
	}


	unsigned int error = 0;
	unsigned int max = 0;
	float time_taken=0;
	for(int i=0;i<count;i++){
        cudaMemcpyAsync(buffer[buf_input_idx], data_set[i].data, 1*28*28*sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        context->enqueue(1, buffer, stream, nullptr);
        cudaStreamSynchronize(stream); 
        cudaMemcpyAsync(output, buffer[buf_output_idx], 10*sizeof(float), cudaMemcpyDeviceToHost, stream);
		
        for(int j=0;j<10;j++){
			if (output[max] < output[j]){
				max = j;
			}
		}

		if (max != data_set[i].label) ++error;

	}
	printf("Error Rate = %f%% (%d out of 10,000)\n", float(error)/float(count)*100.0, error);
	printf("Accuracy = %.3f%% (%d out of 10,000)\n",100.0 - float(error)/float(count)*100.0, count - error);
	printf("Ex time = %f (ms) \n", time_taken);

	free(data_set);
    cudaFree(buffer[0]);
    cudaFree(buffer[1]);
    context->destroy();
    engine->destroy();

	return 0;

}