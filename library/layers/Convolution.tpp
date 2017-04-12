
#include <cstdlib>
#include <ctime>
#include "Convolution.h"
#include "../util/NeuronOperations.h"


template <class LayerType>
Convolution <LayerType> :: Convolution(std::vector<unsigned> dimensions,
                                       std::vector<unsigned> kernel,
                                       const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                       std::vector<unsigned int> stride,
                                       NeuronInitializer<LayerType> *neuronInitializer,
                                       Bias<LayerType> *bias)
        : kernel(kernel),
          stride(stride),
          BaseHiddenLayer <LayerType> (dimensions, previousLayers, neuronInitializer, bias) {

    if( this -> stride.empty() )
        stride  = std :: vector <unsigned> ( dimensions.size(), 1 );
}


template <class LayerType>
Convolution <LayerType> :: Convolution(std::vector<unsigned> dimensions,
                                       std::vector<unsigned> kernel,
                                       BaseActivationFunction<LayerType> *activationFunction,
                                       const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                       std::vector<unsigned int> stride,
                                       Bias<LayerType> *bias)
        : Convolution(dimensions,
                      kernel,
                      previousLayers,
                      stride, new SimpleNeuronInitializer <LayerType>(activationFunction),
                      bias) {

}


template <class LayerType>
Convolution <LayerType> :: Convolution(std::vector<unsigned> dimensions,
                                       std::vector<unsigned> kernel,
                                       std::vector<BaseNeuron<LayerType> *> neurons,
                                       const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                       std::vector<unsigned int> stride,
                                       Bias<LayerType> *bias)
        : kernel(kernel),
          stride(stride),
          BaseHiddenLayer <LayerType> (dimensions, previousLayers, neurons, bias){

    if( this -> stride.empty() )
        stride  = std :: vector <unsigned> ( dimensions.size(), 1 );
}




template <class LayerType>
void Convolution <LayerType> :: connectNeurons() {

    for( const BaseLayer <LayerType>* previousLayer : previousLayers ) {
        connectLayer( previousLayer, 0, 0, 0 );
    }
}

template <class LayerType>
void Convolution <LayerType> :: connectOne( BaseNeuron<LayerType> *neuron,
                                            const BaseLayer <LayerType>* previousLayer,
                                            int previousLayerStart,
                                            int currentDimension,
                                            int weightIndex ) {

    if( currentDimension == previousLayer -> getDimensions().size() ) {
        /// connect neuron to the neuron in the previous layer at position [previousLayerStart]
//        printf( "Connect *%d to prev[%d]   --with [%d]--> %lf\n", neuron, previousLayerStart, weightIndex, *weights[weightIndex] );
        connectTwoNeurons( previousLayer -> getNeurons()[previousLayerStart], neuron, weightIndex );
    }
    else {
        int step = math::multiply(previousLayer->getDimensions(), (size_t) currentDimension + 1);
        int weightStep = math::multiply(kernel, (size_t) (currentDimension + 1));
        for( int i=0; i < kernel[ currentDimension ]; ++i ) {
            connectOne( neuron,
                        previousLayer,
                        previousLayerStart + i * step,
                        currentDimension + 1,
                        weightIndex + i * weightStep );
        }
    }
}

template <class LayerType>
void Convolution <LayerType> :: connectLayer( const BaseLayer <LayerType>* previousLayer,
                                              int currentLayerStart,
                                              int previousLayerStart,
                                              int currentDimension,
                                              int weightIndex ) {

    if( currentDimension == dimensions.size() ) {
//        printf( "\n\nConnect this[%d]...\n", currentLayerStart );
        connectOne( neurons[ currentLayerStart ], previousLayer, previousLayerStart, 0, weightIndex );
        if( bias != nullptr ) {
            int biasIndex = weightIndex + math::multiply(kernel); /// connect all kernels and reach the bias term
            NeuronOperations::connectConvolutionalNeurons( bias,
                                                           neurons[ currentLayerStart ],
                                                           numberOfUsages[ biasIndex ],
                                                           weights[ biasIndex ],
                                                           deltaWeights[ biasIndex ] );
        }
    }
    else {
        int currentLayerStep = math::multiply(dimensions, (size_t) currentDimension + 1);
        int previousLayerStep =
                math::multiply(previousLayer->getDimensions(), (size_t) currentDimension + 1) *
                                stride[ currentDimension ];
        int weightStep = currentDimension == 0
                         ? math::multiply(kernel) + ( bias != nullptr ? 1 : 0 )
                         : 0;  /// weights are different for different feature maps
        if( currentDimension == 0 )printf( "WeightStep: %d\n", weightStep );

        for( int i=0; i< dimensions[ currentDimension ]; ++i ) {
            connectLayer( previousLayer,
                          currentLayerStart + i * currentLayerStep,
                          previousLayerStart + i * previousLayerStep,
                          currentDimension + 1,
                          weightIndex + (i * weightStep) );
        }
    }
}


template <class LayerType>
void Convolution <LayerType> :: createWeights() {

    /// number of feature maps * ( size of the kernel + bias )
    /// structure -> [[weights of feature map, biasWeight]]
    int numberOfWeights = dimensions[0] * (math::multiply(kernel) + ( bias != nullptr ? 1 : 0 ) );
    for( int i=0; i < numberOfWeights; ++i ) {
        weights.push_back( new LayerType( LayerType(rand() / LayerType(RAND_MAX) - 0.5) ) );
        deltaWeights.push_back( new LayerType( 0 ) );
        numberOfUsages.push_back( new int( this -> size() ) );
    }


    printf( "\nWeights:->\n" );
    for( auto item : weights ) {
        printf( "%lf\t", *item );
    }
    printf( "\n" );
}

template <class LayerType>
void Convolution <LayerType> :: connectTwoNeurons(BaseNeuron<LayerType> *previousNeuron,
                                                  BaseNeuron<LayerType> *neuron,
                                                  int weightIndex) {


    NeuronOperations::connectConvolutionalNeurons(previousNeuron,
                                                  neuron,
                                                  numberOfUsages[weightIndex],
                                                  weights[weightIndex],
                                                  deltaWeights[weightIndex]);
}
