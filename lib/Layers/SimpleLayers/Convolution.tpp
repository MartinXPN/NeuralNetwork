
#include <cstdlib>
#include <ctime>
#include "Convolution.h"
#include "../../Utilities/MathOperations.h"
#include "../../Utilities/NeuronOperations.h"


template <class LayerType>
Convolution <LayerType> ::Convolution(std::vector<unsigned> dimensions,
                                      std::vector<unsigned> kernel,
                                      BaseActivationFunction<LayerType> *activationFunction,
                                      const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                      std::vector<unsigned> stride,
                                      Bias<LayerType> *bias)
        : kernel( kernel ),
          stride( stride ),
          BaseHiddenLayer <LayerType> ( dimensions, previousLayers, activationFunction, bias ) {

    if( this -> stride.empty() )
        stride  = std :: vector <unsigned> ( dimensions.size(), 1 );


    int numberOfWeights = math :: multiply( kernel );
    for( int i=0; i < numberOfWeights; ++i ) {
        weights.push_back( LayerType(rand() / LayerType(RAND_MAX) - 0.5) );
        deltaWeights.push_back( 0 );
        numberOfUsages.push_back( numberOfNeurons );
    }

    printf( "\nWeights:->\t" );
    for( auto item : weights ) {
        printf( "%lf\t", item );
    }
    printf( "\n" );
}


template <class LayerType>
void Convolution <LayerType> :: connectNeurons() {

    for( const BaseLayer <LayerType>* previousLayer : previousLayers ) {
        connectLayer( previousLayer, 0, 0, 0 );
    }

    if( bias != nullptr ) {
        weights.push_back( LayerType(rand() / LayerType(RAND_MAX) - 0.5) );
        deltaWeights.push_back( 0 );
        numberOfUsages.push_back( numberOfNeurons );

        for( auto neuron : neurons ) {
            NeuronOperations::connectConvolutionalNeurons(bias,
                                                          neuron,
                                                          &numberOfUsages.back(),
                                                          &weights.back(),
                                                          &deltaWeights.back());
        }
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
//        printf( "Connect *%d to prev[%d]   --with [%d]--\n", neuron, previousLayerStart, weightIndex );
        NeuronOperations::connectConvolutionalNeurons(previousLayer->getNeurons()[previousLayerStart],
                                                      neuron,
                                                      &numberOfUsages[weightIndex],
                                                      &weights[weightIndex],
                                                      &deltaWeights[weightIndex]);
    }
    else {
        int step = math::multiply( previousLayer -> getDimensions(), (size_t) currentDimension + 1 );
        int weightStep = math :: multiply( kernel, (size_t) (currentDimension + 1) );
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
                                              int currentDimension) {

    if( currentDimension == dimensions.size() ) {
//        printf( "\n\nConnect this[%d]!!!\n", currentLayerStart );
        connectOne( neurons[ currentLayerStart ], previousLayer, previousLayerStart, 0 );
    }
    else {
        int currentLayerStep = math :: multiply( dimensions, (size_t) currentDimension + 1 );
        int previousLayerStep = math :: multiply( previousLayer -> getDimensions(), (size_t) currentDimension + 1 ) *
                                stride[ currentDimension ];

        for( int i=0; i< dimensions[ currentDimension ]; ++i ) {
            connectLayer( previousLayer,
                          currentLayerStart + i * currentLayerStep,
                          previousLayerStart + i * previousLayerStep,
                          currentDimension + 1 );
        }
    }
}
