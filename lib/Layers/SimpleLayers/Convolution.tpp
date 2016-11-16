
#include "Convolution.h"
#include "../../Utilities/MathOperations.h"


template <class LayerType>
Convolution <LayerType> ::Convolution(std::vector<unsigned> dimensions,
                                      std::vector<unsigned> kernel,
                                      BaseActivationFunction<LayerType> *activationFunction,
                                      const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                      std::vector<unsigned> stride,
                                      Bias<LayerType> *bias)
        : dimensions( dimensions ),
          kernel( kernel ),
          stride( stride ),
          BaseHiddenLayer( dimensions, previousLayers, activationFunction, bias ) {

    if( this -> stride.empty() )
        stride  = std :: vector <unsigned> ( dimensions.size(), 1 );

    for( unsigned item : stride )
        assert( item != 0 );
}


template <class LayerType>
void Convolution <LayerType> :: connectNeurons() {


}

template <class LayerType>
void Convolution <LayerType> :: connectOne( BaseInputNeuron<LayerType> *&neuron,
                                            BaseLayer <LayerType>* previousLayer,
                                            int previousLayerStart,
                                            int currentDimension ) {

    if( currentDimension == previousLayer -> dimensions.size() ) {
        /// connect neuron to the neuron in the previous layer at position [previousLayerStart]
    }
    else {
        int step = math :: multiply( previousLayer -> dimensions, (size_t) currentDimension + 1 );
        for( int i=0; i < kernel[ currentDimension ]; ++i ) {
            connectOne( neuron,
                        previousLayer,
                        previousLayerStart + i * step,
                        currentDimension + 1 );
        }
    }
}

template <class LayerType>
void Convolution <LayerType> :: connectLayer( BaseLayer <LayerType>* previousLayer,
                                              int currentLayerStart,
                                              int previousLayerStart,
                                              int currentDimension) {

    if( currentDimension == dimensions.size() ) {
        connectOne( neurons[ currentLayerStart], previousLayer, previousLayerStart, 0 );
    }
    else {
        int currentLayerStep = math :: multiply( dimensions, (size_t) currentDimension + 1 );
        int previousLayerStep = math :: multiply( previousLayer -> dimensions, (size_t) currentDimension + 1 ) *
                                stride[ currentDimension ];

        for( int i=0; i< dimensions[ currentDimension ]; ++i ) {
            connectLayer( previousLayer,
                          currentLayerStart + i * currentLayerStep,
                          previousLayerStart + i * previousLayerStep,
                          currentDimension + 1 );
        }
    }
}
