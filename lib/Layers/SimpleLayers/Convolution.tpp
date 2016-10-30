
#include "Convolution.h"


template <class LayerType>
Convolution <LayerType> ::Convolution(unsigned numberOfFilters,
                                      std::vector<unsigned> dimensions,
                                      std::vector<unsigned> windowSize,
                                      BaseActivationFunction<LayerType> *activationFunction,
                                      const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                      std::vector<unsigned> stride,
                                      std::vector<unsigned> padding,
                                      Bias<LayerType> *bias)
        : numberOfFilters( numberOfFilters ),
          dimensions( dimensions ),
          windowSize( windowSize ),
          stride( stride ),
          padding( padding ),
          BaseHiddenLayer( calculateNumberOfNeurons(), previousLayers, activationFunction, bias ) {

    if( this -> stride.empty() )    stride  = std :: vector <unsigned> ( dimensions.size(), 1 );
    if( this -> padding.empty() )   padding = std :: vector <unsigned> ( dimensions.size(), 0 );

    for( unsigned item : stride )
        assert( item != 0 );
}

template <class LayerType>
unsigned Convolution <LayerType> :: calculateNumberOfNeurons() {

    unsigned res = numberOfFilters;
    for( unsigned dimension : dimensions ) {
        res *= dimension;
    }
    return res;
}


template <class LayerType>
void Convolution <LayerType> :: createNeurons() {

    for( int i=0; i < numberOfNeurons; ++i )
        neurons.push_back( new BaseNeuron <LayerType> ( activationFunction ) );
}


// TODO implement connectNeurons and connectOne
template <class LayerType>
void Convolution <LayerType> :: connectOne( int dimension ) {

    if( dimension == 0 ) {

    }

    for( int i=0; i < dimensions[ dimension-1 ]; ++i ) {
        connectOne( dimension - 1 );
    }
}

template <class LayerType>
void Convolution <LayerType> :: connectNeurons() {

    for( int i=0; i < numberOfFilters; ++i ) {
        connectOne( dimensions.size() );
    }
}
