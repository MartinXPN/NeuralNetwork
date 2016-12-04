
#include "FullyConnected.h"
#include "../../Utilities/NeuronOperations.h"


template <class LayerType>
FullyConnected <LayerType> :: FullyConnected(const std :: vector <unsigned>& dimensions,
                                             BaseActivationFunction<LayerType> *activationFunction,
                                             const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                                             Bias <LayerType>* bias) :
        BaseHiddenLayer <LayerType> ( dimensions,
                                      previousLayers,
                                      activationFunction,
                                      bias ) {

}

template <class LayerType>
void FullyConnected <LayerType> :: connectNeurons() {

    for( auto previousLayer : previousLayers ) {
        for( int i=0; i < size(); ++i ) {
            for( int j=0; j < previousLayer -> size(); ++j ) {
                NeuronOperations::connectNeurons( previousLayer -> getNeurons()[j],
                                                  neurons[i],
                                                  weights[ i * previousLayer -> size() + j ],
                                                  deltaWeights[ i * previousLayer -> size() + j ] );
            }
        }
    }

    if( bias != nullptr ) {
        int start = (int) (weights.size() - size());
        for( int i=0; i < size(); ++i ) {
            NeuronOperations::connectNeurons( bias,
                                              neurons[i],
                                              weights[ start + i ],
                                              deltaWeights[ start + i ] );
        }
    }
}

template <class LayerType>
void FullyConnected <LayerType> :: createWeights() {

    for( auto previousLayer : previousLayers ) {
        for( int i=0; i < previousLayer -> size() * size(); ++i ) {
            weights.push_back( new LayerType( LayerType(rand() / LayerType(RAND_MAX) - 0.5) ) );
            deltaWeights.push_back( new LayerType( 0 ) );
        }
    }

    if( bias != nullptr )
        for( int i=0; i < size(); ++i ) {
            weights.push_back( new LayerType( LayerType(rand() / LayerType(RAND_MAX) - 0.5) ) );
            deltaWeights.push_back( new LayerType( 0 ) );
        }
}
