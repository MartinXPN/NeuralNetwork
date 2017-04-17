
#include <vector>
#include "FullyConnected.h"
#include "../util/NeuronOperations.h"


template <class LayerType>
void FullyConnected <LayerType> :: connectNeurons() {

    for( auto previousLayer : previousLayers ) {
        for( int i=0; i < size(); ++i ) {
            for( int j=0; j < previousLayer -> size(); ++j ) {
                connectNeurons( previousLayer -> getNeurons()[j],
                                neurons[i],
                                weights[ i * previousLayer -> size() + j ],
                                deltaWeights[ i * previousLayer -> size() + j ] );
            }
        }
    }

    if( bias != nullptr ) {
        int start = (int) (weights.size() - size());
        for( int i=0; i < size(); ++i ) {
            connectNeurons( bias,
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
