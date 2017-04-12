
#include "OneToOne.h"
#include "../util/NeuronOperations.h"


template <class LayerType>
void OneToOne <LayerType> :: createWeights()  {

    for( int i=0; i < this -> size(); ++i ) {
        weights.push_back(new LayerType(LayerType(rand() / LayerType(RAND_MAX) - 0.5)));
        deltaWeights.push_back(new LayerType(0));
    }
}


template <class LayerType>
void OneToOne <LayerType> :: connectNeurons() {

    int currentNeuron = 0;
    for( auto layer : previousLayers ) {
        for( int i=0; i < layer -> size(); ++i ) {
            // printf( "Connect %d -> %d\n", i, currentNeuron );
            connectNeurons( layer -> getNeurons()[i],
                            neurons[currentNeuron],
                            weights[currentNeuron],
                            deltaWeights[currentNeuron] );
            ++currentNeuron;
        }
    }
}
