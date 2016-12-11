
#include "OneToOne.h"


template <class LayerType>
OneToOne <LayerType> ::OneToOne(const std::vector<unsigned int> &dimensions,
                                const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                BaseActivationFunction<LayerType> *activationFunction)
        : BaseHiddenLayer <LayerType> (dimensions,
                                       previousLayers,
                                       activationFunction,
                                       nullptr) {
}


template <class LayerType>
void OneToOne <LayerType> :: createWeights()  {

    for( int i=0; i < numberOfNeurons; ++i ) {
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
            NeuronOperations::connectNeurons( layer -> getNeurons()[i],
                                              neurons[currentNeuron],
                                              weights[currentNeuron],
                                              deltaWeights[currentNeuron] );
            ++currentNeuron;
        }
    }
}
