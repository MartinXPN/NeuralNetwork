
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
void FullyConnected <LayerType> ::connectNeurons() {

    for( auto currentNeuron : neurons ) {

        /// connect neuron to all neurons in all previous layers
        for( auto previousLayer : previousLayers )
            for (auto previousNeuron : previousLayer -> getNeurons())
                NeuronOperations::connectNeurons( previousNeuron, currentNeuron );

        if( bias != nullptr ) {
            NeuronOperations::connectNeurons( bias, currentNeuron );
        }
    }
}
