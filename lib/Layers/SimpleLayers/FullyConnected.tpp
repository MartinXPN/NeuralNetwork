
#include "FullyConnected.h"
#include "../../Utilities/NeuronOperations.h"


template <class LayerType>
FullyConnected <LayerType> :: FullyConnected(unsigned int numberOfNeurons,
                                             const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                                             BaseActivationFunction<LayerType> *activationFunction,
                                             BaseBias <LayerType>* bias) :
        BaseHiddenLayer <LayerType> ( numberOfNeurons,
                                      previousLayers,
                                      activationFunction,
                                      bias ) {

}

template <class LayerType>
void FullyConnected <LayerType> :: createNeurons(unsigned numberOfNeurons,
                                                 BaseActivationFunction<LayerType> *activationFunction)  {

    for( int i=0; i < numberOfNeurons; ++i )
        neurons.push_back( new BaseNeuron <LayerType> ( activationFunction ) );
}

template <class LayerType>
void FullyConnected <LayerType> ::connectNeurons( const BaseLayer<LayerType>& previous) {

    for( auto currentNeuron : neurons ) {
        for( auto previousNeuron : previous.getNeurons() ) {
            NeuronOperations::connectNeurons( previousNeuron, currentNeuron );
        }

        if( bias != nullptr ) {
            NeuronOperations::connectNeurons( bias, currentNeuron );
        }
    }
}
