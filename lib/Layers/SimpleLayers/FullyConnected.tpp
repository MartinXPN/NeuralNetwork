
#include "FullyConnected.h"


template <class LayerType>
FullyConnected <LayerType> :: FullyConnected(unsigned int numberOfNeurons,
                                             const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                                             BaseActivationFunction<LayerType> *activationFunction, bool hasBias) :
        BaseHiddenLayer <LayerType> ( numberOfNeurons,
                                      previousLayers,
                                      activationFunction,
                                      hasBias ) {

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
            LayerType *weight = new LayerType(rand() / LayerType(RAND_MAX) - 0.5);
            BaseEdge<LayerType>* edge = new BaseEdge<LayerType>(previousNeuron, currentNeuron, weight);

            previousNeuron -> addNextLayerConnection( edge );
            currentNeuron -> addPreviousLayerConnection( edge );
        }
    }
}
