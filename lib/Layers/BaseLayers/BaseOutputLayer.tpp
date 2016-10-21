
#include <cstdlib>
#include "BaseOutputLayer.h"
#include "../../Neurons/BaseNeurons/BaseOutputNeuron.h"


template <class LayerType>
BaseOutputLayer <LayerType> :: BaseOutputLayer(unsigned int numberOfNeurons,
                                               const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                                               BaseLossFunction<LayerType> *lossFunction,
                                               BaseActivationFunction<LayerType> *activationFunction, bool hasBias)
        : activationFunction( activationFunction ),
          lossFunction( lossFunction ),
          hasBias( hasBias ),
          BaseLayer <LayerType> (numberOfNeurons, previousLayers) {

}


template <class LayerType>
void BaseOutputLayer <LayerType> :: createNeurons(unsigned numberOfNeurons) {

    for( int i=0; i < numberOfNeurons; ++i )
        neurons.push_back( new BaseOutputNeuron <LayerType>(lossFunction, activationFunction ) );
}


template <class LayerType>
void BaseOutputLayer <LayerType> :: connectNeurons( const BaseLayer<LayerType>& previous) {

    for( auto currentNeuron : neurons ) {
        for (auto previousNeuron : previous.getNeurons()) {
            LayerType *weight = new LayerType(rand() / LayerType(RAND_MAX) - 0.5);
            BaseEdge<LayerType>* edge = new BaseEdge<LayerType>(previousNeuron, currentNeuron, weight);

            previousNeuron -> addNextLayerConnection( edge );
            currentNeuron -> addPreviousLayerConnection( edge );
        }
    }
}

