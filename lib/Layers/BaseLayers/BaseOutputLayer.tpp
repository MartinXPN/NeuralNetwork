
#include <cstdlib>
#include "BaseOutputLayer.h"
#include "../../Neurons/BaseNeurons/BaseOutputNeuron.h"
#include "../../Utilities/NeuronOperations.h"


template <class LayerType>
BaseOutputLayer <LayerType> :: BaseOutputLayer(unsigned int numberOfNeurons,
                                               const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                                               BaseLossFunction<LayerType> *lossFunction,
                                               BaseActivationFunction<LayerType> *activationFunction,
                                               BaseBias <LayerType>* bias)
        : activationFunction( activationFunction ),
          lossFunction( lossFunction ),
          bias( bias ),
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
            NeuronOperations::connectNeurons( previousNeuron, currentNeuron );
        }

        if( bias != nullptr ) {
            NeuronOperations::connectNeurons( bias, currentNeuron );
        }
    }
}

