
#include "BaseHiddenLayer.h"

template <class LayerType>
BaseHiddenLayer <LayerType> :: BaseHiddenLayer(unsigned int numberOfNeurons,
                                               const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                                               BaseActivationFunction<LayerType> *activationFunction,
                                               bool hasBias)
        : activationFunction( activationFunction),
          hasBias( hasBias ),
          BaseLayer <LayerType> (numberOfNeurons, previousLayers)  {

}


template <class LayerType>
void BaseHiddenLayer <LayerType> :: createNeurons(unsigned numberOfNeurons) {
    createNeurons( numberOfNeurons, activationFunction );
}
