
#include "BaseHiddenLayer.h"

template <class LayerType>
BaseHiddenLayer <LayerType> :: BaseHiddenLayer(unsigned int numberOfNeurons,
                                               const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                                               BaseActivationFunction<LayerType> *activationFunction,
                                               Bias <LayerType>* bias)
        : BaseLayer <LayerType> (numberOfNeurons, previousLayers),
          activationFunction( activationFunction),
          bias( bias ) {

}