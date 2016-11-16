
#include "BaseHiddenLayer.h"

template <class LayerType>
BaseHiddenLayer <LayerType> :: BaseHiddenLayer(const std :: vector <unsigned>& dimensions,
                                               const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                                               BaseActivationFunction<LayerType> *activationFunction,
                                               Bias <LayerType>* bias)
        : BaseLayer <LayerType> (dimensions, previousLayers),
          activationFunction( activationFunction),
          bias( bias ) {

}