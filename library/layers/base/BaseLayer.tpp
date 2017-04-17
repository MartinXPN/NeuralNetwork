
#include "BaseLayer.h"

template <class LayerType>
BaseLayer <LayerType> :: BaseLayer(const std :: vector <unsigned>& dimensions,
                                   const std :: vector< const BaseLayer* >& previousLayers,
                                   const std :: vector< BaseNeuron <LayerType>* >& neurons)
        : dimensions( dimensions ),
          previousLayers( previousLayers ),
          neurons(neurons) {
}
