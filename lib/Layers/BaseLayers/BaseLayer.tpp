
#include "BaseLayer.h"

template <class LayerType>
BaseLayer <LayerType> :: BaseLayer(unsigned numberOfNeurons,
                                   const std :: vector< const BaseLayer* > previousLayers)
        : numberOfNeurons( numberOfNeurons ),
          previousLayers( previousLayers ) {

}
