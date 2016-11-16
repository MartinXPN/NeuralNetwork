
#include "BaseLayer.h"

template <class LayerType>
BaseLayer <LayerType> :: BaseLayer(const std :: vector <unsigned>& dimensions,
                                   const std :: vector< const BaseLayer* >& previousLayers)
        : dimensions( dimensions ),
          previousLayers( previousLayers ) {

    numberOfNeurons = 1;
    for( auto dimension : dimensions )
        numberOfNeurons *= dimension;
}
