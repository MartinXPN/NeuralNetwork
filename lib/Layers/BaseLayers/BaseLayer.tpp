
#include "BaseLayer.h"

template <class LayerType>
BaseLayer <LayerType> :: BaseLayer(unsigned numberOfNeurons,
                                   const BaseLayer* previous) {

    createNeurons( numberOfNeurons );
    connectNeurons( previous );
}
