
#include <cstdlib>
#include "LossLayer.h"


template <class LayerType>
LossLayer <LayerType> :: LossLayer(const std::vector<unsigned> &dimensions,
                                   const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                   OutputNeuronInitializer<LayerType> *neuronInitializer)
        : OneToOne <LayerType> (dimensions, previousLayers, neuronInitializer) {

}

template <class LayerType>
void LossLayer <LayerType> :: createWeights() {
    for( int i=0; i < this -> size(); ++i ) {
        weights.push_back(new LayerType(1));
        deltaWeights.push_back(new LayerType(0));
    }
}
