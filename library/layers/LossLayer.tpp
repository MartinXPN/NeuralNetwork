
#include <cstdlib>
#include "LossLayer.h"
#include "../edges/ConstantEdge.h"


template <class LayerType>
LossLayer <LayerType> :: LossLayer(const std::vector<unsigned> &dimensions,
                                   const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                   OutputNeuronInitializer<LayerType> *neuronInitializer)
        : OneToOne <LayerType> (dimensions, previousLayers, neuronInitializer) {

}


template <class LayerType>
LossLayer <LayerType> :: LossLayer(const std::vector<unsigned> &dimensions,
                                   const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                   BaseLossFunction<LayerType> *lossFunction)
        : LossLayer(dimensions, previousLayers, new OutputNeuronInitializer<LayerType>(lossFunction) ) {
}


template <class LayerType>
void LossLayer <LayerType> :: createWeights() {
    for( int i=0; i < this -> size(); ++i ) {
        weights.push_back(new LayerType(1));
        deltaWeights.push_back(new LayerType(0));
    }
}


template <class LayerType>
void LossLayer <LayerType> :: connectNeurons(BaseNeuron<LayerType> *source,
                                             BaseNeuron<LayerType> *target,
                                             LayerType *weight,
                                             LayerType *deltaWeight) {

    BaseEdge<LayerType> *edge = new ConstantEdge<LayerType>(source, target, weight );
    source->addNextLayerConnection(edge);
    target->addPreviousLayerConnection(edge);
}
