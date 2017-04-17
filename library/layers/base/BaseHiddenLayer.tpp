
#include "BaseHiddenLayer.h"
#include "../../util/MathOperations.h"
#include "../../initializers/neuron/SimpleNeuronInitializer.h"

template <class LayerType>
BaseHiddenLayer <LayerType> :: BaseHiddenLayer(const std::vector<unsigned> &dimensions,
                                               const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                               NeuronInitializer<LayerType> *neuronInitializer,
                                               Bias<LayerType> *bias)
        : BaseHiddenLayer(dimensions,
                          previousLayers,
                          neuronInitializer -> createNeurons( math::multiply(dimensions) ),
                          bias) {

}


template <class LayerType>
BaseHiddenLayer <LayerType> :: BaseHiddenLayer(const std::vector<unsigned> &dimensions,
                                               BaseActivationFunction<LayerType> *activationFunction,
                                               const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                               Bias<LayerType> *bias)
        : BaseHiddenLayer(dimensions,
                          previousLayers,
                          new SimpleNeuronInitializer<LayerType>(activationFunction),
                          bias) {

}



template <class LayerType>
BaseHiddenLayer <LayerType> :: BaseHiddenLayer(const std::vector<unsigned> &dimensions,
                                               const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                               const std::vector<BaseNeuron<LayerType> *> &neurons,
                                               Bias<LayerType> *bias)
        : BaseLayer <LayerType> (dimensions, previousLayers, neurons),
          bias(bias) {

}


template <class LayerType>
BaseHiddenLayer <LayerType> :: ~BaseHiddenLayer() {

    for( int i=0; i < weights.size(); ++i )         delete weights[i];
    for( int i=0; i < deltaWeights.size(); ++i )    delete deltaWeights[i];
}


template <class LayerType>
void BaseHiddenLayer <LayerType> :: connectNeurons(BaseNeuron<LayerType> *source,
                                                   BaseNeuron<LayerType> *target,
                                                   LayerType* weight,
                                                   LayerType* deltaWeight) {

    if( deltaWeight == nullptr )    deltaWeight = new LayerType( 0 );

    BaseEdge<LayerType> *edge = new BaseEdge<LayerType>(source, target, weight, deltaWeight );

    source->addNextLayerConnection(edge);
    target->addPreviousLayerConnection(edge);
}
