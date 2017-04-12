
#include "BaseHiddenLayer.h"
#include "../../util/MathOperations.h"

template <class LayerType>
BaseHiddenLayer <LayerType> :: BaseHiddenLayer(const std::vector<unsigned> &dimensions,
                                               const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                               NeuronInitializer<LayerType> *neuronInitializer,
                                               Bias<LayerType> *bias) :
        BaseHiddenLayer(dimensions,
                        previousLayers,
                        neuronInitializer -> createNeurons( math::multiply(dimensions) ),
                        bias) {

}


template <class LayerType>
BaseHiddenLayer <LayerType> :: BaseHiddenLayer(const std::vector<unsigned> &dimensions,
                                               const std::vector<const BaseLayer<LayerType> *> &previousLayers,
                                               const std::vector<BaseNeuron<LayerType> *> &neurons,
                                               Bias<LayerType> *bias) :
        BaseLayer <LayerType> (dimensions, previousLayers, neurons),
        bias(bias) {

}


template <class LayerType>
BaseHiddenLayer <LayerType> :: ~BaseHiddenLayer() {

    for( int i=0; i < weights.size(); ++i )         delete weights[i];
    for( int i=0; i < deltaWeights.size(); ++i )    delete deltaWeights[i];
}
