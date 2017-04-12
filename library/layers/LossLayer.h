
#ifndef NEURALNETWORK_BASEOUTPUTLAYER_H
#define NEURALNETWORK_BASEOUTPUTLAYER_H


#include "base/BaseLayer.h"
#include "../neurons/Bias.h"
#include "../neurons/base/BaseOutputNeuron.h"
#include "../lossfunctions/base/BaseLossFunction.h"
#include "../activations/Identitiy.h"
#include "../util/NeuronOperations.h"
#include "OneToOne.h"
#include "../initializers/neuron/OutputNeuronInitializer.h"
#include "../lossfunctions/CrossEntropyCost.h"


/**
 * Base class for all output layers
 * Its responsibility is to create Output neurons and connect them to previous layers
 * Currently due to the implementation this layer is a fully connected layer, but hopefully it will be changed later
 */
template <class LayerType>
class LossLayer : public OneToOne<LayerType> {

protected:
    using BaseLayer <LayerType> :: previousLayers;
    using BaseLayer <LayerType> :: neurons;
    using OneToOne <LayerType> :: weights;
    using OneToOne <LayerType> :: deltaWeights;

public:
    /**
     * @param dimensions: dimensions of the layer
     * @param previousLayers: all the layers that are connected to the output layer
     * @param lossFunction: loss function of all neurons in the output layer            | to have multiple loss function for different neurons we need to create multiple output layers
     */
    LossLayer(const std :: vector <unsigned>& dimensions,
              const std::vector< const BaseLayer<LayerType> *> &previousLayers,
              OutputNeuronInitializer <LayerType>* neuronInitializer = new OutputNeuronInitializer <LayerType> (new CrossEntropyCost <LayerType>()));


    void createWeights() override;
};


#include "LossLayer.tpp"

#endif //NEURALNETWORK_BASEOUTPUTLAYER_H
