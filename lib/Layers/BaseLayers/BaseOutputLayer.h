
#ifndef NEURALNETWORK_BASEOUTPUTLAYER_H
#define NEURALNETWORK_BASEOUTPUTLAYER_H


#include "BaseLayer.h"
#include "../../LossFunctions/BaseLoss/BaseLossFunction.h"
#include "../../Activations/BaseActivation/BaseActivationFunction.h"
#include "../../Activations/SimpleActivations/Identitiy.h"
#include "../../Neurons/SimpleNeurons/Bias.h"

/**
 * Base class for all output layers
 * Its responsibility is to create Output neurons and connect them to previous layers
 * Currently due to the implementation this layer is a fully connected layer, but hopefully it will be changed later
 */
template <class LayerType>
class BaseOutputLayer : public BaseLayer <LayerType> {

protected:
    using BaseLayer <LayerType> :: numberOfNeurons;
    using BaseLayer <LayerType> :: previousLayers;
    using BaseLayer <LayerType> :: neurons;
    BaseActivationFunction <LayerType>* activationFunction;
    BaseLossFunction <LayerType>* lossFunction;
    Bias <LayerType>* bias;

public:
    /**
     * @param numberOfNeurons: number of neurons the layer has to create
     * @param previousLayers: all the layers that are connected to the output layer
     * @param lossFunction: loss function of all neurons in the output layer            | to have multiple loss function for different neurons we need to create multiple output layers
     * @param activationFunction: actuvatuib function of all neurons in this layer      | to have different activations for different neurons we need to create multiple output layers
     * @param bias: pointer to the bias neuron, as it's more efficient to have only one instance of bias for the whole network it's recommended to pass this one instance to all layers
     */
    BaseOutputLayer(unsigned int numberOfNeurons,
                    const std::vector< const BaseLayer<LayerType> *> &previousLayers,
                    BaseLossFunction <LayerType>* lossFunction,
                    BaseActivationFunction <LayerType> * activationFunction = new Identity <LayerType>(),
                    Bias <LayerType>* bias = nullptr);

    virtual void createNeurons() override;

    virtual void connectNeurons() override;
};

#include "BaseOutputLayer.tpp"

#endif //NEURALNETWORK_BASEOUTPUTLAYER_H
