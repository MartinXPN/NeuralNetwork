
#ifndef NEURALNETWORK_BASEOUTPUTLAYER_H
#define NEURALNETWORK_BASEOUTPUTLAYER_H


#include "BaseLayer.h"
#include "../../LossFunctions/BaseLoss/BaseLossFunction.h"
#include "../../Activations/BaseActivation/BaseActivationFunction.h"
#include "../../Activations/SimpleActivations/Identitiy.h"


template <class LayerType>
class BaseOutputLayer : public BaseLayer <LayerType> {

protected:
    using BaseLayer <LayerType> :: neurons;
    BaseActivationFunction <LayerType>* activationFunction;
    BaseLossFunction <LayerType>* lossFunction;
    bool hasBias;

public:
    BaseOutputLayer(unsigned int numberOfNeurons,
                    const std::vector< const BaseLayer<LayerType> *> &previousLayers,
                    BaseLossFunction <LayerType>* lossFunction,
                    BaseActivationFunction <LayerType> * activationFunction = new Identity <LayerType>(),
                    bool hasBias = true);

    virtual void createNeurons(unsigned numberOfNeurons) override;

    virtual void connectNeurons(const BaseLayer<LayerType> &previous) override;
};

#include "BaseOutputLayer.tpp"

#endif //NEURALNETWORK_BASEOUTPUTLAYER_H
