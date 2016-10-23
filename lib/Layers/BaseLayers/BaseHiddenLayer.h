
#ifndef NEURALNETWORK_BASEHIDDENLAYER_H
#define NEURALNETWORK_BASEHIDDENLAYER_H


#include "BaseLayer.h"
#include "../../Neurons/BaseNeurons/BaseBias.h"

template <class LayerType>
class BaseHiddenLayer : public BaseLayer <LayerType> {

private:

protected:
    using BaseLayer <LayerType> :: neurons;
    BaseActivationFunction <LayerType>* activationFunction;
    BaseBias <LayerType>* bias;

public:
    BaseHiddenLayer(unsigned int numberOfNeurons,
                    const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                    BaseActivationFunction<LayerType> *activationFunction,
                    BaseBias <LayerType>* bias = nullptr );

    virtual void createNeurons(unsigned numberOfNeurons) override;

    virtual void createNeurons(unsigned numberOfNeurons, BaseActivationFunction <LayerType>* activationFunction) = 0;
};

#include "BaseHiddenLayer.tpp"

#endif //NEURALNETWORK_BASEHIDDENLAYER_H
