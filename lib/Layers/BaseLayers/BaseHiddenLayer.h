
#ifndef NEURALNETWORK_BASEHIDDENLAYER_H
#define NEURALNETWORK_BASEHIDDENLAYER_H


#include "BaseLayer.h"

template <class LayerType>
class BaseHiddenLayer : public BaseLayer <LayerType> {

private:

protected:
    using BaseLayer <LayerType> :: neurons;
    BaseActivationFunction <LayerType>* activationFunction;
    bool hasBias;

public:
    BaseHiddenLayer(unsigned int numberOfNeurons,
                    const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                    BaseActivationFunction<LayerType> *activationFunction,
                    bool hasBias = true );

    virtual void createNeurons(unsigned numberOfNeurons) override;

    virtual void createNeurons(unsigned numberOfNeurons, BaseActivationFunction <LayerType>* activationFunction) = 0;
};

#include "BaseHiddenLayer.tpp"

#endif //NEURALNETWORK_BASEHIDDENLAYER_H
