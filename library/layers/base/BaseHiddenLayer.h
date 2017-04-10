
#ifndef NEURALNETWORK_BASEHIDDENLAYER_H
#define NEURALNETWORK_BASEHIDDENLAYER_H


#include "BaseLayer.h"
#include "../../neurons/Bias.h"
#include "../../activations/base/BaseActivationFunction.h"

/**
 * Base ABSTRACT class for all hidden layers
 * Any class inherited from BaseHiddenLayer has to implement two functions:
 *      1. createNeurons
 *      2. createWeights
 *      3. connectNeurons
 */
template <class LayerType>
class BaseHiddenLayer : public BaseLayer <LayerType> {

private:

protected:
    using BaseLayer <LayerType> :: neurons;
    using BaseLayer <LayerType> :: numberOfNeurons;
    using BaseLayer <LayerType> :: previousLayers;
    std :: vector <LayerType*> weights;
    std :: vector <LayerType*> deltaWeights;
    BaseActivationFunction <LayerType>* activationFunction;
    Bias <LayerType>* bias;

public:
    // TODO add weight initializer as argument
    BaseHiddenLayer(const std :: vector <unsigned>& dimensions,
                    const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                    BaseActivationFunction<LayerType> *activationFunction,
                    Bias <LayerType>* bias = nullptr );

    virtual void createNeurons() override;
    virtual void createWeights() = 0;

    virtual ~BaseHiddenLayer();
};

#include "BaseHiddenLayer.tpp"

#endif //NEURALNETWORK_BASEHIDDENLAYER_H
