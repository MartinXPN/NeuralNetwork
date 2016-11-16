
#ifndef NEURALNETWORK_BASEHIDDENLAYER_H
#define NEURALNETWORK_BASEHIDDENLAYER_H


#include "BaseLayer.h"
#include "../../Neurons/SimpleNeurons/Bias.h"

/**
 * Base ABSTRACT class for all hidden layers
 * Any class inherited from BaseHiddenLayer has to implement two functions:
 *      1. createNeurons
 *      2. connectNeurons
 */
template <class LayerType>
class BaseHiddenLayer : public BaseLayer <LayerType> {

private:

protected:
    using BaseLayer <LayerType> :: neurons;
    using BaseLayer <LayerType> :: numberOfNeurons;
    using BaseLayer <LayerType> :: previousLayers;
    BaseActivationFunction <LayerType>* activationFunction;
    Bias <LayerType>* bias;

public:
    BaseHiddenLayer(const std :: vector <unsigned>& dimensions,
                    const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                    BaseActivationFunction<LayerType> *activationFunction,
                    Bias <LayerType>* bias = nullptr );
};

#include "BaseHiddenLayer.tpp"

#endif //NEURALNETWORK_BASEHIDDENLAYER_H
