
#ifndef NEURALNETWORK_FULLYCONNECTED_H
#define NEURALNETWORK_FULLYCONNECTED_H


#include <cstdlib>
#include "base/BaseHiddenLayer.h"
#include "base/BaseLayer.h"
#include "../activations/base/BaseActivationFunction.h"
#include "../neurons/Bias.h"


/**
 * Fully Connected layer
 * It connects all neurons in it to all neurons in all the previous layers
 */
template <class LayerType>
class FullyConnected : public BaseHiddenLayer <LayerType> {

protected:
    using BaseHiddenLayer <LayerType> :: previousLayers;
    using BaseHiddenLayer <LayerType> :: neurons;
    using BaseHiddenLayer <LayerType> :: bias;
    using BaseHiddenLayer <LayerType> :: weights;
    using BaseHiddenLayer <LayerType> :: deltaWeights;
    using BaseHiddenLayer <LayerType> :: connectNeurons;

public:
    using BaseHiddenLayer <LayerType> :: BaseHiddenLayer;

    using BaseHiddenLayer <LayerType> :: size;

    virtual void createWeights() override;

    virtual void connectNeurons() override;
};

#include "FullyConnected.tpp"

#endif //NEURALNETWORK_FULLYCONNECTED_H
