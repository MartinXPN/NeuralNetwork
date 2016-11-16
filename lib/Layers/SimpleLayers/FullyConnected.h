
#ifndef NEURALNETWORK_FULLYCONNECTED_H
#define NEURALNETWORK_FULLYCONNECTED_H


#include <cstdlib>
#include "../BaseLayers/BaseHiddenLayer.h"


/**
 * Fully Connected layer
 * It connects all neurons in it to all neurons in all the previous layers
 */
template <class LayerType>
class FullyConnected : public BaseHiddenLayer <LayerType> {

protected:
    using BaseHiddenLayer <LayerType> :: numberOfNeurons;
    using BaseHiddenLayer <LayerType> :: previousLayers;
    using BaseHiddenLayer <LayerType> :: activationFunction;
    using BaseHiddenLayer <LayerType> :: neurons;
    using BaseHiddenLayer <LayerType> :: bias;

public:
    FullyConnected(const std :: vector <unsigned>& dimensions,
                   BaseActivationFunction<LayerType>* activationFunction,
                   const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                   Bias <LayerType>* bias = nullptr );

    virtual void connectNeurons() override;
};

#include "FullyConnected.tpp"

#endif //NEURALNETWORK_FULLYCONNECTED_H
