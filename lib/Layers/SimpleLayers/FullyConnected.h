
#ifndef NEURALNETWORK_FULLYCONNECTED_H
#define NEURALNETWORK_FULLYCONNECTED_H


#include <cstdlib>
#include "../BaseLayers/BaseHiddenLayer.h"


template <class LayerType>
class FullyConnected : public BaseHiddenLayer <LayerType> {

protected:
    using BaseLayer <LayerType> :: neurons;

public:
    FullyConnected(unsigned int numberOfNeurons,
                   const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                   BaseActivationFunction<LayerType>* activationFunction,
                   bool hasBias = true );

    virtual void createNeurons(unsigned numberOfNeurons, BaseActivationFunction<LayerType> *activationFunction) override;

    virtual void connectNeurons( const BaseLayer<LayerType>& previous) override;
};

#include "FullyConnected.tpp"

#endif //NEURALNETWORK_FULLYCONNECTED_H
