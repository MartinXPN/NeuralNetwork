
#ifndef NEURALNETWORK_ONETOONE_H
#define NEURALNETWORK_ONETOONE_H

#include <cstdlib>
#include "BaseLayers/BaseHiddenLayer.h"
#include "../Utilities/NeuronOperations.h"


template <class LayerType>
class OneToOne : public BaseHiddenLayer <LayerType> {
protected:
    using BaseHiddenLayer <LayerType> :: numberOfNeurons;
    using BaseHiddenLayer <LayerType> :: neurons;
    using BaseHiddenLayer <LayerType> :: weights;
    using BaseHiddenLayer <LayerType> :: deltaWeights;
    using BaseHiddenLayer <LayerType> :: previousLayers;


public:
    OneToOne(const std::vector<unsigned int> &dimensions,
             const std::vector<const BaseLayer<LayerType> *> &previousLayers,
             BaseActivationFunction<LayerType> *activationFunction);

    void createWeights() override;

    void connectNeurons() override;
};


#include "OneToOne.tpp"

#endif //NEURALNETWORK_ONETOONE_H
