
#ifndef NEURALNETWORK_BASEINPUTLAYER_H
#define NEURALNETWORK_BASEINPUTLAYER_H


#include "base/BaseLayer.h"
#include "../neurons/base/BaseInputNeuron.h"
#include "../initializers/neuron/NeuronInitializer.h"
#include "../initializers/neuron/InputNeuronInitializer.h"


/**
 * Base class of Input layers
 * Its only responsibility is to create neurons and populate the collection - vector <> neurons
 */
template <class LayerType>
class InputLayer : public BaseLayer <LayerType> {

private:
    /// Hide this method from subclasses
    virtual void connectNeurons() override {
        throw "Input layer does not have previous layer to connect neurons";
    }

protected:
    using BaseLayer <LayerType> :: neurons;

public:

    InputLayer(const std :: vector <unsigned>& dimensions,
               NeuronInitializer <LayerType>* neuronInitializer = new InputNeuronInitializer <LayerType>() );

    InputLayer(const std :: vector <unsigned>& dimensions,
               const std :: vector< BaseNeuron <LayerType>* >& neurons);
};

#include "InputLayer.tpp"

#endif //NEURALNETWORK_BASEINPUTLAYER_H
