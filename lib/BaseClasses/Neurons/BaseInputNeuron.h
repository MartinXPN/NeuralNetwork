
#ifndef NEURALNETWORK_BASEINPUTNEURON_H
#define NEURALNETWORK_BASEINPUTNEURON_H


#include "BaseNeuron.h"

template <class NeuronType>
class BaseInputNeuron : private BaseNeuron <NeuronType> {


protected:
    using BaseNeuron <NeuronType> :: activatedValue;
    using BaseNeuron <NeuronType> :: next;

public:

    BaseInputNeuron();
    BaseInputNeuron( NeuronType inputValue, const std::vector < BaseEdge <NeuronType>* >& next = {} );
    virtual ~BaseInputNeuron() {}

    using BaseNeuron <NeuronType> :: backpropagateNeuron;
    using BaseNeuron <NeuronType> :: addNextLayerConnection;

    inline NeuronType getValue() const          { return activatedValue; }
    inline void setValue( NeuronType value )    { activatedValue = value; }
};


#include "BaseInputNeuron.tpp"


#endif //NEURALNETWORK_BASEINPUTNEURON_H
