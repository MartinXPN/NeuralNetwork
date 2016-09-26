
#ifndef NEURALNETWORK_BASEINPUTNEURON_H
#define NEURALNETWORK_BASEINPUTNEURON_H


#include "BaseNeuron.h"

template <class NeuronType>
class BaseInputNeuron : private BaseNeuron <NeuronType> {


protected:
    using BaseNeuron <NeuronType> :: activatedValue;
    using BaseNeuron <NeuronType> :: next;

    virtual NeuronType activation(NeuronType x) override            { return x; }
    virtual NeuronType activationDerivative(NeuronType x) override  { return x; }

public:

    BaseInputNeuron();
    BaseInputNeuron( NeuronType inputValue );
    BaseInputNeuron(  const std::vector < BaseEdge <NeuronType>* >& next, const std::vector < BaseEdge <NeuronType>* >& previous  );
    virtual ~BaseInputNeuron() {}

    using BaseNeuron <NeuronType> :: calculateLoss;
    using BaseNeuron <NeuronType> :: backpropagateNeuron;
    using BaseNeuron <NeuronType> :: addNextLayerConnection;

    inline NeuronType getValue() const          { return activatedValue; }
    inline void setValue( NeuronType value )    { activatedValue = value; }
};

#include "BaseInputNeuron.tpp"


#endif //NEURALNETWORK_BASEINPUTNEURON_H
