
#ifndef NEURALNETWORK_SIGMOIDNEURON_H
#define NEURALNETWORK_SIGMOIDNEURON_H


#include <cmath>
#include <vector>
#include "../BaseClasses/BaseNeuron.h"

template <class NeuronType>
class SigmoidNeuron : public BaseNeuron <NeuronType> {

public:

    SigmoidNeuron() :
            BaseNeuron <NeuronType> () {}
    SigmoidNeuron(const std::vector< BaseEdge< NeuronType > *> &after, const std::vector< BaseEdge<NeuronType> *> &before) :
            BaseNeuron <NeuronType> ( after, before ) {}

    virtual ~SigmoidNeuron() {}


    inline void setValue( NeuronType value ) { BaseNeuron <NeuronType> :: activatedValue = value; }

    virtual NeuronType activation(NeuronType x) override;
    virtual NeuronType activationDerivative(NeuronType x) override;
};

#include "SigmoidNeuron.tpp"

#endif //NEURALNETWORK_SIGMOIDNEURON_H
