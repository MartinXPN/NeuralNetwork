
#ifndef NEURALNETWORK_SIGMOIDNEURON_H
#define NEURALNETWORK_SIGMOIDNEURON_H


#include <cmath>
#include <vector>
#include "BaseNeuron.h"

template <class NeuronType>
class SigmoidNeuron : public BaseNeuron <NeuronType> {

public:

    SigmoidNeuron() :
            BaseNeuron <NeuronType> () {}
    SigmoidNeuron(const std::vector< BaseEdge< NeuronType > *> &after, const std::vector< BaseEdge<NeuronType> *> &before) :
            BaseNeuron <NeuronType> ( after, before ) {}

    virtual ~SigmoidNeuron() {}


    virtual NeuronType activation(NeuronType x) override;
    virtual NeuronType activationDerivative(NeuronType x) override;
};


template <class NeuronType>
NeuronType SigmoidNeuron <NeuronType> :: activation(NeuronType x) {
    return  1. / ( 1. + std::exp( -x ) );
}

template <class NeuronType>
NeuronType SigmoidNeuron <NeuronType> :: activationDerivative(NeuronType x) {
    return std::exp( x ) / ( ( 1. + std::exp( x ) ) * ( 1. + std::exp( x ) ) );
}

#endif //NEURALNETWORK_SIGMOIDNEURON_H
