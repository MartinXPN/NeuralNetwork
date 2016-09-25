
#include "SigmoidNeuron.h"

template <class NeuronType>
NeuronType SigmoidNeuron <NeuronType> :: activation(NeuronType x) {
    return  1. /
            ( 1. + std::exp( -x ) );
}

template <class NeuronType>
NeuronType SigmoidNeuron <NeuronType> :: activationDerivative(NeuronType x) {
    return std::exp( x ) /
           std::pow( ( 1. + std::exp( x ) ), 2 );
}