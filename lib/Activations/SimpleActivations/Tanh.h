
#ifndef NEURALNETWORK_TANH_H
#define NEURALNETWORK_TANH_H


#include <cmath>
#include "../BaseActivation/BaseActivationFunction.h"


/**
 * Tanh activation function
 * @see: https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent
 */
template <class Type>
class Tanh : public BaseActivationFunction <Type> {

public:

    /**
     * @param x input of the neuron
     * @returns ( 1 - e^(-2x) ) / ( 1 + e^(-2x) )
     */
    virtual Type activation(Type x) override {
        return  ( 1. - exp( -2. * x ) ) /
                ( 1. + exp( -2. * x ) );
    }

    /**
     * @returns 1 - ( ( 1 - e^(-2x) ) / ( 1 + e^(2x) ) )^2
     */
    virtual Type activationDerivative(Type x) override {
        return 1. -
                pow( ( 1. - exp( -2. * x ) ) /
                     ( 1. + exp( -2. * x ) ), 2. );
    }
};


#endif //NEURALNETWORK_TANH_H
