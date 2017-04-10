
#ifndef NEURALNETWORK_SIGMOID_H
#define NEURALNETWORK_SIGMOID_H


#include <cmath>
#include "base/BaseActivationFunction.h"

/**
 * Sigmoid activation function
 * @see: https://en.wikipedia.org/wiki/Sigmoid_function
 */
template <class Type>
class Sigmoid : public BaseActivationFunction <Type> {

public:

    /**
     * @param x input of the neuron
     * @returns 1 / ( 1 + e^-x )
     */
    virtual Type activation(Type x) override {
        if( x < -5 )    return 0.0000001; /// done for optimization purpouses as exp( -x ) will be a huge number when x < -5
        if( x > 5 )     return 0.9999999; /// done for optimization purpouses as exp( x ) will be negligible when x > 5
        return  1. /
                ( 1. + std::exp( -x ) );
    }

    /**
     * @returns e^x / ( 1 + e^x )^2
     */
    virtual Type activationDerivative(Type x) override {
        if( std::abs(x) > 5 )    return 0.0000001;   /// done for optimization purpouses as derivative tends to 0 when |x| > 5
        return std::exp( x ) /
               std::pow( ( 1. + std::exp( x ) ), 2 );
    }
};


#endif //NEURALNETWORK_SIGMOID_H
