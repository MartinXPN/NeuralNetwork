
#ifndef NEURALNETWORK_SIGMOID_H
#define NEURALNETWORK_SIGMOID_H


#include <cmath>
#include "../BaseActivation/BaseActivationFunction.h"


template <class Type>
class Sigmoid : public BaseActivationFunction <Type> {

public:

    virtual Type activation(Type x) override {
        return  1. /
                ( 1. + exp( -x ) );
    }

    virtual Type activationDerivative(Type x) override {
        return exp( x ) /
               pow( ( 1. + exp( x ) ), 2 );
    }
};


#endif //NEURALNETWORK_SIGMOID_H
