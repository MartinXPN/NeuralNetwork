
#ifndef NEURALNETWORK_TANH_H
#define NEURALNETWORK_TANH_H


#include <cmath>
#include "../BaseActivation/BaseActivationFunction.h"


template <class Type>
class Tanh : public BaseActivationFunction <Type> {

public:

    virtual Type activation(Type x) override {
        return  ( 1. - exp( -2. * x ) ) /
                ( 1. + exp( -2. * x ) );
    }

    virtual Type activationDerivative(Type x) override {
        return 1. -
                pow( ( 1. - exp( -2. * x ) ) /
                     ( 1. + exp( -2. * x ) ), 2. );
    }
};


#endif //NEURALNETWORK_TANH_H
