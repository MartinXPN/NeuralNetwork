
#ifndef NEURALNETWORK_RELU_H
#define NEURALNETWORK_RELU_H


#include <algorithm>
#include "../BaseActivation/BaseActivationFunction.h"


template <class Type>
class ReLU : public BaseActivationFunction <Type> {

public:

    virtual Type activation(Type x) override {
        return std :: max( 0, x );
    }

    virtual Type activationDerivative(Type x) override {
        return ( x < 0 ? 0 : 1 );
    }
};

#endif //NEURALNETWORK_RELU_H
