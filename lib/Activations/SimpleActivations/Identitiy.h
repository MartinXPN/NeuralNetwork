
#ifndef NEURALNETWORK_IDENTITIY_H
#define NEURALNETWORK_IDENTITIY_H


#include "../BaseActivation/BaseActivationFunction.h"

/**
 * Identity activation function
 * activation -> x
 * derivative -> 1
 */
template <class Type>
class Identity : public BaseActivationFunction <Type> {

public:

    virtual Type activation(Type x) override {
        return x;
    }

    virtual Type activationDerivative(Type x) override {
        return 1;
    }
};


#endif //NEURALNETWORK_IDENTITIY_H
