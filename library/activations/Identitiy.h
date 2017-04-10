
#ifndef NEURALNETWORK_IDENTITIY_H
#define NEURALNETWORK_IDENTITIY_H


#include "base/BaseActivationFunction.h"

/**
 * Identity activation function
 * activation -> x
 * derivative -> 1
 */
template <class Type>
class Identity : public BaseActivationFunction <Type> {

public:

    /**
     * @param x input of the neuron
     * @returns x
     */
    virtual Type activation(Type x) override {
        return x;
    }

    /**
     * @returns 1
     */
    virtual Type activationDerivative(Type x) override {
        return 1;
    }
};


#endif //NEURALNETWORK_IDENTITIY_H
