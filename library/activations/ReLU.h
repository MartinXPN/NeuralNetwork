
#ifndef NEURALNETWORK_RELU_H
#define NEURALNETWORK_RELU_H


#include "base/BaseActivationFunction.h"


template <class Type>
class ReLU : public BaseActivationFunction <Type> {

public:

    /**
     * @param x input of the neuron
     * @returns
     *      x < 0  => 0
     *      x >= 0 => x
     */
    virtual Type activation(Type x) override {
        return ( x < 0 ? 0 : x );
    }

    /**
     * @returns
     *      x < 0  => 0
     *      x >= 0 => x
     */
    virtual Type activationDerivative(Type x) override {
        return ( x < 0 ? 0 : 1 );
    }
};

#endif //NEURALNETWORK_RELU_H
