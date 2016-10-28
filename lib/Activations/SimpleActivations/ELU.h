
#ifndef NEURALNETWORK_ELU_H
#define NEURALNETWORK_ELU_H


#include <cmath>
#include "../BaseActivation/BaseActivationFunction.h"

/**
 * Exponential Linear Unit
 * @see: https://arxiv.org/abs/1511.07289v3
 */
template <class Type>
class ELU : public BaseActivationFunction <Type> {

private:
    Type alpha;

public:

    /**
     * The activation function is:
     * x >= 0 => x
     * x <  0 => alpha * ( e^x - 1 )
     * @param alpha default value is 1
     */
    ELU( Type alpha = 1 ) {
        this -> alpha = alpha;
    }


    /**
     * x >= 0 => x
     * x <  0 => alpha * ( e^x - 1 )
     * @param x the input of the neuron
     */
    virtual Type activation(Type x) override {
        if( x < 0 ) return alpha * ( exp( x ) - 1 );
        else        return x;
    }


    /**
     * @returns
     *      x >= 0 => 1
     *      x <  0 => alpha * ELU( X )
     */
    virtual Type activationDerivative(Type x) override {
        if( x < 0 ) return alpha + activation( x );
        else        return 1;
    }
};

#endif //NEURALNETWORK_ELU_H
