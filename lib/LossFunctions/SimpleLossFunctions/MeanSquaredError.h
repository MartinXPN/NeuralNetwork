
#ifndef NEURALNETWORK_MEANSQUAREDERROR_H
#define NEURALNETWORK_MEANSQUAREDERROR_H


#include "../BaseLoss/BaseLossFunction.h"

template <class Type>
class MeanSquaredError : public BaseLossFunction <Type> {

public:
    /**
     * @param output the output of a neuron
     * @param target the target value
     * @return (target - output)^2
     */
    virtual Type loss(Type output, Type target) override {
        return ( target - output ) * ( target - output );
    }


    /**
     * @param output the output of a neuron
     * @param target the target value
     * @return output - target
     */
    virtual Type lossDerivative(Type output, Type target) override {
        return output - target;
    }
};


#endif //NEURALNETWORK_MEANSQUAREDERROR_H
