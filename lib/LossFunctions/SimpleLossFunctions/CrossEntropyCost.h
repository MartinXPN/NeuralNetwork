
#ifndef NEURALNETWORK_CROSSENTROPYCOST_H
#define NEURALNETWORK_CROSSENTROPYCOST_H

#include <cmath>
#include "../BaseLoss/BaseLossFunction.h"


/**
 * Cross Entropy Cost
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 */
template <class Type>
class CrossEntropyCost : public BaseLossFunction <Type> {

public:
    /**
     * @param output the output of a neuron
     * @param target the target value
     * @return - target * ln(output) + ( 1−target ) * ln(1−output)
     */
    virtual Type loss(Type output, Type target) override {
        return - target * log(output) + ( 1 - target ) * log( 1 - output );
    }


    /**
     * @param output the output of a neuron
     * @param target the target value
     * @return (output−target) / ( output * (1−output) )
     */
    virtual Type lossDerivative(Type output, Type target) override {
        return ( output - target ) / ( output * ( 1 - output ) );
    }
};


#endif //NEURALNETWORK_CROSSENTROPYCOST_H
