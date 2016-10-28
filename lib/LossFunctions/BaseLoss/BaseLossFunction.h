
#ifndef NEURALNETWORK_BASELOSSFUNCTION_H
#define NEURALNETWORK_BASELOSSFUNCTION_H


/**
 * Interface for Loss functions
 * @see https://en.wikipedia.org/wiki/Loss_function
 */
template <class Type>
class BaseLossFunction {

public:
    /**
     * @param output the output of a neuron
     * @param target the target value
     * @return loss
     */
    virtual Type loss( Type output, Type target ) = 0;

    /**
     * @param output the output of a neuron
     * @param target the target value
     * @return loss derivative
     */
    virtual Type lossDerivative( Type output, Type target ) = 0;
};


#endif //NEURALNETWORK_BASELOSSFUNCTION_H
