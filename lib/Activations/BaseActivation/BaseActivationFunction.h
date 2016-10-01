

#ifndef NEURALNETWORK_BASEACTIVATIONFUNCTION_H
#define NEURALNETWORK_BASEACTIVATIONFUNCTION_H

/**
 * Interface for activation function
 * contains functions:
 *      1. activation
 *      2. activationDerivative
 *
 * @see: https://en.wikipedia.org/wiki/Activation_function
 */
template <class Type>
class BaseActivationFunction {

public:
    /**
     * @param x: input for the activation function
     * @return: activated value of x
     */
    virtual Type activation( Type x ) = 0;

    /**
     * @param x: input for the activation derivative
     * @return: derivative of the activation function at point x
     */
    virtual Type activationDerivative( Type x ) = 0;
};


#endif //NEURALNETWORK_BASEACTIVATIONFUNCTION_H
