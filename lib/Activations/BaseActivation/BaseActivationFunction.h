

#ifndef NEURALNETWORK_BASEACTIVATIONFUNCTION_H
#define NEURALNETWORK_BASEACTIVATIONFUNCTION_H


template <class Type>
class BaseActivationFunction {

public:
    virtual Type activation( Type x ) = 0;
    virtual Type activationDerivative( Type x ) = 0;
};


#endif //NEURALNETWORK_BASEACTIVATIONFUNCTION_H
