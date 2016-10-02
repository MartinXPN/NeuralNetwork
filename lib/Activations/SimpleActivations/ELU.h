
#ifndef NEURALNETWORK_ELU_H
#define NEURALNETWORK_ELU_H


#include <cmath>
#include "../BaseActivation/BaseActivationFunction.h"

template <class Type>
class ELU : public BaseActivationFunction <Type> {

private:
    Type alpha;

public:

    ELU( Type alpha = 1 ) {
        this -> alpha = alpha;
    }

    virtual Type activation(Type x) override {
        if( x < 0 ) return alpha * ( exp( x ) - 1 );
        else        return x;
    }

    virtual Type activationDerivative(Type x) override {
        if( x < 0 ) return alpha + activation( x );
        else        return 1;
    }
};

#endif //NEURALNETWORK_ELU_H
