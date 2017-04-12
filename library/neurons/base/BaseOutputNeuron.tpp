
#include <iostream>
#include "BaseOutputNeuron.h"


template <class NeuronType>
BaseOutputNeuron <NeuronType> ::BaseOutputNeuron(BaseLossFunction<NeuronType> *lossFunction,
                                                 BaseActivationFunction<NeuronType> *activationFunction,
                                                 std::vector<BaseEdge<NeuronType> *> previous) :
        BaseNeuron <NeuronType> ( activationFunction, {}, previous ), lossFunction(lossFunction) {
}


template <class NeuronType>
void BaseOutputNeuron <NeuronType> :: calculateLoss( NeuronType targetValue ) {
    loss = lossFunction -> lossDerivative( getValue(), targetValue ) *
           activationFunction -> activationDerivative( preActivatedValue );
}

template <class NeuronType>
NeuronType BaseOutputNeuron <NeuronType> :: getError(NeuronType targetValue) {
//    std::cout << "loss( " << getValue() << ", " << targetValue << ") = " << lossFunction -> loss(getValue(), targetValue) << std::endl;
    return lossFunction -> loss( getValue(), targetValue );
}
