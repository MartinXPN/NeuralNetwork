
#include "BaseOutputNeuron.h"
#include "../../Activations/SimpleActivations/Identitiy.h"


template <class NeuronType>
BaseOutputNeuron <NeuronType> :: BaseOutputNeuron(BaseLossFunction<NeuronType> *lossFunction) :
        BaseOutputNeuron( lossFunction, new Identity <NeuronType>, {} ) {

    this -> lossFunction = lossFunction;
}

template <class NeuronType>
BaseOutputNeuron <NeuronType> ::BaseOutputNeuron(BaseLossFunction<NeuronType> *lossFunction,
                                                 BaseActivationFunction<NeuronType> *activationFunction,
                                                 std::vector<BaseEdge<NeuronType> *> previous) :
        BaseNeuron <NeuronType> ( activationFunction, {}, previous ) {

    this -> lossFunction = lossFunction;
}


template <class NeuronType>
void BaseOutputNeuron <NeuronType> :: calculateLoss( NeuronType targetValue ) {
    loss = lossFunction -> lossDerivative( getValue(), targetValue ) *
           activationFunction -> activationDerivative( preActivatedValue );
}

template <class NeuronType>
NeuronType BaseOutputNeuron <NeuronType> :: getError(NeuronType targetValue) {
    return lossFunction -> loss( getValue(), targetValue );
}
