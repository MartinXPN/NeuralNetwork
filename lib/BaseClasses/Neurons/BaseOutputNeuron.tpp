
#include "BaseOutputNeuron.h"
#include "../../Activations/SimpleActivations/Identitiy.h"


template <class NeuronType>
BaseOutputNeuron <NeuronType> :: BaseOutputNeuron() :
        BaseOutputNeuron( new Identity <NeuronType>(), {} ) {

}

template <class NeuronType>
BaseOutputNeuron <NeuronType> :: BaseOutputNeuron(BaseActivationFunction<NeuronType> *activationFunction) :
        BaseOutputNeuron( activationFunction, {} ) {

}

template <class NeuronType>
BaseOutputNeuron <NeuronType> :: BaseOutputNeuron(BaseActivationFunction<NeuronType> * activationFunction,
                                                  std::vector<BaseEdge<NeuronType> *> previous) :
        BaseNeuron <NeuronType> ( activationFunction, {}, previous ) {

}



template <class NeuronType>
NeuronType BaseOutputNeuron <NeuronType> :: calculateLoss( NeuronType targetValue ) {
    return loss = ( activatedValue - targetValue ) *
                  activationFunction -> activationDerivative( preActivatedValue );
}
