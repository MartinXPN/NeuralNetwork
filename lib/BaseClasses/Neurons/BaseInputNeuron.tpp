
#include "BaseInputNeuron.h"
#include "../../Activations/SimpleActivations/Identitiy.h"


template <class NeuronType>
BaseInputNeuron <NeuronType> :: BaseInputNeuron() :
        BaseInputNeuron( 0., {} ) {

}

template <class NeuronType>
BaseInputNeuron <NeuronType> :: BaseInputNeuron(NeuronType inputValue,
                                                const std::vector<BaseEdge<NeuronType> *> &next) :
        BaseNeuron <NeuronType> ( new Identity <NeuronType>, next ) {

    activatedValue = inputValue;
}


