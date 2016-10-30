
#include "BaseInputNeuron.h"


template <class NeuronType>
BaseInputNeuron <NeuronType> :: BaseInputNeuron() :
        BaseInputNeuron( 0., {} ) {

}

template <class NeuronType>
BaseInputNeuron <NeuronType> :: BaseInputNeuron(NeuronType inputValue,
                                                std::vector<BaseEdge<NeuronType> *> next) :
        BaseNeuron <NeuronType> (nullptr, next ) {

    activatedValue = inputValue;
}


