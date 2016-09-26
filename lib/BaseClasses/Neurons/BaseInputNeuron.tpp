
#include "BaseInputNeuron.h"

template <class NeuronType>
BaseInputNeuron <NeuronType> :: BaseInputNeuron() :
        BaseNeuron <NeuronType> () {}


template <class NeuronType>
BaseInputNeuron <NeuronType> ::BaseInputNeuron(const std::vector<BaseEdge<NeuronType> *> &next,
                                               const std::vector<BaseEdge<NeuronType> *> &previous) :
        BaseNeuron <NeuronType> ( next, previous ) {}



template <class NeuronType>
BaseInputNeuron <NeuronType> :: BaseInputNeuron(NeuronType inputValue) :
        BaseNeuron <NeuronType> () {

    activatedValue = inputValue;
}
