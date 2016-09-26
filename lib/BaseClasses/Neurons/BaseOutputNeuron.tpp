
#include "BaseOutputNeuron.h"


template <class NeuronType>
BaseOutputNeuron <NeuronType> :: BaseOutputNeuron() :
        BaseNeuron <NeuronType> () {}


template <class NeuronType>
BaseOutputNeuron <NeuronType> :: BaseOutputNeuron(const std::vector<BaseEdge<NeuronType> *> &next,
                                                  const std::vector<BaseEdge<NeuronType> *> &previous) :
        BaseNeuron <NeuronType> (next, previous) {

}

template <class NeuronType>
NeuronType BaseOutputNeuron <NeuronType> :: calculateLoss( NeuronType realValue ) {
    return loss = ( realValue - getValue() ) * ( realValue - getValue() );
}
