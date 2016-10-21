
#ifndef NEURALNETWORK_BASEBIAS_H
#define NEURALNETWORK_BASEBIAS_H


#include "BaseInputNeuron.h"

template <class NeuronType>
class BaseBias : public BaseInputNeuron <NeuronType> {

public:
    BaseBias() : BaseInputNeuron <NeuronType> ( 1 ){};
    BaseBias( std::vector < BaseEdge <NeuronType>* > next = {} ) : BaseInputNeuron <NeuronType> ( 1, next ){};
};


#endif //NEURALNETWORK_BASEBIAS_H
