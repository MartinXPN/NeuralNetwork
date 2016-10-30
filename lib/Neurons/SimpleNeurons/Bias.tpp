
#include "Bias.h"

template <class NeuronType>
Bias <NeuronType> :: Bias( std::vector < BaseEdge <NeuronType>* > next )
        : BaseInputNeuron <NeuronType> ( 1, next ){

}

template <class NeuronType>
Bias :: ~Bias() {
}
