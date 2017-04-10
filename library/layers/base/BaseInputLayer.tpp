
#include "BaseInputLayer.h"


template <class LayerType>
void BaseInputLayer <LayerType> ::createNeurons() {

    for( int i=0; i < numberOfNeurons; ++i )
        neurons.push_back( new BaseInputNeuron <LayerType>() );
}