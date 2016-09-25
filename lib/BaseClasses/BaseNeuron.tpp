
#include "BaseNeuron.h"


template <class NeuronType>
BaseNeuron <NeuronType> :: BaseNeuron() : BaseNeuron( {}, {} ) {}


template <class NeuronType>
BaseNeuron <NeuronType> :: BaseNeuron( const std::vector < BaseEdge <NeuronType>* >& next,
                                       const std::vector < BaseEdge <NeuronType>* >& previous ):
        next( next ), previous( previous ), activatedValue(0.), preActivatedValue(0.), loss(0.) {}


template <class NeuronType>
BaseNeuron <NeuronType> :: ~BaseNeuron() {

    /// can't delete edges with delete because they may not be created with new
    // for( auto edge : next )     free( edge );
    // for( auto edge : previous ) free( edge );
}


template <class NeuronType>
void BaseNeuron <NeuronType> :: onActivation() {

    /// preactivatedValue = sum of [values of neurons from previous layer * weights connected to them ]
    preActivatedValue = 0.;
    for( auto edge : previous ) {
        preActivatedValue += ( *(edge -> getWeight()) ) *
                             ( edge -> getFrom() -> getActivatedValue() );
    }
    activatedValue = activation( preActivatedValue );
}


template <class NeuronType>
void BaseNeuron <NeuronType> :: onBackpropagation( NeuronType learningRate, int batchSize ) {

    for( auto edge : next ) {

        NeuronType currentDeltaWeight = edge -> getDeltaWeight();

        currentDeltaWeight -= ( learningRate / batchSize ) *( edge -> getTo() -> getLoss() ) * getActivatedValue();
        edge -> setDeltaWeight( currentDeltaWeight );
    }
}
