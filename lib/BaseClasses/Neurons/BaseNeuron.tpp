
#include "BaseNeuron.h"


template <class NeuronType>
BaseNeuron <NeuronType> :: BaseNeuron() :
        BaseNeuron( {}, {} ) {}


template <class NeuronType>
BaseNeuron <NeuronType> :: BaseNeuron( const std::vector < BaseEdge <NeuronType>* >& next,
                                       const std::vector < BaseEdge <NeuronType>* >& previous ) :
        next( next ), previous( previous ), activatedValue(1.), preActivatedValue(0.), loss(1.) {}


template <class NeuronType>
BaseNeuron <NeuronType> :: ~BaseNeuron() {

    /// can't delete edges with delete because they may not be created with new
    // for( auto edge : next )     free( edge );
    // for( auto edge : previous ) free( edge );
}


template <class NeuronType>
void BaseNeuron <NeuronType> :: activateNeuron() {

    /// preactivatedValue = sum of [values of neurons from previous layer * weights connected to them ]
    preActivatedValue = 0.;
    for( auto edge : previous ) {
        preActivatedValue += edge -> getWeight() *
                             edge -> getFrom().getActivatedValue();
    }
    activatedValue = activation( preActivatedValue );
}


template <class NeuronType>
void BaseNeuron <NeuronType> :: calculateLoss() {

    /// loss = sum( weights_to_next_layer * next.loss ) * activationDerivative( preactivatedValue )
    loss = 0.;
    for( auto edge : next ) {
        loss += edge -> getWeight() *
                edge -> getTo().getLoss();
    }
    loss *= activationDerivative( this -> preActivatedValue );
}



template <class NeuronType>
void BaseNeuron <NeuronType> :: backpropagateNeuron(NeuronType learningRate, int batchSize) {

    for( auto edge : next ) {

        /// update current weight
        /// weight -= (learningRate / batchSize)  * next.loss * activatedValue
        edge -> setDeltaWeight( edge -> getDeltaWeight() +
                                learningRate / batchSize *
                                edge -> getTo().getLoss() *
                                getActivatedValue() );
    }
}
