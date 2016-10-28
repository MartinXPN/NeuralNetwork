
#include "BaseNeuron.h"



template <class NeuronType>
BaseNeuron <NeuronType> :: BaseNeuron( BaseActivationFunction<NeuronType>* activationFunction,
                                       std::vector<BaseEdge<NeuronType> *> next,
                                       std::vector<BaseEdge<NeuronType> *> previous) :
        activationFunction( activationFunction ),
        next( next ),
        previous( previous ),
        activatedValue( 0. ),
        preActivatedValue( 0. ),
        loss( 0. ) {

}

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
    if( !previous.empty() )
        activatedValue = activationFunction -> activation( preActivatedValue );
}


template <class NeuronType>
void BaseNeuron <NeuronType> :: calculateLoss() {

    /// loss = sum( weights_to_next_layer * next.loss ) * activationDerivative( preactivatedValue )
    loss = 0.;
    for( auto edge : next ) {
        loss += edge -> getWeight() *
                edge -> getTo().getLoss();
    }
    if( activationFunction != nullptr )
        loss *= activationFunction -> activationDerivative( preActivatedValue );
}



template <class NeuronType>
void BaseNeuron <NeuronType> :: backpropagateNeuron() {

    for( auto edge : previous ) {

        /// update deltaWeight
        /// deltaWeight += this.loss * previous.activatedValue
        edge -> setDeltaWeight( edge -> getDeltaWeight() +
                                getLoss() * edge->getFrom().getActivatedValue() );
    }
}

template <class NeuronType>
void BaseNeuron <NeuronType> :: updateWeights( NeuronType learningRate, int batchSize ) {

    /// weight -= deltaWeight * (learningRate / batchSize)
    for( auto edge : previous ) {
        edge->updateWeight( ( learningRate / batchSize ) );
    }
}

