
#include <queue>
#include <set>
#include <cassert>
#include <map>
#include "NeuralNetwork.h"



template <class NetworkType>
NeuralNetwork <NetworkType> :: NeuralNetwork( std::vector<BaseInputLayer <NetworkType>* > inputLayers,
                                              std::vector<BaseHiddenLayer <NetworkType>* > hiddenLayers,
                                              std::vector<BaseOutputLayer <NetworkType>* > outputLayers)
        : inputLayers( inputLayers ), hiddenLayers( hiddenLayers ), outputLayers( outputLayers ) {
}



template <class NetworkType>
void NeuralNetwork <NetworkType> :: initializeNetwork() {

    /// initialize all layers by calling 3 virtual function -> ( createNeurons, createWeights, connectNeurons )
    for( auto layer : inputLayers )     layer -> createNeurons();
    for( auto layer : hiddenLayers )    layer -> createNeurons();
    for( auto layer : outputLayers )    layer -> createNeurons();

    for( auto layer : hiddenLayers )    layer -> createWeights();

    for( auto layer : hiddenLayers )    layer -> connectNeurons();
    for( auto layer : outputLayers )    layer -> connectNeurons();


    calculatePropagationOrder();
}


template <class NetworkType>
void NeuralNetwork <NetworkType> :: calculatePropagationOrder() {

    /// get input neurons
    for( auto layer : inputLayers )
        for( auto neuron : layer -> getNeurons() )
            inputNeurons.push_back( (BaseInputNeuron<NetworkType>*)neuron );

    /// get buckets
    buckets = divideIntoBuckets( getInputNeuronsAndBiases() );

    /// get output neurons
    for( auto layer : outputLayers )
        for( auto neuron : layer -> getNeurons() )
            outputNeurons.push_back( (BaseOutputNeuron<NetworkType>*)neuron );
}

template <class NetworkType>
std::vector<BaseNeuron<NetworkType> *> NeuralNetwork <NetworkType> :: getInputNeuronsAndBiases() {

    std :: vector <BaseNeuron <NetworkType>* > result;
    std :: queue <BaseNeuron <NetworkType>* > q;
    std :: set <BaseNeuron <NetworkType>* > used;
    for( auto layer : inputLayers )
        for( auto neuron : layer -> getNeurons() ) {
            q.push( neuron );
            used.insert( neuron );
        }

    while( !q.empty() ) {
        BaseNeuron <NetworkType>* neuron = q.front();
        q.pop();
        if( neuron == nullptr )
            continue;

        if( neuron -> getPreviousConnections().empty() ) {
            assert( neuron -> getPreviousConnections().empty() );
            result.push_back(neuron);
        }

        for( BaseEdge <NetworkType>* edge : neuron -> getNextConnections() ) {
            auto next = (BaseNeuron <NetworkType>*) &edge->getTo();
            if (used.find(next) == used.end()) {
                used.insert(next);
                q.push(next);
            }
        }

        for( BaseEdge <NetworkType>* edge : neuron -> getPreviousConnections() ) {
            auto previous = (BaseNeuron <NetworkType>*) &edge->getFrom();
            if (used.find(previous) == used.end()) {
                used.insert(previous);
                q.push(previous);
            }
        }
    }

    return result;
}

template <class NetworkType>
std :: vector< std :: vector <BaseNeuron<NetworkType>* > > NeuralNetwork <NetworkType> :: divideIntoBuckets( std::vector<BaseNeuron<NetworkType> *> startOffNeurons ) {

    for( BaseNeuron<NetworkType> * neuron : startOffNeurons ) {
        assert( neuron -> getPreviousConnections().empty() );
    }

    std :: vector< std :: vector <BaseNeuron<NetworkType>* > > buckets;
    std :: map <BaseNeuron <NetworkType>*, std :: pair <size_t, size_t> > used; /// [neuron] -> {number_of_activated_previous_neurons, time}
    std :: queue <BaseNeuron <NetworkType>*> q;

//    buckets.push_back( startOffNeurons );
    for( auto neuron : startOffNeurons ) {
        q.push(neuron);
        used[neuron] = {0, -1};
    }

    while( !q.empty() ) {
        BaseNeuron <NetworkType>* neuron = q.front();
        q.pop();
        if( neuron == nullptr )
            continue;

        for( BaseEdge <NetworkType>* edge : neuron -> getNextConnections() ) {
            auto nextNeuron = ( (BaseNeuron <NetworkType>*) &edge->getTo() );
            if( ++used[nextNeuron].first == nextNeuron->getPreviousConnections().size() ) {
                q.push(nextNeuron);
                used[nextNeuron].second = used[neuron].second + 1;

                /// add neuron to corresponding bucket if it's not an output neuron
                if( nextNeuron -> getNextConnections().empty() )
                    continue;
                if( buckets.size() <= used[nextNeuron].second )
                    buckets.push_back( {} );
                buckets[ used[nextNeuron].second ].push_back( nextNeuron );
            }
        }
    }

    return buckets;
}


template <class NetworkType>
void NeuralNetwork <NetworkType> :: trainEpoch( size_t numberOfInputs,
                                                size_t batchSize,
                                                double learningRate,
                                                std::vector <NetworkType> (*inputLoader)(size_t),
                                                std::vector <NetworkType> (*labelLoader)(size_t),
                                                void (*onEpochTrained)(void),
                                                void (*onBatchProcessed)(void)) {


    for (size_t batchStart = 0; batchStart < numberOfInputs; batchStart += batchSize) {
        for (size_t i = batchStart; i < batchStart + batchSize && i < numberOfInputs; ++i) {
            /// set values of input neurons
            auto currentInput = inputLoader( i );
            for( size_t j=0 ; j < inputNeurons.size(); ++j )
                inputNeurons[j] -> setValue( currentInput[j] );

            /// activate neurons
            for( auto bucket : buckets )
                for( auto neuron : bucket )
                    neuron -> activateNeuron();

            /// calculate loss
            auto currentLabel = labelLoader( i );
            for( int j=0; j < outputNeurons.size(); ++j )
                outputNeurons[j] -> calculateLoss( currentLabel[j] );
            for( int bucketNumber = buckets.size()-1; bucketNumber >= 0; --bucketNumber ) {
                for( auto neuron : buckets[bucketNumber] ) {
                    neuron -> calculateLoss();
                }
            }

            /// backpropagate neurons
            for( auto neuron : outputNeurons )
                neuron -> backpropagateNeuron();
            for( int bucketNumber = buckets.size()-1; bucketNumber >= 0; --bucketNumber ) {
                for( auto neuron : buckets[bucketNumber] ) {
                    neuron -> backpropagateNeuron();
                }
            }
        }

        if( onBatchProcessed != nullptr )
            onBatchProcessed();
    }

    /// update weights
    for (auto neuron : outputNeurons)
        neuron->updateWeights(learningRate, batchSize);
    for( auto bucket : buckets )
        for( auto neuron : bucket )
            neuron->updateWeights(learningRate, batchSize);

    if( onEpochTrained != nullptr )
        onEpochTrained();
}