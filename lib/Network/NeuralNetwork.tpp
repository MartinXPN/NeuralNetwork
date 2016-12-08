
#include <queue>
#include <set>
#include <cassert>
#include <map>
#include "NeuralNetwork.h"



template <class NetworkType>
NeuralNetwork <NetworkType> ::NeuralNetwork(std::vector<BaseInputLayer <NetworkType>* > inputLayers,
                                            std::vector<BaseHiddenLayer <NetworkType>* > hiddenLayers,
                                            std::vector<BaseOutputLayer <NetworkType>* > outputLayers) :
        inputLayers( inputLayers ), hiddenLayers( hiddenLayers ), outputLayers( outputLayers ) {


    for( auto layer : inputLayers )     layer -> createNeurons();
    for( auto layer : hiddenLayers )    layer -> createNeurons();
    for( auto layer : outputLayers )    layer -> createNeurons();

    for( auto layer : hiddenLayers )    layer -> createWeights();

    for( auto layer : hiddenLayers )    layer -> connectNeurons();
    for( auto layer : outputLayers )    layer -> connectNeurons();
}



template <class NetworkType>
void NeuralNetwork <NetworkType> :: initializeNetwork() {

    calculatePropagationOrder();
}


template <class NetworkType>
void NeuralNetwork <NetworkType> :: calculatePropagationOrder() {

    buckets = divideIntoBuckets( getInputNeuronsAndBiases() );
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
            auto next = ( (BaseNeuron <NetworkType>*) &edge->getTo() );
            if (used.find(next) == used.end()) {
                used.insert(next);
                q.push(next);
            }
        }

        for( BaseEdge <NetworkType>* edge : neuron -> getPreviousConnections() ) {
            auto previous = ( (BaseNeuron <NetworkType>*) &edge->getFrom() );
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

    buckets.push_back( startOffNeurons );
    for( auto neuron : startOffNeurons ) {
        q.push(neuron);
        used[neuron] = {0, 0};
    }

    while( !q.empty() ) {
        BaseNeuron <NetworkType>* neuron = q.front();
        q.pop();
        if( neuron == nullptr )
            continue;

        for( BaseEdge <NetworkType>* edge : neuron -> getNextConnections() ) {
            auto nextNeuron = ( (BaseNeuron <NetworkType>*) &edge->getTo() );
            if( used.find( nextNeuron ) == used.end() ) {
                used[nextNeuron] = { 0, used[neuron].second + 1 };
            }
            if( ++used[nextNeuron].first == nextNeuron->getPreviousConnections().size() ) {
                q.push(nextNeuron);
                used[nextNeuron].second = used[neuron].second + 1;
                /// add neuron to corresponding bucket
                if( buckets.size() <= used[nextNeuron].second ) {
                    buckets.push_back({});
                }
                buckets[ used[nextNeuron].second ].push_back( nextNeuron );
            }
        }
    }

    return buckets;
}
