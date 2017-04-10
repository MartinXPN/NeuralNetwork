
#include "DynamicNetwork.h"


template <class NetworkType>
size_t DynamicNetwork <NetworkType> :: getSmallWeightsNumber(NetworkType threshold) {

    size_t res = 0;
    for( const auto& bucket : buckets )
        for( auto neuron : bucket )
            for( auto edge : neuron->getPreviousConnections() )
                if (fabs(edge->getWeight()) < threshold) {
                    ++res;
                }

    return res;
}


template <class NetworkType>
void DynamicNetwork <NetworkType> :: pruneNetwork( NetworkType threshold ) {

    for( auto& bucket : buckets ) {
        for( int i=0; i < bucket.size(); ++i ) {
            auto neuron = bucket[i];
            pruneNeuronPreviousLayerConnections( neuron, threshold );
            pruneNeuronNextLayerConnections( neuron, threshold );

            /// if at some point there is a neuron that doesn't contribute to the output of a network
            /// (i.e. doesn't have connections to the next layer or the previous one) => we have to remove it
            if( neuron -> getNextConnections().empty() || neuron -> getPreviousConnections().empty() ) {
                bucket.erase( std::find( bucket.begin(), bucket.end(), neuron ) );
                delete neuron;
                --i;
            }
        }
        if( bucket.empty() ) {
            throw "The whole layer just got empty!";
        }
    }
}


template <class NetworkType>
void DynamicNetwork <NetworkType> :: pruneLayers( NetworkType threshold, std::vector<BaseLayer<NetworkType> *> layers ) {

    for( auto layer : layers ) {
        for( auto neuron : layer -> getNeurons() ) {
            pruneNeuronPreviousLayerConnections( neuron, threshold );
        }
    }
}


template <class NetworkType>
void DynamicNetwork <NetworkType> :: pruneNeuronPreviousLayerConnections( BaseNeuron<NetworkType> *neuron, NetworkType threshold ) {

    for( int i=0; i < neuron->getPreviousConnections().size(); ++i ) {
        auto edge = neuron->getPreviousConnections()[i];
        if( fabs(edge->getWeight()) < threshold ) {
            neuron->removePreviousLayerConnection( &( (NetworkType &) edge -> getWeight() ) );
            --i;
        }
    }
}


template <class NetworkType>
void DynamicNetwork <NetworkType> :: pruneNeuronNextLayerConnections( BaseNeuron<NetworkType> *neuron, NetworkType threshold ) {

    for( int i=0; i < neuron->getNextConnections().size(); ++i ) {
        auto edge = neuron->getNextConnections()[i];
        if( fabs(edge->getWeight()) < threshold ) {
            neuron->removeNextLayerConnection( &( (NetworkType &) edge -> getWeight() ) );
            --i;
        }
    }
}
