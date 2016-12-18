
#include "DynamicNetwork.h"


template <class NetworkType>
size_t DynamicNetwork <NetworkType> :: getSmallWeightsNumber(NetworkType threshold) {

    size_t res = 0;
    for( auto bucket : buckets )
        for( auto neuron : bucket ) {
            for( auto edge : neuron->getPreviousConnections() )
                if (fabs(edge->getWeight()) < threshold) {
                    ++res;
                }
        }

    return res;
}


template <class NetworkType>
void DynamicNetwork <NetworkType> :: pruneNetwork( NetworkType threshold ) {

    for( auto bucket : buckets ) {
        for( auto neuron : bucket ) {
            for( int i=0; i < neuron->getPreviousConnections().size(); ++i ) {
                auto edge = neuron->getPreviousConnections()[i];
                if (fabs(edge->getWeight()) < threshold) {
                    neuron->removePreviousLayerConnection( &( (NetworkType &) edge -> getWeight() ) );
                    --i;
                }
            }
        }
    }
}


template <class NetworkType>
void DynamicNetwork <NetworkType> :: pruneLayers( NetworkType threshold, std::vector<BaseLayer<NetworkType> *> layers ) {

    for( BaseLayer<NetworkType> * layer : layers ) {
        for( auto neuron : layer -> getNeurons() ) {
            for( int i=0; i < neuron->getPreviousConnections().size(); ++i ) {
                auto edge = neuron->getPreviousConnections()[i];
                if (fabs(edge->getWeight()) < threshold) {
                    neuron->removePreviousLayerConnection( &( (NetworkType &) edge -> getWeight() ) );
                    --i;
                }
            }
        }
    }
}
