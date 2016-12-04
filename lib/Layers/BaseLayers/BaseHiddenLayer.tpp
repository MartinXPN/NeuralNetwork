
#include "BaseHiddenLayer.h"

template <class LayerType>
BaseHiddenLayer <LayerType> :: BaseHiddenLayer(const std :: vector <unsigned>& dimensions,
                                               const std::vector< const BaseLayer<LayerType>* > &previousLayers,
                                               BaseActivationFunction<LayerType> *activationFunction,
                                               Bias <LayerType>* bias)
        : BaseLayer <LayerType> (dimensions, previousLayers),
          activationFunction( activationFunction),
          bias( bias ) {

}


template <class LayerType>
void BaseHiddenLayer <LayerType> :: createNeurons() {

    for( int i=0; i < numberOfNeurons; ++i )
        neurons.push_back( new BaseNeuron <LayerType> ( activationFunction ) );
}

template <class LayerType>
BaseHiddenLayer <LayerType> :: ~BaseHiddenLayer() {

    for( int i=0; i < weights.size(); ++i )         delete weights[i];
    for( int i=0; i < deltaWeights.size(); ++i )    delete deltaWeights[i];
}
