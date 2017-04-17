
#include "InputLayer.h"
#include "../util/MathOperations.h"


template <class LayerType>
InputLayer <LayerType> :: InputLayer( const std::vector<unsigned> &dimensions,
                                      NeuronInitializer<LayerType> *neuronInitializer )
        : InputLayer(dimensions, neuronInitializer -> createNeurons( math::multiply(dimensions)) ) {

}


template <class LayerType>
InputLayer <LayerType> :: InputLayer( const std::vector<unsigned> &dimensions,
                                      const std :: vector< BaseNeuron <LayerType>* >& neurons)
        : BaseLayer <LayerType> ( dimensions, {}, neurons ) {

}
