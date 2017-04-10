
#ifndef NEURALNETWORK_BASELAYER_H
#define NEURALNETWORK_BASELAYER_H


#include <vector>
#include "../../neurons/base/BaseNeuron.h"


/**
 * Base ABSTRACT class for Layers
 * Every Layers is responsible for only 2 things:
 *      1. create neurons
 *      2. connect neurons
 * Layer is not responsible for activating neurons or backpropagarion...
 *
 * Type has to indicate the type of neurons that are kept in it, example : Layer <double> l;
 * Contains 1D collection of neurons, i.e even if we use convolutional layers,
 *      they have to be implemented in a way to be stored in 1D array
 */
template <class LayerType>
class BaseLayer {

protected:
    /// collections of neurons in this layer
    std :: vector< BaseNeuron <LayerType>* > neurons;

    /// number of neurons in this layer
    unsigned numberOfNeurons;

    /// dimensions of the layer
    std :: vector <unsigned> dimensions;

    /// all the previous layer that have connection to this layer
    std :: vector< const BaseLayer <LayerType>* > previousLayers;


public:
    /**
     * @param dimensions every layer may have different dimensions. For example when training a model on pictures one may use 2D convolutions + 1 dimensions for filters
     * @param previousLayers all previous layers that are connected to this layer
     */
    BaseLayer( const std :: vector <unsigned>& dimensions,
               const std :: vector< const BaseLayer <LayerType>* >& previousLayers );


    /**
     * @returns read-only collection of neurons in this layer
     */
    virtual const std :: vector< BaseNeuron <LayerType>* >& getNeurons() const {
        return neurons;
    }


    virtual unsigned size() const {
        return numberOfNeurons;
    }

    virtual const std :: vector <unsigned>& getDimensions() const {
        return dimensions;
    }



    /**
     * Has to create neurons and populate the vector< BaseNeuron <LayerType>* > neurons
     */
    virtual void createNeurons() = 0;



    /**
     * Has to connect all neurons in this layer to the neurons of the previous layer
     * To create a connection manually use function addPreviousLayerConnection( edge ); (the same edge as below)
     * And add connection from the previous layer too addNextLayerConnection( edge );   (the same edge as above)
     */
    virtual void connectNeurons() = 0;
};


#include "BaseLayer.tpp"

#endif //NEURALNETWORK_BASELAYER_H
