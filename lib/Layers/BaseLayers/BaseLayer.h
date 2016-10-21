
#ifndef NEURALNETWORK_BASELAYER_H
#define NEURALNETWORK_BASELAYER_H


#include <vector>
#include "../../Neurons/BaseNeurons/BaseNeuron.h"


/**
 * Base ABSTRACT class for Layers
 * Type has to indicate the type of neurons that are kept in it, example : Layer <double> l;
 * Contains 1D collection of neurons, i.e even if we use convolutional layers,
 *      they have to be implemented in a way to be stored in 1D array
 *
 * The constructor of BaseLayer calls 2 abstract functions
 *      1. createNeurons
 *      2. connectNeurons
 */
template <class LayerType>
class BaseLayer {

protected:
    std :: vector< BaseNeuron <LayerType>* > neurons;        /// collections of neurons in this layer

public:
    /**
     * Calls 2 abstract functions
     *      1. createNeurons
     *      2. connectNeurons
     *
     * @param numberOfNeurons number of neurons in the layer
     * @param previousLayers all previous layers that are connected to this layer
     */
    BaseLayer( unsigned numberOfNeurons,
               const std :: vector< const BaseLayer <LayerType>* > previousLayers );


    /**
     * @returns read-only collection of neurons in this layer
     */
    virtual const std :: vector< BaseNeuron <LayerType>* >& getNeurons() const { return neurons; }



    /**
     * Has to create neurons and push them in vector< BaseNeuron <LayerType>* > neurons
     * @param numberOfNeurons number of neurons to create (equal to the number of neurons in this layer given to constructor)
     */
    virtual void createNeurons( unsigned numberOfNeurons ) = 0;



    /**
     * Has to connect all neurons in this layer to the neurons of the previous layer
     * To create a connection use function addPreviousLayerConnection( edge );          (the same edge as below)
     * And add connection from the previous layer too addNextLayerConnection( edge );   (the same edge as the above)
     *
     * @param previous previous layer which is connected to this layer
     */
    virtual void connectNeurons( const BaseLayer <LayerType>& previous ) = 0;
};


#include "BaseLayer.tpp"

#endif //NEURALNETWORK_BASELAYER_H
