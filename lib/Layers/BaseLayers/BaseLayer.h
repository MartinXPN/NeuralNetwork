
#ifndef NEURALNETWORK_BASELAYER_H
#define NEURALNETWORK_BASELAYER_H


#include <vector>
#include "../../Neurons/BaseNeurons/BaseNeuron.h"


/**
 * Base ABSTRACT class for Layers
 * Layer type has to indicate the neurons that are kept in it, example : Layer < BaseInputNeuron <double> > l;
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
    std :: vector< LayerType* > neurons;        /// collections of neurons in this layer

public:
    /**
     * Calls 2 abstract functions
     *      1. createNeurons
     *      2. connectNeurons
     */
    BaseLayer( unsigned numberOfNeurons,
               const BaseLayer <LayerType>* previous );


    /**
     * Returns the collection of neurons in this layer
     */
    virtual const std :: vector< LayerType* >& getNeurons() const { return neurons; }



    /**
     * Has to create neurons and push them in vector <LayerType*> neurons
     */
    virtual void createNeurons( unsigned numberOfNeurons ) = 0;



    /**
     * Has to connect all neurons in this layer to the neurons of the previous layer
     * To create a connection use function addPreviousLayerConnection( edge );          (the same edge as below)
     * And add connection from the previous layer too addNextLayerConnection( edge );   (the same edge as the above)
     */
    virtual void connectNeurons( BaseLayer <LayerType>* previous ) = 0;
};


#include "BaseLayer.tpp"

#endif //NEURALNETWORK_BASELAYER_H
