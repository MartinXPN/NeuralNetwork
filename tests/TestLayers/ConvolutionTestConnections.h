
#ifndef NEURALNETWORK_CONVOLUTIONTESTCONNECTIONS_H
#define NEURALNETWORK_CONVOLUTIONTESTCONNECTIONS_H

#include "../../lib/Layers/SimpleLayers/Convolution.h"
#include "../../lib/Activations/SimpleActivations/ReLU.h"
#include "../../lib/Layers/BaseLayers/BaseInputLayer.h"

void testConvolutionConnections() {
//    Convolution <double> first( {2, 9, 9}, {3, 3, 3}, new ReLU <double>(), {} );
//    Convolution <double> conv1( {2, 4, 4}, {2, 3, 3}, new ReLU <double>(), {&first}, {0, 2, 2} );

    BaseInputLayer <double> inputLayer( {1, 28, 28} );
    Convolution <double> conv1( { 10, 13, 13 }, { 1, 4, 4 }, new ReLU <double>(), {&inputLayer}, {0, 2, 2} );

    inputLayer.createNeurons();
    conv1.createNeurons();

    conv1.connectNeurons();

    printf( "Sample connection:\n" );
    for( auto connection : conv1.getNeurons()[0]->getPreviousConnections() ) {
        printf( "%lf\t", connection -> getWeight() );
    }
}

#endif //NEURALNETWORK_CONVOLUTIONTESTCONNECTIONS_H
