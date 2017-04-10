
#ifndef NEURALNETWORK_NETWORKBUCKETTEST_H
#define NEURALNETWORK_NETWORKBUCKETTEST_H

#include <algorithm>
#include <iostream>
#include "../../library/network/NeuralNetwork.h"
#include "../../library/layers/FullyConnected.h"
#include "../../library/activations/ReLU.h"
#include "../../library/lossfunctions/CrossEntropyCost.h"

void testBuckets() {

    using std::cout;
    using std::endl;
    using std::find;

    Bias <double> *bias = new Bias <double>();
    auto in1 = BaseInputLayer <double>( {10, 20} );
    auto in2 = BaseInputLayer <double>( {20, 30} );
    auto fc1 = FullyConnected <double> ( {20}, new ReLU <double>(), {&in1, &in2}, bias );
    auto fc2 = FullyConnected <double> ( {30}, new ReLU <double>(), {&in2} );
    auto out1 = BaseOutputLayer <double> ( {10}, {&fc1, &fc2}, new CrossEntropyCost <double>(), new ReLU <double>(), bias );
    auto out2 = BaseOutputLayer <double> ( {10}, {&in1}, new CrossEntropyCost <double>(), new ReLU <double>(), bias );

    NeuralNetwork <double> net( { &in1, &in2 },
                                { &fc1, &fc2 },
                                { &out1, &out2 } );


    net.initializeNetwork();
    auto buckets = net.getBuckets();
    cout << "Size of buckets: " << buckets.size() << endl;
    cout << "Shape of buckets: ";
    for( auto& bucket : buckets )
        cout << bucket.size() << " ";
    cout << endl;

    for( auto neuron : buckets[0] ) {
        if( find( fc1.getNeurons().begin(), fc1.getNeurons().end(), neuron ) != fc1.getNeurons().end() ||
            find( fc2.getNeurons().begin(), fc2.getNeurons().end(), neuron ) != fc2.getNeurons().end() );
        else {
            cout << "Error -> " << neuron << endl;
        }
    }
}

#endif //NEURALNETWORK_NETWORKBUCKETTEST_H
