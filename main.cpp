#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;


#include "lib/BaseClasses/Neurons/BaseInputNeuron.h"
#include "lib/BaseClasses/Neurons/BaseOutputNeuron.h"
#include "lib/Activations/SimpleActivations/ReLU.h"
#include "lib/Activations/SimpleActivations/Sigmoid.h"


int main() {

    const int numberOfNeurons = 8;
    const int numberOfEdges = 13;

    vector< BaseNeuron <double>* > neurons;
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 1 ) );  // layer{1} first input neuron
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 0 ) );  // layer{1} second input neuron
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 1 ) );  // layer{1} bias neuron

    neurons.push_back( new BaseNeuron <double>( new ReLU <double>() ) );            // layer{2} first neuron
    neurons.push_back( new BaseNeuron <double>( new ReLU <double>() ) );            // layer{2} second neuron
    neurons.push_back( new BaseNeuron <double>( new ReLU <double>() ) );            // layer{2} third neuron
    neurons.push_back( (BaseNeuron <double>*) new BaseInputNeuron <double>( 1 ) );  // layer{2} bias neuron

    neurons.push_back( (BaseNeuron <double>*) new BaseOutputNeuron <double>( new Sigmoid <double> () ) );    // layer{3} output neuron

/*
0 3
0 4
0 5
1 3
1 4
1 5
2 3
2 4
2 5
3 7
4 7
5 7
6 7
*/
    srand( 1 );
    cout << "Get the edges..." << endl;
    /// get values of all edges
    for( int i=0; i < numberOfEdges; ++i ) {
        int from, to;
        double *weight = new double( ( rand()%10 - 5 ) / 5. );
        cin >> from >> to;

        BaseEdge <double>* edge = new BaseEdge <double>( neurons[from], neurons[to], weight );
        neurons[from] -> addNextLayerConnection( edge );
        neurons[to] -> addPreviousLayerConnection( edge );
    }


    const int batchSize = 4;
    double learningRate = 0.01;
    for( int iteration = 0; iteration < 100000; ++iteration ) {

        double loss = 0;
        for( int batch = 0; batch < batchSize; ++batch ) {
            /// set values of input neurons
            double one = rand() % 2;
            double two = rand() % 2;
            double out = (int) one ^ (int) two;

            ((BaseInputNeuron<double> *) neurons[0])->setValue(one);
            ((BaseInputNeuron<double> *) neurons[1])->setValue(two);

            /// activate neurons
            for (int i = 3; i < numberOfNeurons; ++i) {
                neurons[i]->activateNeuron();
            }


            /// calculate losses
            double currentLoss = ((BaseOutputNeuron<double> *) neurons.back())->calculateLoss(out);
            loss += currentLoss;
            if( iteration == 49999 ) {
                cout << "(" << one << "," << two << ") -> " << out << "\tout: "
                     << ((BaseOutputNeuron<double> *) neurons.back())->getValue() << "   \t"
                     << "loss #" << iteration << ": " << currentLoss
                     << endl;
            }

            for (int i = numberOfNeurons - 2; i >= 3; --i) {
                neurons[i]->calculateLoss();
            }

            /// backpropagate neurons
            for (int i = numberOfNeurons - 2; i >= 0; --i) {
                neurons[i]->backpropagateNeuron(learningRate, batchSize);
            }
        }

//        cout << "Loss #" << iteration << ": " << loss / batchSize << endl;

        /// update weights
        for (int i = numberOfNeurons - 2; i >= 0; --i) {
            neurons[i] -> updateWeights();
        }
    }

    cout << "Everything is done" << endl;
    return 0;
}