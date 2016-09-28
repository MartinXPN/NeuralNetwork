cmake_minimum_required(VERSION 3.6)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")

set(SOURCE_FILES main.cpp lib/BaseClasses/Neurons/BaseNeuron.h lib/BaseClasses/Edges/BaseEdge.h lib/Neurons/SigmoidNeuron.h lib/BaseClasses/Edges/BaseEdge.tpp lib/BaseClasses/Neurons/BaseNeuron.tpp lib/Neurons/SigmoidNeuron.tpp lib/BaseClasses/Neurons/BaseInputNeuron.tpp lib/BaseClasses/Neurons/BaseInputNeuron.h lib/BaseClasses/Neurons/BaseOutputNeuron.tpp lib/BaseClasses/Neurons/BaseOutputNeuron.h lib/Activations/BaseActivation/BaseActivationFunction.h lib/Activations/SimpleActivations/Sigmoid.h lib/Activations/SimpleActivations/ReLU.h lib/Activations/SimpleActivations/Tanh.h lib/Activations/SimpleActivations/Identitiy.h)
add_executable(NeuralNetwork ${SOURCE_FILES})
