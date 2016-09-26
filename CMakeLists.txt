cmake_minimum_required(VERSION 3.6)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")

set(SOURCE_FILES main.cpp lib/BaseClasses/BaseNeuron.h lib/BaseClasses/BaseEdge.h lib/BaseClasses/BaseNeuralNetwork.h lib/Neurons/SigmoidNeuron.h lib/BaseClasses/BaseEdge.tpp lib/BaseClasses/BaseNeuron.tpp lib/Neurons/SigmoidNeuron.tpp)
add_executable(NeuralNetwork ${SOURCE_FILES})
