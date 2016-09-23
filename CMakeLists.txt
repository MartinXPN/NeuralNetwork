cmake_minimum_required(VERSION 3.6)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

set(SOURCE_FILES main.cpp BaseNeuron.cpp BaseNeuron.h BaseEdge.cpp BaseEdge.h BaseNeuralNetwoek.cpp BaseNeuralNetwoek.h)
add_executable(NeuralNetwork ${SOURCE_FILES})