# Bacon
Bacon is a neural network written purely in Swift and packaged as a Cocoa framework. It uses Apple's [Accelerate](https://developer.apple.com/reference/accelerate) framework to perform high speed mathematical computations. One of our overarching goals is to make Bacon capable of being trained in a production environment. This requires the framework to be as efficient as possible.

## The Name
Curious where the name Bacon came from? This repository contains a Blue Cocoa Neural Network. The acronym for Blue Cocoa Neural Network is BCNN. Try pronouncing BCNN phonetically. What does it sound like? Bacon 🥓

## Functionality
- Neural network with one hidden layer and one activation function. Accepts input and output as `[[Double]]`.
- Written in pure Swift and with the Accelerate framework for high efficiency.

## Installation
Copy the BaconFramework files (including the .xcodeproj) to your project directory. Drag the .xcodeproj into your project. Ensure that BaconFramework.framework is added as an Embedded Binary and Linked Framework. Build the framework. Voilà.

You can use the BaconSampleProject to see an example of Bacon in action. Go to `ViewController.swift` to see the configuration and training call. Run the project and watch the neural network get trained.

## Usage
### Creating a simple input and output matrix
If you just want to mess around with the neural network, you can make up any pattern and test it. This pattern is that the output is 1 whenever the first two input columns have some combination of 0 and 1. In all other input situations, the output is 0. This pattern is similar to the XOR problem, except there is an extra column.
```swift
var input: [[Double]] = [ 
        [1, 0, 1],
        [1, 1, 1], 
        [0, 0, 1], 
        [1, 1, 0], 
        [1, 0, 0] 
]

var output: [[Double]] = [
        [1],
        [0],
        [0],
        [0],
        [1]
]
```
### Configuring the network
Initialize the `NeuralNetwork` class with the parameters you wish.
```swift
let neuralNetwork = NeuralNetwork(activationFunction: .sigmoid, iterations: 1000, verbose: true)
```
### Training the network
After initializing the `NeuralNetwork` class, call `train(input, output)`. This function will train the neural network based on the sample data you provide. After completing the training process, this function will return an optional tuple containing the hidden layer matrix and weights matrix flattened as arrays. You can store this tuple for later evaluation or weights use.
```swift
let structure: (hiddenLayer: [Double], weights: [Double])? = neuralNetwork.train(input, output)
```

## Roadmap
- More activation functions
- N hidden layers
- Save weights for testing
- Various kinds of layers built modularly
- Convolutional neural network
- Metal exploration

## License
Bacon is available under the Apache License 2.0. See the LICENSE file for more information.
