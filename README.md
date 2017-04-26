# Bacon
Bacon is a neural network written purely in Swift. It uses Apple's [Accelerate](https://developer.apple.com/reference/accelerate) framework to make high speed mathematical computations. One of our overarching goals is to make Bacon capable of being trained in a production environment. This requires the framework to be as efficient as possible.

## The Name
Curious where the name Bacon came from? This repository contains a Blue Cocoa Neural Network. The acronym for this phrase is BCNN. Try pronouncing BCNN phonetically. What does it sound like? Bacon.

## Functionality
- Neural network with one hidden layer and one activation function. Accepts input and output as `Double`.
- Written in pure Swift and with the Accelerate framework for high efficiency.

## Roadmap
- More activation functions
- Save weights for testing
- Various kinds of layers built modularly
- Convolutional neural network
- Metal exploration

## Installation
Copy the BaconFramework files (including the .xcodeproj) to your project directory. Drag the .xcodeproj into your project. Ensure that BaconFramework.framework is added as an Embedded Binary and Linked Framework. Voilà.

You can use the BaconSampleProject to see an example of Bacon in action. Go to `ViewController.swift` to see the configuration and training call. Run the project and watch the neural network get trained.

## License
Bacon is available under the Apache License 2.0. See the LICENSE file for more information.
