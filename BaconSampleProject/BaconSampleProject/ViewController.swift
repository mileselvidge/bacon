//
//  ViewController.swift
//  BaconSampleProject
//
//  Created by Shaan Singh on 4/24/17.
//  Copyright © 2017 Blue Cocoa, Inc. All rights reserved.
//

import Cocoa
import BaconFramework

class ViewController: NSViewController {

    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
        
        /* **** The pattern here is: if either column 1 or 2 of the input is 1, then the output is 1. However, if column 1 and 2 are both 1 or both 0, then the output is 0. Column 3 is irrelevant. If you run this neural network with ActivationFunction.sigmoid and 100,000 iterations, the hidden layer results should be quite accurate. Below is the results of one of our tests:
            
            Hidden layer:
            [0.99495044083462603, 2.1832008463678404e-05, 0.005533223089738218, 0.0039085393430607846, 0.99997176240828434]
         
            Weights:
            [10.47483834281852, -16.015518754305521, -5.1914412862863717]
        **** */
        
        let input: [[Double]] = [
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 0]
        ]
        
        let output: [[Double]] = [
            [1],
            [0],
            [0],
            [0],
            [1]
        ]
        
        let neuralNetwork = NeuralNetwork(activationFunction: .sigmoid, iterations: 100000, verbose: true)
        neuralNetwork.train(input, output)
    }

    override var representedObject: Any? {
        didSet {
        // Update the view, if already loaded.
        }
    }


}

