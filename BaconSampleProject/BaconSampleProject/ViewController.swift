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
        
        /* **** The pattern here is: if either column 1 or 2 of the input is 1, then the output is 1. However, if column 1 and 2 are both 1 or both 0, then the output is 0. Column 3 is irrelevant. Below are the results and configuration of one of our tests:
         
         input:
         [[1, 0, 1],
         [1, 1, 1],
         [0, 0, 1],
         [1, 1, 0],
         [1, 0, 0]]
         
         output:
         [[1],
         [0],
         [0],
         [0],
         [1]]
         
         Hidden layer:
         [0.99495044083462603, 2.1832008463678404e-05, 0.005533223089738218, 0.0039085393430607846, 0.99997176240828434]
         
         Weights:
         [10.47483834281852, -16.015518754305521, -5.1914412862863717]
         **** */
        
        // M = how many samples you want
        let M = 1000
        
        // Creates a M x 3 input matrix with random zeros and ones
        var input = [[Double]](repeating: [Double](repeating: 0, count: 3), count: M)
        input = input.map {
            $0.map {
                $0 + Double(arc4random_uniform(2))
            }
        }
        
        // Creates a M x 1 output matrix based on the pattern described above
        var output = [[Double]](repeating: [Double](repeating: 0, count: 1), count: M)
        for (rowIndex, row) in input.enumerated() {
            for (index, value) in row.enumerated() {
                if index == 0 && value == 1 && row[index + 1] == 0 {
                    output[rowIndex] = [1]
                    break
                } else if index == 1 && value == 1 && row[index - 1] == 0 {
                    output[rowIndex] = [1]
                    break
                }
            }
        }
        
        // Setup and train the neural network
        let neuralNetwork = NeuralNetwork(activationFunction: .sigmoid, iterations: 10000, verbose: true)
        let structure = neuralNetwork.train(input, output)
        
        // Calculate the number of unsure values. Values in the hidden layer that = 0.5 imply that the neural network wasn't sure of whether they should be a 0 or 1.
        if let structure = structure {
            print("\nNumber of unsure values (value = 0.5):")
            print(structure.hiddenLayer.filter { $0 == 0.5 }.count)
        }
    }
    
    override var representedObject: Any? {
        didSet {
            // Update the view, if already loaded.
        }
    }
    
    
}

