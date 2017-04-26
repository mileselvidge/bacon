//
//  NeuralNetwork.swift
//  Bacon
//
//  Created by Shaan Singh on 4/21/17.
//  Copyright © 2017 Blue Cocoa, Inc. All rights reserved.
//

import Foundation
import Accelerate

public class NeuralNetwork {
    
    var activationFunction: ActivationFunction!
    var iterations: Int!
    var verbose: Bool!
    
    /**
     Initialize the neural network. All parameters come with defaults (listed). Call train() when ready.
     - parameter activationFunction: ActivationFunction.sigmoid
     - parameter iterations: 1000
     - parameter verbose: false
     */
    public init(activationFunction: ActivationFunction = .sigmoid, iterations: Int = 1000, verbose: Bool = false) {
        self.activationFunction = activationFunction
        self.iterations = iterations
        self.verbose = verbose
    }
    
    /**
     Trains the neural network after initialization.
     - parameter input: Each row in the input matrix corresponds to one sample. The number of columns is the number of input nodes.
     - parameter output: Like the input matrix, each row corresponds to one sample. The number of columns is the number of output nodes.
     */
    public func train(_ input: [[Double]], _ output: [[Double]]) {
        
        // Seed
        srand48(1)
        
        // Prepare for weights initialization
        let inpNodes = input[0].count
        let outNodes = output[0].count
        var synapse = [[Double]](repeating: [Double](repeating: 0, count: outNodes), count: inpNodes)
        
        // Use node counts to randomly generate weights
        for i in 0 ..< inpNodes {
            for o in 0 ..< outNodes {
                synapse[i][o] = 2 * drand48() - 1
            }
        }
        
        print("Beginning training...")
        
        for iteration in 1...iterations {
            
            if verbose {
                print("Iteration \(iteration)")
            }
            
            let layer0 = input
            
            // Forward prop step 1: layer0 * synapse
            var dotted = [Double](repeating: 0, count: layer0.count * synapse[0].count)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(layer0.count), Int32(synapse[0].count), Int32(layer0[0].count), 1.0, Array(layer0.joined()), Int32(layer0[0].count), Array(synapse.joined()), Int32(synapse[0].count), 0.0, &(dotted), Int32(synapse[0].count))
            
            // Forward prop step 2: activation function
            let layer1 = calculate(dotted, with: .sigmoid)
            
            // Make layer1 negative
            var negLayer1 = layer1
            vDSP_vnegD(layer1, 1, &(negLayer1), 1, vDSP_Length(negLayer1.count))
            
            // Layer1 error = output - layer1
            var layer1Error = negLayer1
            cblas_daxpy(Int32(Array(output.joined()).count), 1.0, Array(output.joined()), 1, &(layer1Error), 1)
            
            // Get derivative of sigmoid
            let derivative = calculateDerivative(layer1, with: .sigmoid)
            
            // Error weighted derivative = error * derivative of sigmoid (element-wise multiplication)
            var errorWeightedDerivative = [Double](repeating: 0, count: layer1Error.count)
            vDSP_vmulD(layer1Error, 1, derivative, 1, &errorWeightedDerivative, 1, vDSP_Length(layer1Error.count))
            
            // Transpose layer0
            var transposed = [Double](repeating: 0, count: Array(layer0.joined()).count)
            vDSP_mtransD(Array(layer0.joined()), 1, &(transposed), 1, vDSP_Length(layer0[0].count), vDSP_Length(layer0.count))
            
            // Compute layer0.T * errorWeightedDerivative
            var dottedFinal = [Double](repeating: 0, count: layer0[0].count * output[0].count)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(layer0[0].count), Int32(output[0].count), Int32(layer0.count), 1.0, transposed, Int32(layer0.count), errorWeightedDerivative, Int32(output[0].count), 0.0, &(dottedFinal), Int32(output[0].count))
            
            // Adjust the weights
            var results = dottedFinal
            cblas_daxpy(Int32(Array(synapse.joined()).count), 1.0, Array(synapse.joined()), 1, &(results), 1)
            
            // Map results back to synapse's inherent structure
            synapse = synapse.enumerated().map({ (rowIndex, row) in
                return row.enumerated().map({ (index, value) in
                    return results[index + rowIndex * row.count]
                })
            })
            
            if iteration == iterations {
                print("Training complete!\n")
                
                print("Hidden layer:")
                print("\(layer1)\n")
                
                print("Weights:")
                print(Array(synapse.joined()))
            }
            
        }
    }
    
}
