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
     Trains the neural network after initialization. Returns the hidden layer matrix and weights matrix flattened as arrays.
     - parameter input: Each row in the input matrix corresponds to one sample. The number of columns is the number of input nodes.
     - parameter output: Like the input matrix, each row corresponds to one sample. The number of columns is the number of output nodes.
     */
    public func train(_ input: [[Double]], _ output: [[Double]]) -> (hiddenLayer: [Double], weights: [Double])? {
        
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
            
            let layer0 = input
            
            // Forward prop step 1: layer0 * synapse
            var multiplied = [Double](repeating: 0, count: layer0.count * synapse[0].count)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(layer0.count), Int32(synapse[0].count), Int32(layer0[0].count), 1.0, Array(layer0.joined()), Int32(layer0[0].count), Array(synapse.joined()), Int32(synapse[0].count), 0.0, &(multiplied), Int32(synapse[0].count))
            
            // Forward prop step 2: activation function
            let layer1 = calculate(multiplied, with: .sigmoid)
            
            // Make layer1 negative
            var negLayer1 = layer1
            vDSP_vnegD(layer1, 1, &(negLayer1), 1, vDSP_Length(negLayer1.count))
            
            // Layer1 cost derivative = output - layer1
            var layer1CostDerivative = negLayer1
            cblas_daxpy(Int32(Array(output.joined()).count), 1.0, Array(output.joined()), 1, &(layer1CostDerivative), 1)
            
            // cost derivative.T
            var l1CostDerivTransposed = [Double](repeating: 0, count: output.count * output[0].count)
            vDSP_mtransD(layer1CostDerivative, 1, &(l1CostDerivTransposed), 1, vDSP_Length(output[0].count), vDSP_Length(output.count))
            
            // Layer1 cost = cost derivative.T * cost derivative
            var layer1Cost = [Double](repeating: 0, count: output[0].count)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(output[0].count), Int32(output[0].count), Int32(output.count), 1.0, l1CostDerivTransposed, Int32(output.count), layer1CostDerivative, Int32(output[0].count), 0.0, &(layer1Cost), Int32(output[0].count))
            
            // Get derivative of sigmoid
            let derivative = calculateDerivative(layer1, with: .sigmoid)
            
            // Error weighted derivative = cost derivative * derivative of sigmoid (element-wise multiplication)
            var errorWeightedDerivative = [Double](repeating: 0, count: layer1CostDerivative.count)
            vDSP_vmulD(layer1CostDerivative, 1, derivative, 1, &errorWeightedDerivative, 1, vDSP_Length(layer1CostDerivative.count))
            
            // Transpose layer0
            var transposed = [Double](repeating: 0, count: Array(layer0.joined()).count)
            vDSP_mtransD(Array(layer0.joined()), 1, &(transposed), 1, vDSP_Length(layer0[0].count), vDSP_Length(layer0.count))
            
            // Compute layer0.T * errorWeightedDerivative
            var multipliedFinal = [Double](repeating: 0, count: layer0[0].count * output[0].count)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(layer0[0].count), Int32(output[0].count), Int32(layer0.count), 1.0, transposed, Int32(layer0.count), errorWeightedDerivative, Int32(output[0].count), 0.0, &(multipliedFinal), Int32(output[0].count))
            
            // Adjust the weights
            var results = multipliedFinal
            cblas_daxpy(Int32(Array(synapse.joined()).count), 1.0, Array(synapse.joined()), 1, &(results), 1)
            
            // Map results back to synapse's inherent structure
            synapse = synapse.enumerated().map({ (rowIndex, row) in
                return row.enumerated().map({ (index, value) in
                    return results[index + rowIndex * row.count]
                })
            })
            
            if verbose {
                print("Iteration \(iteration), cost = \(layer1Cost[0])")
            }
            
            if iteration == iterations {
                print("Training complete!\n")
                
                print("Hidden layer:")
                print(layer1)
                
                print("\nWeights:")
                print(Array(synapse.joined()))
                
                return (layer1, Array(synapse.joined()))
            }
            
        }
        
        return nil
    }
    
}
