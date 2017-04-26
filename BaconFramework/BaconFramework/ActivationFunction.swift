//
//  ActivationFunction.swift
//  Bacon Framework
//
//  Created by Shaan Singh on 4/24/17.
//  Copyright © 2017 Blue Cocoa, Inc. All rights reserved.
//

import Foundation

public extension NeuralNetwork {
    
    public enum ActivationFunction {
        case sigmoid
    }
    
    internal func calculate(_ nodeInput: [Double], with activationFunction: ActivationFunction) -> [Double] {
        switch activationFunction {
        case .sigmoid:
            return nodeInput.map {
                return 1 / (1 + exp(-$0))
            }
        }
    }
    
    internal func calculateDerivative(_ nodeOutput: [Double], with activationFunction: ActivationFunction) -> [Double] {
        switch activationFunction {
        case .sigmoid:
            return nodeOutput.map {
                return $0 * (1 - $0)
            }
        }
    }
    
}
