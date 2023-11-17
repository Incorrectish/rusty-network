use crate::network::reshape::get_ref;
use crate::network::reshape::reshape_testing_data;
use crate::network::reshape::wrap;

use self::reshape::Reshaped;
use ndarray::Array2;
use ndarray::ArrayD;
use ndarray_rand::RandomExt;
use num_traits::Float;

// use rand::distributions::{Distribution, StandardNormal};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::format;

use chrono::prelude::*;
use rayon::prelude::*;
use serde_json::{from_reader, to_writer};
use std::fs::File;
use std::io::{self, Write};

// contains all layers in the network
// pub struct Network<F: Float> {
//     num_layers: usize,
//     sizes: Vec<usize>,
//     weights: Vec<Array2<F>>,
//     biases: Vec<Array2<F>>,
//     // input_layer: Vec<F>, // input_layer: [InputNode<F>; N],
// }

#[derive(Debug, Serialize, Deserialize)]
pub struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    // input_layer: Vec<F>, // input_layer: [InputNode<F>; N],
}

// contains the values of one layer of nodes
// type Layer<F: Float> = Vec<Node<F>>;

// // Contains the values of one node
// struct Node<F: Float> {
//     activation: F,
//     weights: Vec<F>,
//     bias: F,
// }

// type InputNode<F> = F;

// impl<F: Float> Node<F> {
//     pub fn new(weights: Vec<F>, bias: F) -> Self {
//         Node {
//             activation: F::zero(),
//             weights,
//             bias,
//         }
//     }
// }

#[derive(Debug)]
pub struct NetworkError {
    message: String,
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

// struct InputNode<F: Float>(F);

fn sigmoid(z: &Array2<f64>) -> Array2<f64> {
    1.0 / (1.0 + (-z).mapv(f64::exp))
}

fn sigmoid_prime(z: &Array2<f64>) -> Array2<f64> {
    let sigmoid_z = sigmoid(z);
    &sigmoid_z * (1.0 - &sigmoid_z)
}

impl Network {
    fn feed_forward(&self, mut input_layer: Array2<f64>) -> Array2<f64> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .for_each(|(weight, bias)| input_layer = sigmoid(&(weight.dot(&input_layer) + bias)));
        input_layer
    }

    pub fn new(layer_sizes: Vec<usize>) -> Result<Self, NetworkError> {
        // let mut rng = StdRng::seed_from_u64(layer_sizes.len() as u64);
        if layer_sizes.len() < 2 {
            return Err(NetworkError {
                message: String::from("Network must contain at least an input and output layer"),
            });
        }
        let biases: Vec<Array2<f64>> = layer_sizes[1..]
            .par_iter()
            .map(|&y| {
                let normal =
                    Normal::new(0.0, 1.0).expect("Invalid parameters for the normal distribution");
                let mut rng = rand::thread_rng();
                Array2::from_shape_fn((y, 1), |_| normal.sample(&mut rng))
            })
            .collect();
        let weights: Vec<Array2<f64>> = layer_sizes
            .par_iter()
            .zip(layer_sizes.par_iter().skip(1))
            .map(|(&x, &y)| Array2::random((y, x), Normal::new(0.0, 1.0).unwrap()))
            .collect();
        Ok(Network {
            num_layers: layer_sizes.len(),
            sizes: layer_sizes,
            weights,
            biases,
        })
    }

    /*
     * training_data is the set of training data samples as expected, with each image represented as Array2, representing a matrix of all the grayscale values in the image, while the usize is the correct output
     * mini_batch_size is the size of the sample to use for stochastic gradient descent
     * epochs is the number of training cycles that our model undertakes
     * eta is η, the learning rate
     * test_data is an optional argument, and if given, the model will be evaluated on the training data after every training epoch
     * The function actually computes the gradient descent algorithm
     */
    pub fn stochastic_gradient_descent(
        &mut self,
        training_data: &mut Vec<(Array2<f64>, Array2<f64>)>,
        mini_batch_size: usize,
        epochs: usize,
        eta: f64,
        testing_data: Option<Vec<(Array2<f64>, Array2<f64>)>>,
    ) {
        let mut rng = rand::thread_rng();
        let reshaped_testing_data = if let Some(testing_data_unwrapped) = testing_data {
            Some(reshape_testing_data(testing_data_unwrapped))
        } else {
            None
        };
        for j in 0..epochs {
            training_data.shuffle(&mut rng);
            let mini_batches = (0..training_data.len())
                .step_by(mini_batch_size)
                .map(|k| &training_data[k..k + mini_batch_size])
                .collect::<Vec<_>>();
            for mini_batch in mini_batches {
                self.update_mini_batch(mini_batch, eta);
            }
            if let Some(reshaped_testing_data_unwrapped) = reshaped_testing_data.as_ref() {
                let accuracy = self.evaluate(reshaped_testing_data_unwrapped);
                if accuracy >= 0.97 {
                    let _ = cache_network(accuracy, self);
                }
                println!("Epoch {j}, accuracy is: {accuracy}",);
            } else {
                println!("Epoch {j} complete")
            }
        }
    }

    // turns a matrix of n * m columns into a vector with n * m elements in one row(using Array2, but it has only one row)
    pub fn evaluate(&self, testing_data: &Reshaped) -> f64 {
        // for every element in the testing data, run the feedforward, and get the max of output[0] then check if it is correct
        let mut correct: usize = 0;
        let mut tested = 0;
        // let max_fl = |(max_idx, max_val), (idx, current): (usize, &f64)| {
        fn max_fl<'b, 'a>(
            (max_idx, max_val): (usize, &'a f64),
            (idx, current): (usize, &'b f64),
        ) -> (usize, &'a f64)
        where
            'b: 'a,
        {
            if current.partial_cmp(max_val) == Some(std::cmp::Ordering::Greater) {
                (idx, current)
            } else {
                (max_idx, max_val)
            }
        }
        for (test, result) in get_ref(testing_data) {
            let (result_val, _max_red_value) = result
                .column(0)
                .iter()
                .enumerate()
                .reduce(max_fl)
                .expect("Outputs should always have exactly 10 elements");
            let outputs = self.feed_forward(test.clone());
            let (max_index, _max_value) = outputs
                .column(0)
                .iter()
                .enumerate()
                .reduce(max_fl)
                .expect("Outputs should always have exactly 10 elements");
            let recognized_character = max_index;
            if recognized_character == result_val {
                correct += 1;
                tested += 1
            } else {
                tested += 1;
            }
        }
        (correct as f64) / (tested as f64)
    }

    fn update_mini_batch(&mut self, mini_batch: &[(Array2<f64>, Array2<f64>)], eta: f64) {
        let mut nabla_biases: Vec<Array2<f64>> = self
            .biases
            .par_iter()
            .map(|bias| {
                let [dim1, dim2] = bias.shape() else {
                    panic!("bias must be a two dimensional array")
                };
                Array2::zeros((*dim1, *dim2))
            })
            .collect();

        let mut nabla_weights: Vec<Array2<f64>> = self
            .biases
            .par_iter()
            .map(|bias| {
                let [dim1, dim2] = bias.shape() else {
                    panic!("bias must be a two dimensional array")
                };
                Array2::zeros((*dim1, *dim2))
            })
            .collect();

        for (test, result) in mini_batch {
            let (delta_nabla_biases, delta_nabla_weights) = self.backprop(test, result);
            nabla_biases = nabla_biases
                .into_par_iter()
                .zip(delta_nabla_biases.into_par_iter())
                .map(|(nb, dnb)| nb + dnb)
                .collect();
            nabla_weights = nabla_weights
                .into_par_iter()
                .zip(delta_nabla_weights.into_par_iter())
                .map(|(nw, dnw)| nw + dnw)
                .collect();
        }

        self.weights
            .par_iter_mut()
            .zip(nabla_weights.into_par_iter())
            .for_each(|(weight, nabla_weight)| {
                *weight -= &((eta / (mini_batch.len() as f64)) * nabla_weight);
            });

        self.biases
            .par_iter_mut()
            .zip(nabla_biases.into_par_iter())
            .for_each(|(bias, nabla_bias)| {
                *bias -= &((eta / (mini_batch.len() as f64)) * nabla_bias);
            });
    }

    fn backprop(
        &self,
        test: &Array2<f64>,
        result: &Array2<f64>,
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut nabla_biases: Vec<Array2<f64>> = self
            .biases
            .par_iter()
            .map(|bias| {
                let [dim1, dim2] = bias.shape() else {
                    panic!("bias must be a two dimensional array")
                };
                Array2::zeros((*dim1, *dim2))
            })
            .collect();

        let mut nabla_weights: Vec<Array2<f64>> = self
            .biases
            .par_iter()
            .map(|bias| {
                let [dim1, dim2] = bias.shape() else {
                    panic!("bias must be a two dimensional array")
                };
                Array2::zeros((*dim1, *dim2))
            })
            .collect();
        let mut activation = test.clone();
        let mut activations = vec![activation];
        let mut zs = vec![];
        for (weight, bias) in self.weights.iter().zip(self.biases.iter()) {
            // let activations_len = activations.len();
            // activations[activations.len() - 1] houses the moved value of activation
            let z = weight.dot(&activations[activations.len() - 1]) + bias;
            activation = sigmoid(&z);
            zs.push(z);
            activations.push(activation);
        }
        let delta = self.cost_derivative(&activations[activations.len() - 1], result)
            * sigmoid_prime(&zs[zs.len() - 1]);

        let nabla_weights_len = nabla_biases.len();
        let activations_len = activations.len();
        nabla_weights[nabla_weights_len - 1] = delta.dot(&activations[activations_len - 2].t());

        let nabla_biases_len = nabla_biases.len();
        nabla_biases[nabla_biases_len - 1] = delta;
        // for l in xrange(2, self.num_layers):
        //             z = zs[-l]
        //             sp = sigmoid_prime(z)
        //             delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        //             nabla_b[-l] = delta
        //             nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        for l in 2..self.num_layers {
            let z = &zs[zs.len() - l];
            let sp = sigmoid_prime(z);
            // nabla_biases[nabla_biases_len - (l - 1] is the SAME as delta because we insert it at nabla_biases_len - l
            let delta = self.weights[self.weights.len() - (l - 1)]
                .t()
                .dot(&nabla_biases[nabla_biases_len - (l - 1)])
                * sp;
            nabla_weights[nabla_weights_len - l] =
                delta.dot(&activations[activations.len() - (l + 1)].t());
            nabla_biases[nabla_biases_len - l] = delta;
        }
        (nabla_biases, nabla_weights)
    }

    fn cost_derivative(
        &self,
        output_activations: &Array2<f64>,
        result: &Array2<f64>,
    ) -> Array2<f64> {
        output_activations - result
    }

    // def cost_derivative(self, output_activations, y):
    //     """Return the vector of partial derivatives \partial C_x /
    //     \partial a for the output activations."""
    //     return (output_activations-y)
    // fn cost_derivative(&self, output)
}

fn cache_network(accuracy: f64, network: &Network) -> std::io::Result<()> {
    // Generate a timestamp
    let timestamp = Local::now();
    let timestamp_str = timestamp.format("%Y%m%d%H%M%S").to_string();

    // Create the file name with the timestamp and accuracy
    let file_name = format!("{accuracy}_{timestamp_str}.json");

    // Create the file
    let mut file = File::create(&file_name)?;

    // Serialize the network and write it to the file
    to_writer(&mut file, network)?;

    Ok(())
}
// Simply a helper module to ensure that my data has been sufficiently shaped before I pass it into my neural network, prevents errors and so forth
pub(crate) mod reshape {
    use ndarray::Array2;
    use rayon::prelude::*;

    pub enum Reshaped {
        Reshaped(Vec<(Array2<f64>, Array2<f64>)>),
    }

    impl Reshaped {
        pub fn len(&self) -> usize {
            match self {
                Reshaped::Reshaped(e) => e.len(),
            }
        }
    }

    pub fn wrap(input: Vec<(Array2<f64>, Array2<f64>)>) -> Reshaped {
        Reshaped::Reshaped(input)
    }

    pub fn get_ref(input: &Reshaped) -> &Vec<(Array2<f64>, Array2<f64>)> {
        match input {
            Reshaped::Reshaped(e) => e,
        }
    }

    pub fn unwrap(input: Reshaped) -> Vec<(Array2<f64>, Array2<f64>)> {
        match input {
            Reshaped::Reshaped(e) => e,
        }
    }

    pub fn reshape_testing_data(testing_data: Vec<(Array2<f64>, Array2<f64>)>) -> Reshaped {
        Reshaped::Reshaped(
            testing_data
                .into_par_iter()
                .map(|(test, result)| {
                    let [num_rows, num_columns] = *test.shape() else {
                        panic!("Shape is two dimensional")
                    };
                    (
                        test.into_shape(((num_columns * num_rows), 1))
                            .expect("Reshaping an n * m matrix into an n * m length vector"),
                        result,
                    )
                })
                .collect(),
        )
    }
}

// pub(crate) fn new_layer<F: Float>(size: usize, rng: &mut StdRng) -> <F> {
//     (0..size)
//         .map(|i| {
//             Node::new(
//                 (0..size).map(|_| rng.sample(StandardNormal)).collect(),
//                 rng.sample(StandardNormal),
//             )
//         })
//         .collect()
// }

// Need a struct to represent the connections, the weights and biases of one layer to another
// The question is, should the outgoing neuron store the weights and biases for each neuron, or should the incoming neuron store it
// Better for incoming is my guess
