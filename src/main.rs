mod feed_forward;
mod network;
mod read_image;

use crate::{network::Network, read_image::read_print_image};

use crate::network::reshape::wrap;
use image::{DynamicImage, Rgb};
use ndarray::{Array2, ArrayView2, Axis};
use serde_json::{from_reader, to_writer};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

struct MnistDataloader {
    training_images_filepath: String,
    training_labels_filepath: String,
    test_images_filepath: String,
    test_labels_filepath: String,
}

impl MnistDataloader {
    fn read_images_labels(
        &self,
        images_filepath: &str,
        labels_filepath: &str,
    ) -> Result<(Vec<Array2<f64>>, Vec<u8>), Box<dyn std::error::Error>> {
        // Read labels
        let labels = {
            let mut file = File::open(labels_filepath)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            buffer
        };

        // Read images
        let images = {
            let mut file = File::open(images_filepath)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            buffer
        };

        let labels = labels.into_iter().skip(8).collect::<Vec<_>>();
        let images = images.into_iter().skip(16).collect::<Vec<_>>();

        // Parse labels
        let labels: Vec<u8> = labels.into_iter().map(|b| b as u8).collect();

        // Parse images
        let image_size = 28 * 28;
        let image_count = images.len() / image_size;
        let images: Vec<Array2<f64>> = (0..image_count)
            .map(|i| {
                let start = i * image_size;
                let end = start + image_size;
                let image_data = &images[start..end];
                Array2::from_shape_vec(
                    (784, 1),
                    image_data
                        .into_iter()
                        .map(|&pixel| pixel as f64 / 255.0)
                        .collect::<Vec<_>>(),
                )
                .expect("Failed to create image array")
                // Array2::from_shape_vec((28, 28), image_data.to_vec())
                //     .expect("Failed to create image array")
                //     .into_shape((784, 1))
                //     .expect("Any 28 * 28 image should convert to a 784 * 1 image")
            })
            .collect();

        Ok((images, labels))
    }

    fn load_training_data(
        &self,
    ) -> Result<(Vec<Array2<f64>>, Vec<u8>), Box<dyn std::error::Error>> {
        let (x_train, y_train) = self.read_images_labels(
            &self.training_images_filepath,
            &self.training_labels_filepath,
        )?;
        Ok((x_train, y_train))
    }

    fn load_testing_data(&self) -> Result<(Vec<Array2<f64>>, Vec<u8>), Box<dyn std::error::Error>> {
        let (x_test, y_test) =
            self.read_images_labels(&self.test_images_filepath, &self.test_labels_filepath)?;
        Ok((x_test, y_test))
    }
}

fn show_images(images: &[Array2<u8>], title_texts: &[String]) {
    let cols = 5;
    let rows = (images.len() + cols - 1) / cols;

    // let mut img_buf = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            let idx = i * cols + j;
            if idx < images.len() {
                let image = &images[idx];
                let mut image_data = Vec::new();

                for pixel in image.iter() {
                    image_data.push(Rgb([*pixel, *pixel, *pixel]));
                }

                let dynamic_image =
                    DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(28, 28, |x, y| {
                        image_data[(y * 28 + x) as usize]
                    }));

                let title = &title_texts[idx];
                let title = if title.is_empty() { "Untitled" } else { title };
                let title = format!("{}_{}.png", title, idx);

                dynamic_image
                    .save_with_format(&title, image::ImageFormat::Png)
                    .expect("Failed to save image");
            }
        }
    }
}

fn main() {
    // Specify the path to the MNIST dataset files

    let mut args = std::env::args().collect::<Vec<_>>();

    let data_path = "/Users/ishan/Code/neural_net/data/";

    // Set file paths for testing images and labels
    let test_images_filepath = format!(
        "{}/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
        data_path
    );
    let test_labels_filepath = format!(
        "{}/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
        data_path
    );

    // Create MnistDataloader instance
    let mnist_dataloader = MnistDataloader {
        training_images_filepath: format!(
            "{}/train-images-idx3-ubyte/train-images-idx3-ubyte",
            data_path
        ),
        training_labels_filepath: format!(
            "{}/train-labels-idx1-ubyte/train-labels-idx1-ubyte",
            data_path
        ),
        test_images_filepath,
        test_labels_filepath,
    };

    if args.len() == 3 {
        /*
         * ./program test path
         *  */
        let path = args.remove(2);
        let network = load_network_from_file(&path).expect("Failure to read network from the file");
        let (x_test, y_test) = mnist_dataloader
            .load_testing_data()
            .expect("Error reading MNIST test data");

        let testing_data = x_test
            .into_iter()
            .zip(y_test.into_iter().map(vectorized_result))
            .collect::<Vec<_>>();
        let reshaped = wrap(testing_data);
        println!("The model's accuracy is {}", network.evaluate(&reshaped));
    } else {
        // Load testing data
        let (x_test, y_test) = mnist_dataloader
            .load_testing_data()
            .expect("Error reading MNIST test data");

        let testing_data = x_test
            .into_iter()
            .zip(y_test.into_iter().map(vectorized_result))
            .collect::<Vec<_>>();

        // let first_tenth_data: Vec<_> = testing_data[..testing_data.len() / 10].to_vec();

        let (x_train, y_train) = mnist_dataloader
            .load_training_data()
            .expect("Error reading MNIST train data");

        let mut training_data = x_train
            .into_iter()
            .zip(y_train.into_iter().map(vectorized_result))
            .collect::<Vec<_>>();

        // we can create a network
        let mut network = Network::new(vec![784, 100, 40, 16, 10]).expect("Passed in array length");
        network.stochastic_gradient_descent(
            &mut training_data,
            10,
            1000,
            3.0,
            // Some(first_tenth_data),
            Some(testing_data),
        );
    }
    // Show some random testing images
}

fn vectorized_result(result: u8) -> Array2<f64> {
    let mut result_array = Array2::zeros((10, 1));
    result_array[[result as usize, 0]] = 1.0;
    result_array
}

fn load_network_from_file(file_path: &str) -> std::io::Result<Network> {
    // Open the file
    let file = File::open(file_path)?;

    // Deserialize the network from the file
    let network: Network = from_reader(file)?;

    Ok(network)
}
