## Configuration

### Model Spec: `model.yaml`

This file contains the configuration settings for the WaveNet and diffusion models.

- **WaveNet Model Parameters**:
  - `input_channels`: Number of input channels.
  - `output_channels`: Number of output channels.
  - `residual_layers`: Number of residual layers.
  - `residual_channels`: Number of channels in residual blocks.
  - `skip_channels`: Number of channels in skip connections.
  - `diffusion_step_embed_dim_input`: Input dimension for diffusion step embedding.
  - `diffusion_step_embed_dim_hidden`: Middle dimension for diffusion step embedding.
  - `diffusion_step_embed_dim_output`: Output dimension for diffusion step embedding.
  - `s4_max_sequence_length`: Maximum sequence length for the Structured State Spaces sequence model (S4).
  - `s4_state_dim`: State dimension for the S4 model.
  - `s4_dropout`: Dropout rate for the S4 model.
  - `s4_bidirectional`: Whether to use bidirectional layers in the S4 model.
  - `s4_use_layer_norm`: Whether to use layer normalization in the S4 model.

- **Diffusion Model Parameters**:
  - `T`: Number of diffusion steps.
  - `beta_0`: Initial beta value for the diffusion process.
  - `beta_T`: Final beta value for the diffusion process.


### Training process: `training.yaml`

This file contains the configuration settings for the training process.

- `batch_size`: Batch size for training.
- `output_directory`: Output directory for checkpoints and logs.
- `ckpt_iter`: Checkpoint mode ("max" or "min").
- `iters_per_ckpt`: Checkpoint frequency (number of epochs).
- `iters_per_logging`: Log frequency (number of iterations).
- `n_iters`: Maximum number of iterations.
- `learning_rate`: Learning rate for the optimizer.
- `only_generate_missing`: Whether to generate only missing values.
- `use_model`: Model to use for training (0, 1, or 2).
- `masking`: Masking strategy for missing values (e.g., "forecast").
- `missing_k`: Number of missing values.
- `data.train_path`: Path to the training data.



### Inference process: `inference.yaml`

This file contains the configuration settings for the inference process.

- `batch_size`: Batch size for inference.
- `output_directory`: Output directory for inference results.
- `ckpt_path`: Path to the checkpoint for inference.
- `trials`: Number of replications for inference.
- `only_generate_missing`: Whether to generate only missing values.
- `use_model`: Model to use for training (0, 1, or 2).
- `masking`: Masking strategy for missing values (e.g., "forecast").
- `missing_k`: Number of missing values.
- `data.test_path`: Path to the test data.

## Usage

1. Modify the configuration files according to your requirements.
2. Run the training script with the desired configuration file (e.g., `./scripts/training_job.sh -m configs/model.yaml -t configs/training.yaml`).
3. Run the inference script with the desired configuration file (e.g., `./scripts/inference_job.sh -m configs/model.yaml -i configs/inference.yaml`).
