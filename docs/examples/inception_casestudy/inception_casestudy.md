# Inception V3 with PyTorch

Pull content from
<https://docs.amd.com/bundle/ROCm-Deep-Learning-Guide-v5.4.1/page/Deep_Learning_Training.html>.
Ignore training description.

# Deep Learning Training

Deep Learning models are designed to capture the complexity of the problem and the underlying data. These models are "deep," comprising multiple component layers. Training is finding the best parameters for each model layer to achieve a well-defined objective.

The training data consists of input features in supervised learning, similar to what the learned model is expected to see during the evaluation or inference phase. The target output is also included, which serves to teach the model. A loss metric is defined as part of training that evaluates the model's performance during the training process.

Training also includes the choice of an optimization algorithm that reduces the loss by adjusting the model's parameters. Training is an iterative process where training data is fed in, usually split into different batches, with the entirety of the training data passed during one training epoch. Training usually is run for multiple epochs.

## Training Phases
Training occurs in multiple phases for every batch of training data. Table 2 provides an explanation of the types of training phases.

||
|:--:|
| **Table 2.  Types of Training Phases**|
||

| Types of Phases |  | 
| ----------- | ----------- | 
| Forward Pass | The input features are fed into the model, whose parameters may be randomly initialized initially. Activations (outputs) of each layer are retained during this pass to help in the loss gradient computation during the backward pass. |
| Loss Computation | The output is compared against the target outputs, and the loss is computed. |
| Backward Pass | The loss is propagated backward, and the model's error gradients are computed and stored for each trainable parameter. |
| Optimization Pass | The optimization algorithm updates the model parameters using the stored error gradients. |