# SNC - Smoothness gated, normalized convolutions for efficient Depth Completion

*Disclamer: These are just some results of research in my free time. 
Inference durations may be affected by me doing other stuff on my computer.
All expermients should be replicable with this repo, though there may have beeen minor changes to their implementation.
See the usage section at the bottom if you want to try.*


This repo is about creating a small depth completion model where every every weight and every latent can be understood intuitively by humans.
[1] [2] have shown [Normalized Convolutions](https://github.com/abdo-eldesokey/nconv-nyu)[3] to be a simple and efficient approach to unguided depth completion.
This fork improves NConvs to be smaller, faster and involve stronger inductive biases.
NConvs have a tendency to propagate information across depth discontinuities, which is prevented by smoothness gating in SNC.
Finally, the following steps are attempted with mixed results:
Reflective surfaces with missing lidar points are adressed by initially applying depth completion in an occlusion-free native lidar perspective.
Occluded points are filtered out during the projection to camera perspective.
Color and reflectance information is seamlessly integrated into the model by approximating an initial smoothness estimate for each perspective.


| Model                  | Validation Loss | MAE          | RMSE          | Delta1         | Delta2         | Delta3         |   Parameters | BatchDuration   |
|:-----------------------|:----------------|:-------------|:--------------|:---------------|:---------------|:---------------|-------------:|:----------------|
| NConvCNN [2] | 0.0404 ± 0.0009 | 406 mm ± 11  | 1582 mm ± 21  | 58.18% ± 0.92  | 79.72% ± 0.31  | 88.61% ± 0.26  |          137 | 571 ms ± 66     |
| NConvCNN+2nd_channel [2]   | 0.0394 ± 0.0005 | 398 mm ± 5   | 1580 mm ± 10  | 57.45% ± 1.15  | 81.78% ± 0.24  | 89.60% ± 0.14  |          481 | 459 ms ± 1      |
| NC                     | 0.0383 ± 0.0000 | 379 mm ± 1   | 1655 mm ± 3   | 60.40% ± 0.04  | 80.97% ± 0.02  | 89.58% ± 0.02  |           74 | 1 ms ± 0        |
| NC+filter              | 0.0323 ± 0.0011 | 337 mm ± 8   | 1462 mm ± 27  | 60.67% ± 0.17  | 81.52% ± 0.22  | 90.14% ± 0.18  |           74 | 33 ms ± 1       |
| SNC                    | 0.0252 ± 0.0004 | 284 mm ± 3   | 1332 mm ± 18  | 64.68% ± 0.48  | 85.38% ± 0.41  | 92.57% ± 0.18  |          480 | 26 ms ± 0       |
| SNC+2nd_channel        | 0.0245 ± 0.0005 | 277 mm ± 4   | 1306 mm ± 13  | 65.87% ± 0.25  | 85.99% ± 0.10  | 92.77% ± 0.11  |          932 | 24 ms ± 0       |
| SNC+2nd_module         | 0.0233 ± 0.0001 | 266 mm ± 2   | 1282 mm ± 6   | 66.31% ± 0.17  | 86.40% ± 0.09  | 93.18% ± 0.05  |    484 + 480 | 72 ms ± 1       |

*Figure 0: Comparison with the baseline I modified. \
All experiments in this readme use the same learning rate etc as the NConvCNN baseline.
The largest possible bachsize is used each time for faster training; I use a NVIDEA GeForce GTX 1660.
The full parameter settings and experiment metrics can be found in the experiment folder.
In this table, all models were trained for 20 epochs with 10k examples per batch.*\
<img src="https://render.githubusercontent.com/render/math?math=\text{Delta}_i \coloneqq \mean_{d{j,\text{gt}}>0} \big( \max ( \frac{d_j}{d_{j,\text{gt}}},\frac{d_{j,\text{gt}}}{d_j} ) < 1.01^i \big)">


# Streamlining NConvs

This repo is based on normalized convolutions (NConvs), which were first introduced by [3] and applied to depth completion by [1]. 
I recommend these works for a deeper theoretical background.
NConvs jointly propagate estimates of pixel values and respective confidences through each layer.
No further nonlinearities are required.
In depth completion, NConvs propagate estimates of depth <img src="https://render.githubusercontent.com/render/math?math=d"> and confidences  <img src="https://render.githubusercontent.com/render/math?math=\text{cd}">, which are initialized with 1 where an estimate is present.
<img src="https://render.githubusercontent.com/render/math?math=\text{cd}"> is propagated as a weighted mean, keeping the total amount of confidence approximately constant throughout all layers: \
     <img src="https://render.githubusercontent.com/render/math?math=\text{cd}_i' \coloneqq \frac{\sum_j w_{ij} \text{cd}_j}{\sum_j w_{ij}}">\
     <img src="https://render.githubusercontent.com/render/math?math=w_{ij}  \geq 0">\
Simultaneously, the same weighted mean is modified by <img src="https://render.githubusercontent.com/render/math?math=\text{cd}"> to emphasize confident estimates and propagate <img src="https://render.githubusercontent.com/render/math?math=d"> with a learned offset: \
     <img src="https://render.githubusercontent.com/render/math?math=d_{i,b}' \coloneqq \frac{\sum_j w_{ij} d_j \text{cd}_j}{\sum_j \text{cd}_j w_{ij}} %2B b">\
*Stability terms to prevent divisions by 0 are exluded in this and all following equations.*


This section modifies NConvs for speed and stronger inductive biases while reducing the parameter count as shown in figure 0.
I refer to the resulting CNN as NC.


![Trajectories NC](workspace/NC/metrics_trajectories.png) \
*Figure 1: Metric trajectories of NC over training based on 4 runs each.*

  1. (Re)moving bias parameters \
     It is mathematically equivalent to move the biases out of the convolutions and towards the end of the network, where they can be combined into a single parameter.\
  <img src="https://render.githubusercontent.com/render/math?math=d_{i,b}'' = \frac{\sum_j w_{ij}' d_{i,b}' \text{cd}_j'}{\sum_j \text{cd}_j' w_{ij}'} %2B b' = \frac{\sum_j w_{ij}' (\frac{\sum_j w_{ij} d_j \text{cd}_j}{\sum_j \text{cd}_j w_{ij}} %2B b) \text{cd}_j'}{\sum_j \text{cd}_j' w_{ij}'} %2B b' = \frac{\sum_j w_{ij}' \frac{\sum_j w_{ij} d_j \text{cd}_j}{\sum_j \text{cd}_j w_{ij}} \text{cd}_j'}{\sum_j \text{cd}_j' w_{ij}'} %2B b %2B b' \coloneqq \frac{\sum_j w_{ij}' d_j' \text{cd}_j'}{\sum_j \text{cd}_j' w_{ij}'} %2B b %2B b'  =  d_i'' %2B b %2B b'"> \
     Besides this one remaining offset, SNC has the option to specialize biases towards different confidences or locations.
    These can be either inluded as offsets or integrated via 1x1 NConvs as independend estimates and confidences.
    Based on the rationale that most input data is accurate, NC does not use any bias parameters in the name of interpretability.
    When an error is observed, I will attempt to adress it at the source.
  2. Speedup\
     Precomputing the confidence denominator \
     <img src="https://render.githubusercontent.com/render/math?math=\hat{w}_j \coloneqq \frac{w_{ij}}{\sum_k w_{ij}}"> \
     <img src="https://render.githubusercontent.com/render/math?math=\text{cd}_i' = \frac{\sum_j w_{ij} \text{cd}_j}{\sum_j w_{ij}} = {\sum_j \hat{w}_{ij} \text{cd}_j}"> \
     <img src="https://render.githubusercontent.com/render/math?math=d_i' = \frac{\sum_j w_{ij} d_j \text{cd}_j}{\sum_j \text{cd}_j w_{ij}} = \frac{\sum_j \hat{w}_{ij} d_j \text{cd}_j}{\sum_j \text{cd}_j \hat{w}_{ij}} = \frac{\sum_j \hat{w}_{ij} d_j \text{cd}_j}{c'}"> \
     and skipping the denominators until the last layer \
     <img src="https://render.githubusercontent.com/render/math?math=\text{dcd}_i' \coloneqq  d_i' \text{cd}_i'  =  \frac{\sum_j \hat{w}_{ij} d_j \text{cd}_j}{\text{cd}_i'} \text{cd}_i' = \sum_j \hat{w}_{ij} d_j \text{cd}_j =  \sum_j \hat{w}_{ij} \text{dcd}_j"> \
     removes unnecessary divisions.
     Both values can now be concatenated to a single tensor, reducing NConvs to ordinary convolutions inside the network.\
     <img src="https://render.githubusercontent.com/render/math?math=x_i' \coloneqq \begin{pmatrix}\text{dcd}_i'\\ \text{cd}_i'\end{pmatrix} = \sum_j \hat{w}_{ij} x_j">
  3. NConv based Up- and Downsampling\
     This step is mostly motivated by the aestethics of constructing NC from NConvs only via convolutions, strided convolutions and deconvolutions.
     It requires more parameters and did not affect predictions a lot in my experiments compared to tested alternatives.
     It does however allow for more modifications down the line and only uses a single operation each.
     The unpooled tensors are merged with skip connections through a weighted average (1x1 NConv) per channel instead of concatenating.
  4. Channel reduction \
     Because the entire network minus pre- and postprocessing soly consists of linear convolutions, it could take advantage of optimizations like reduced precision.
     Because it is entirely linear, it is equivalent to a single channeled version, which would also hold for some NC ensembles.
     Figure 1 shows how a single channel trains more effectively than two.
     Linearity does not hold for SNC, the version of NC which incorporates smoothness.
     For NConvs with multiple channels, I decrease the parameter count through sparable convolutions[4] .
     While these reduce the number of flops on paper, they also slow prediction speed on GPU, which is rectified by recombining their weights beforehand.
  5. Symmetric weight sharing \
     One of the most prominent dataset augmentation techniques in vision is horizontal flipping, which models like [8] also employ during inference to average with the mirrored prediction.
     NC instead enforces symmetry explicitly through weigh sharing in each layer, almost halfing the spatial parameter count instead of doubling computation cost.
     In models with multiple channels this also removes the possibility of multiple asymmetric but mirrored channels, which is not a problem here.
Figure 1 shows a more effective training with weight sharing.
  6. Using online limits\
     NConvs require all weights to be positive.
     [5] achieve this by applying a softplus calculation outside the learning procedure.
     This effectively introduces a lower bound for model weights as they are decreased according to their gradients and increased again by the softplus until a balance is reached.
     After including the function into the forward pass, the model is able to learn sparse kernels.

        | Model         | MAE | RMSE | Delta1 | Delta2 | Delta3 | wMAE | wRMSE | wDelta1 | wDelta2 | wDelta3 |
        | --            | --  | --   | --     | --     | --     | --   | --    | --      | --      |  --     |
        |offline limits |0,3012|1,3852|63,11% |83,56%  |91,36%  |0,1574 |0,6756|72,37%   |89,82%   |95,59%   |
        |online limits  |0,2901|1,3493|64,55% |84,97%  |92,24%  |0,1379 |0,5825|75,00%   |91,44%   |96,40%   |

        *Old version of SNC which still had the same layer layout as NConvCNN.*

   7. Simplifying the Layer Structure \
      Among other things, the previous step allowed the old version of SNC above to self-proon the first two expensive, full resolution layers of the inherited NConvCNN[1] layout.
      NC instead employs a single initial 5x5 convolution, followed by three alternating, weight sharing strided 4x4 and regular 3x3 Nconvs in the encoder and alternating 4x4 deconvolutions and 3x3 NConvs in the decoder.
      The initial 5x5 Nconv is used because it offers an improved initial smoothness estimate in SNC and I wanted to keep the layout consistent between models.
   8. Inference Weight Precomputation\
      Combining previous steps, the final kernels are calculated by performing a softplus operation on each weight,
      concatenating the normal and mirrored spatial weights, normalizing the weights per output channel to 1
      and potentially calculating the outer product of the spatial and channel weights.
      These steps take time.
      Since the weights are fixed during inference, the result can be stored and reused instead.
      A new model could also be initialized with the learned weights and finetuned without my parameter reducing assumptions.
   9. Loss function\
       NC adopts the smooth L1 loss from [1] but discards their confidence term.
       While this term decays over time, non-gated NC performs better without it in figure 1, even when confidence weightied metrics are considered, suggesting its regularizing influence is not needed for smaller models.
       On full NC it does improve confidence based metrics, but also slows down training.
       Similarly, the MSE loss improved the RMSE metric at the cost of the rest.

![Weights NC](images/workspace_NC_NC_run0000_ep0020_weights.png)\
*Figure 2: All preprocessed weights of NC. Spatial kernels are mirrored for convenience.*

Figure 2 shows the resulting weights.
NC learns to keep most of its confidence in the origin pixel, shifting everything down in the initial layer and back up in the final one.
One stage deeper distributes it roughly unifomly to sorrounding pixels.
The deepest two layers are specialized into propagating data across large distances in horizontal direction to fill remaining gaps.
The weights for the skip connections are almost one, masking the lower resolution where no gaps exist.
Overall there is a slight trend to propagate depths upwards rather than downwards, possibly because those regions are more sparse and conflicting measurements above an object less likely.

Based on the observation that the first and last layer mostly focus on one pixel, I test a 53 parameter model with no bias and no full resolution NConvs ("small" in figure 1).
It has a worse MAE and Delta3 metric but matches or exceeds NC everywhere else.
Based on w_prop_d, a shallower version of NC might work as well, but only by overfitting to the dataset and leaving large gaps without any predictions. 
Filling these gaps with fixed or shared weights on lower layers would half the learnable parameter count of NC with similar metrics, but ultimately be a deception.

## Smoothness Gating

Because its depth propagation is independent of depth values, NC has no way to distinguish between planes and depth discontinuities if sparse pixel locations are the same.
It is forced to learn a compromise between both situations and predict fuzzy edges.
SNC incorporates smoothness gating instead: \
<img src="https://render.githubusercontent.com/render/math?math=x_i' = \sum_j \hat{w}_{ij} s_{ij} x_j"> \
<img src="https://render.githubusercontent.com/render/math?math=s_{ij} = \prod_{k}^{i \rightarrow j} s_k"> \
<img src="https://render.githubusercontent.com/render/math?math=s_{k} \in [0,1]"> \
<img src="https://render.githubusercontent.com/render/math?math=s_{ij}"> is the combined smoothness along a path from i to j.
By predicting a smoothness <img src="https://render.githubusercontent.com/render/math?math=s_{k} = 0">, SNC can stop itself from propagating information across discontinuities and object borders.
By predicting <img src="https://render.githubusercontent.com/render/math?math=s_{k} = 1">, it is free to interpolate from one side of k to the other.
The conditions of a normalized convolution are fulfilled as long as <img src="https://render.githubusercontent.com/render/math?math=s_{ij}"> is never negative.
When merging upsampled depths and skip connection via 1x1 Nconv, the former are smoothness gated to prefer higher resolution depths near edges.

[10] calculate edges in dense but noisy depth estimates by dividing local minima by local maxima.
This fits the definition of <img src="https://render.githubusercontent.com/render/math?math=s_{k}"> and consists of a sparsity invariant operations[6].
It is adapted to NConvs by searching for weighted extrema and outputting a dedicated confidence: \
<img src="https://render.githubusercontent.com/render/math?math=j_{\text{max},k} = \text{argmax}_{j \in U(k)} \text{cd}_j d_j"> \
<img src="https://render.githubusercontent.com/render/math?math=j_{\text{min},k} = \text{argmax}_{j \in U(k)} \frac{\text{cd}_j}{d_j}"> \
<img src="https://render.githubusercontent.com/render/math?math=s_k = \big(\frac{d_{j_{\text{min},k}}}{d_{j_{\text{max},k}}}\big )^{w_\text{pow}}">  \
<img src="https://render.githubusercontent.com/render/math?math=\text{cs}_k = \text{cd}_{j_{\text{min},k}} \text{cd}_{j_{\text{max},k}}"> \
where  <img src="https://render.githubusercontent.com/render/math?math=w_\text{pow}"> represents a trainable sensitivity to depth deviations in each layer.

Like depth in NC, this version of smoothness is further interpolated with NConvs and thus remains one step ahead of the depth propagation.
However unlike depth its purpose is to represent edges, which are typically lines, not areas.
u(ndirected)SNC, the model described so far, is turned into full SNC by using 4 different smoothness channels to represent 4 possible edge directions.
For each intermediate pixel, <img src="https://render.githubusercontent.com/render/math?math=s_{ij}"> now multiplies the edge direction it has to cross to connect i and j.
<img src="https://render.githubusercontent.com/render/math?math=j_{\text{min},k}">
and <img src="https://render.githubusercontent.com/render/math?math=j_{\text{max},k}">
now refer to opposite sides per edge direction, where the lower smoothness out of both options is used.
When enforcing kernel symmetry through weight sharing, the two diagonal directions are asymmetric but mirrored versions of each other, while the rest are mirrored on the vertical axis.
The full implementation can be found [here](model/SNC_conv.py).

![Trajectories SNC](workspace/SNC/metrics_trajectories.png) \
*Figure 3: Metric trajectories of SNC over training based on 4 runs each. 
SNC+2nd_module incorporates a frozen and a trainable SNC model, resulting a parameter size of 964 if both were trained at the same time.
If you were wondering how P(SNC+2nd_channel) is < 2 P(SNC), it's because my spatial convolutions scale with kernel size and the number of input channels only, meaning the first layer has fewer parameters than you might expect.*

Figure 3 shows SNC experiments.
All versions of SNC outperform uSNC, underlining the need for directed smoothness.
Among uSNC variants the best results are gained when smoothness gating is implemented during up- and downsampling at the cost of slightly slower inference.
This is inherited by SNC.
During focused_unpool_s, 1-s_skip is used to focus pooled smoothness on known edges via gated unpooling.
During full_unpool_s, this mechanism itself is gated by 1-s_pool to only use it near edges. 
Neither version is successfull. 


![Weights SNC](images/workspace_SNC_SNC_run0000_ep0020_weights.png)\
*Figure 4: All weights of SNC after 20 epochs. \
e refers to edge-based smoothness as opposed to undirected smoothness s.*

Figure 4 shows all weights of SNC.
They appear much more random than NC weights in figure 2.
In particular, I would have expected a stronger diagonal matrix for w_dir_e to keep smoothness information within the same edge direction.
Similarly, I would have expected w_spatial_e to propagate smoothness information along an edge direction.
A look at w_prop_e and w_skip_e offers one possible explanation for this potential randomness:
The model prefers skip connections over downsampled data and values newly calculated smoothness over previous estimates in most cases, resulting in much weaker, possibly vanishing gradients for the dicarded option.
When skipping smoothness propagation entirely in figure 3, the resulting model is weaker, but still outperforms uSNC and remains competitive. 
Another possible explenation is a learned occlusion filter (see the MSNC section).

Newly calculated smoothness requires w_pow_e.
This variable is particularly high in lower resolutions, possibly because estimates are less noisy and more indicative of actual object borders.
The model is also more sensitive in the encoder where less interpolation is present.
When comparing edge directions, diagonals are more sensitive to depth deviations early on while vertical edges are more sensitive later.
Horizontal edges are the only direction where both sides of an edge are treated differently, prefering edges with a more distant bottom side in the full resolution and a more distant top side everywhere else.
I expect gaps below an object, like the underside of a car, to appear less frequent in lower resolution.

When looking at weights directly involved in depth propagation, there is a correllation with figure 2 in the first half and middle of the model.
The biggest difference is a weaker reliance on skip connections, possible because less mistakes are made at depth discontinuities.

In contrast to NC, SNC is not linear internally and benefits from multiple channels.
It could e.g. use one depth channel to look ahead, calculate better smoothness estimates and gate the slower second depth channel.
SCN2 uses two SNC modules sequentially, where the first is trained alone and frozen.
The second module uses the following inputs: \
<img src="https://render.githubusercontent.com/render/math?math=d_j'' = d_j"> \
<img src="https://render.githubusercontent.com/render/math?math=d_j'' = \text{cd}_j \min({\frac{d_j'}{d_j},1})^{w_{\text{pow_d}}}"> \
<img src="https://render.githubusercontent.com/render/math?math=s_j'' = s_j'"> \
<img src="https://render.githubusercontent.com/render/math?math=\cs_j'' = \cs_j'^{w_{\text{pow_s}}}"> \
where ' marks outputs of the first module.


![Streaming Perception](images/workspace_SNC_SNC_run0000_ep0020_streaming_perception_4149.png)\
![Streaming Perception](images/workspace_SNC_SNC2_run0000_ep0020_streaming_perception_4149.png)\
*Figure 6: Streaming Perception \
Latent space of every second layer of SNC+2nd_module, which freezes SNC as its first module.
The shown smoothness is the product over all edge directions while smoothness confidence is shown as a mean.*


[7] evaluate model outputs on both accuracy and latency by integrating updated prediction errors over time.
In a similar settting SNC would perform better than its prediction speed suggests,
because every layer of the model outputs updated predictions and confidences like in figure 6, despite only involving the depth ouput in a loss.
SNC can be thought of as a recurrent ensemble choosing different expert over time for different resolutions.
This is true for individual layers as well as groups of layers and the full model, which is used as a layer in SNC+2nd_module.
In a realtime setting with task A downstream of SNC and task B dowstream of both,
task A can rely on intermediate prediction while task B uses the final output.
This is compounded by the fact that most vision tasks use downscaled images while SNC does not.
In a real time setting, data of the previous time step could be used as an additional channel.


## Multi-Projection

![Occluded input](images/occlusion_sparse.png)
![NC](images/occlusion_NC.jpg)
![Gated Pooling](images/occlusion_NC_pool_conf_disp.jpg)
![Smoothness Gating](images/occlusion_SNC.jpg)
![Smoothness Gating](images/occlusion_SNC2.png) \
*Figure 3: Predictions with input errors.\
From left to right: Sparse input, NC, NC with gated pooling, SNC, SNC+2nd_module. Colormaps are based on depth statistics of each whole image.*

![Filtered input](images/occlusion_sparse_filtered.png)
![NC Filtered](images/occlusion_NC_filtered.png)
![NC Filtered](images/occlusion_SNC_filtered.png)
![NC Filtered](images/occlusion_MSNC_filtered.png)\
*Figure 4:  Predictions with occlusion filter approximation.\
From left to right: Filtered input used on NC, SNC, MSNC. Colormaps are based on depth statistics of each whole image.*

Figure 3 demonstates predictions suffering from input errors.
Some points are detected by the moving lidar sensor which would be occluded in the camera shot.
In the same area, the large reflective and/or transparent surface of a car on the left leads to missing non-occluded lidar points.
Consequences of these errors are especially apparent in NC predictions.
This is partially mitigated without additional parameters by gating the pooling operation with inverse depth to favour closer points ("NC+pool_disp" in figure 1).
SNC and SNC+2nd_module are expressive enough to filter dense occluded points correctly despite not being designed to do so.
Neither is able to distinguish real holes in the image from fake ones.

![Lidar Grid](images/workspace_4149_lidar_grid.png)\
*Figure 5: Native lidar grid*

To mitigate this issue I use a native lidar perspective.
The sensor shoots out light rays at 64 fixed angles 2000 times per scan whie rotating on its axis.
These scans can be accumulated as a dense 64 by 2000 grid.
Figure 4 represents a 48 by 512 crop of the forward facing section of this grid in depth and intensity representation, which is recovered from raw scan data based on [9].
Grey pixels in the depth image represent light rays which were not reflected back to the sensor with an intensity high enough to be detected.
I am using this perspective because it does not suffer from occlusion and contains dense reflectance and direction information for all points.

![Trajectories MSNC](workspace/MSNC/metrics_trajectories.png) \
*Figure 6: Metric trajectories of NC, SNC and MSNC over training.*

When projecting these points into camera perspective, I try to replicate the original pipeline of kitti including their egomotion correction.
It should be noted that any delays occured during this process are implicit to all depth completion approaches based on kitti.
Figure 6 shows results of several experiments enabled by this new pipeline:
1) Occlusion Filter\
 Because I know which points are adjacent in the dense lidar grid, I can model occlusion in camera perspective explicitly and filter occluded points.
 My implementation is based on the idea that inside the area spanned by four projected, formerly adjacend points, no depth larger than the maximum of these four points is allowed.
  While NC improves with this filter in figure 6, SNC does not; likely because its own filtering is better than my own. 
  This is supported by SNC+2nd_module in figure 3, which feeds smoothness and outlier filtering of a frozen SNC into a second SNC and better matches the cyclist's outline where my own approximation misses some points in figure 4.
  At the same time, it still predicts holes.
  A different filtering approach could be both faster and more accurate.
  KITTI generates ground truth values by accumulating 20 lidar scans and comparing their results to stereo depth estimates[6], meaning non-occluded inputs should still be present.
  For NC+gt_filter and SNC+filter, i simulate an ideal filter by removing all depth values which differ from gt depths by more than a threshold.
  These models accomplish the best metrics in this repo, suggesting occlusion has a bigger influence than precise object borders in the KITTI depth datset.
  One possible explenation for this is mentioned by the authors: Depth bleeding artifacts at object borders of their stereo based filter[6] result in differences to the lidar measurements, which are subsequently filtered out.
2) Lidar Padding\
  When projecting lidar points onto a plane it is possible to use points outside the camera field of view.
  This way any compatible model is able to use real data where it would have used padding, enabling true spatial invariance in CNNs.
  In praxis, padded regions only make up a small part of the image and rarely offer new information at the top and bottom while differences between my implementation and KITTI's introduce errors.
  NConvs in particular already treat image edges and locations with no information the same[3] and have less to gain from lidar padding. 
  In figure 6, both NC and SNC perform slightly worse with lidar padding than without.
3) Depth Completion in Lidar Space\
  While figure 5 is semi dense, depths of the car's windows and roof are missing.
  In this occlusion free perspective, most of these regions are surrounded by known points of the correct depth.
  By completing them before projecting to camera perspective, potentially with subsequent occlusion filtering, some errors might be avoided.
  The angles of missing lidar scan directions can be linearly interpolated.  
  While M(ulti projection) SNC does improve some of these errors in figure 4, it also introduces its own.
  The model has to approximate the identity fuction to avoid blurring known lidar points while simultaneously interpolating missing ones.
  This exasperates the small learning signal discussed last section while training two modules at the same time.
  In figure 6, MSNC performs worse than regular SNC, suggesting a different approach might be needed to operate in lidar space.
4) Lidar Space Superresolution\
  It is possible to upscale depths in lidar perspective using the same deconvolutions as SNC without a skip connection, potentially with an uneven kernel size. 
  I have not tried this. 
5) Reflectance Inclusion and Guided Depth Completion\
  Reflected lidar intensity can be treated as a form of color available to unguided depth completion.
  Figure 5 shows a dense intensity image of the car despite its missing lidar points.
  By estimating smoothness from reflectance, it can be fed directly into SNC.
  To keep every parameter interpretable, I use a single kernel per direction instead of a plug and play edge detector or dedicated module: \
<img src="https://render.githubusercontent.com/render/math?math=s_{i,\text{dir}} = e^{-\big |\sum_{j\in U(i)} r_i  w_{ij,\text{dir}} \big | }">  \
  This approach is insuficcient, as the the model ultimately learns to discard the new data and rely on its previous depth based estimates instead.
  The results are the same with and without reflectance, which is why this step is not plotted above.\
  The same is true for guided depth completion where kernels with 3 input channels for each color are used to genarate initial smoothness estimates.
  If a reliable edge detector and occlusion filter had been used, SNC could be simplified to NC with one additional multiplication per layer and no extra parameters.

6) Sparse Intensity\
  By mapping reflectance to the same coordinates as the respective lidar points, a sparse intensity map can be geneated in camera space.
  By concatenating it with depth and confidence, it  can be completed by NC without any additional operations or weights.
  In SNC intensity differences could be used as another source of smoothness to be multiplied with the original depth based estimates.
  Because of the additionnal memory requirements and the results above, i have not tried this.\
  Besides 3D points and reflectance, Lidar sensors are able to return information not found in the raw data from KITTI.
  Most useful to MSNC might be the ambient brightness measured by the sensor when it is not detecting is own light rays, a second dense color channel in lidar space.



# Usage
- This repo is based on the kitti depth dataset[6].
Download it and modify the two paths in params.yaml to point to the correct location or modify the beginning of KittiDepthDataloader.py (you could switch to a global config file if you want).
To use a different dataset, you would have modify or replace KittiDepthDataset.py. 
I did this in my master thesis where I recorded a new one, but have not incuded the code here.
(In case you were wondering, it did include the basic idea for uSNC, a few minor parameter reductions and a theoretical guided variant, but I mostly focused on a lot of other things. This repo started a year later.) 

- Create and activate a virtual python 3.7 environment and use 
    ```
    pip install -r requirements.txt
    ```
    the file is created from requirements.in with pip-compile and should take care of all dependencies. You can use pip-sync instead of the line above.
- Launch main with the arguments you want to use. First should probably be something like
    ```
    python main.py -mode train -params_dir workspace/SNC/SNC
    ```
    with additional options for multiple runs with the same params, evaluating all epoch and not just the last one, specific sets to use, checkpoint frequency and a starting epoch.
    If a run already exists, it will load the respective checkpoint and not recalculate individual val metrics.
    You can queue multiple experiments with a .bat file on Windows or a .sh file on Linux.
    I currently do not use tensorboard and instead print batch losses to the console. Feel free to change this, I did not bother since I plot the results myself anyway.
- You can show the results with the results mode and save the image to the respective folder.
    The params_dir argument can now also refer to a folder with multiple experiments. Epoch now represents a limit to your x axis if you want one.
    This will also print a markdown table to the console for the chosen epoch or all epochs.
- Weights are shown with the weights mode and saved in the image folder. It is compatible with all models except the baseline.
- count_parameters mode counts learnable parameters.
- The most interactive mode is predictions. It will show the inputs and predictions of exampes from the val set (default).
  Press the arrow keys to see a neighbouring image, backspace to return to the last one, enter for a random image and type a number and enter to visit a specific one; your current index will be shown in the window title and console.
  Press l to see the different possible lidar inputs and p for streaming perception (again compatible with anything but the baseline).
  Press w to write everything to the images folder (s is used by matplotlib already) and q to quit.


   




# References
[1]:  [Propagating Confidences through CNNs for Sparse Data Regression](https://arxiv.org/abs/1805.11913)\
[2]: [Uncertainty-Aware CNNs for Depth Completion: Uncertainty from Beginning to End](https://arxiv.org/abs/2006.03349)\
[3]: [Normalized and Diﬀerential Convolution](https://www.researchgate.net/publication/3557083_Normalized_and_differential_convolution)\
[4]: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861v1)\
[5]: [Confidence Propagation through CNNs for Guided Sparse Depth Regression ](https://arxiv.org/abs/1811.01791)\
[6]: [Sparsity Invariant CNNs](https://arxiv.org/abs/1708.06500)\
[7]: [Towards Streaming Perception](https://arxiv.org/abs/2005.10420)\
[8]: [Digging Into Self-Supervised Monocular Depth Estimation](https://arxiv.org/abs/1806.01260)\
[9]: [Scan-based Semantic Segmentation of LiDAR Point Clouds: An Experimental Study](https://arxiv.org/pdf/2004.11803.pdf)\
[10]: [Fast robust detection of edges in noisy depth images](https://opus.lib.uts.edu.au/bitstream/10453/100244/4/Fast%20Robust%20Detection%20of%20Edges.pdf)

