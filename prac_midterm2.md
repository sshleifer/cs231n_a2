**1. How does RNN fwd pass change at test time?**

**2. Receptive Field Q**
We have a 9x9 input volume
	followed by 5x5 conv with same padding
	followed by 3x3 conv with stride 2 and 	padding to make math work.

	What is the input receptive field of one neuron in the final conv layer?

**3. Params**

Compare (Receptive Field, Memory, #Params, Flops) for following architectures:
	`VGGNet`
	`AlexNet`


**4. Explain when you would use `{Group, Instance, Layer, Batch} Norm` and why?**

**4b.**

Suppose output of layer `arr = np.random.randn((N,C,H,W))`.
Write the numpy code for calculating the mean for each type of norm. Should be all oneliners.



**5. Compare Resnet50 Bottleneck Block to Standard Resnet Block**


## Answers
1.
	For seq to seq, at train time, we use true y[t-1] as last hidden state instead of actual hidden state produced by model.
	At test time, we dont know y[t-1], so we just use model.
	Some algos slowly wean off using y[t-1] I think.
	Slides Char Level LM example:
		"At test-time sample characters one at a time, feed back to model"
		At train time condition on true past characters

2.

Receptive Field Formula: receptive field size `n = k + s(m- 1)` for `m` neurons, kernel size `k` stride `s` 

for last layer: k=3, s=2, m=1, so n=k=3.

for next layer m=3, k=5, s=1, so 5+2 = 7x7.

**3. Params**

Compare (Receptive Field, Memory, #Params, Flops) for following architectures:
	`VGGNet`
	`AlexNet`

**TODO**

**4. Explain when you would use `{Group, Instance, Layer, Batch} Norm` and why?
and also numpy code**

- Batch Norm: 
	`arr.mean(axis=0,2,3)`
	good for large batches, not RNN.
	randomnness creates regularizing effect.
- Instance Norm: `arr.mean(axis=0,1)`
	basically never, maybe RNN.
- LayerNorm: `arr.mean(axis=(1,2,3))`
	small batches or RNN
- GroupNorm: `arr.reshape(N, G, C // G, H, W).mean(axis=(2,3,4))`
	conv, small batch size.

**5. Compare Resnet50 Bottleneck Block to Standard Resnet Block**

Bottleneck block, I believes, downsamples with 1x1 convs to 64 filters from 256 before running the 3x3 conv layer on the (inW, inH, 64) sized input. Then it upsamples to (inW, inH, 256) before the `+` with the block input. I think Relu happens before concat, and that there is `BatchNorm` everywhere.

The default block, which is used in `resnet18` and `resnet34` doesn't do the downsampling. It just rips 3x3 conv on the (inW, inH, 256) then upsamples?.

