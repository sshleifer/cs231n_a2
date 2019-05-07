**1. How does RNN fwd pass change at test time?**



**2. Receptive Field Q**
We have a 9x9 input volume
	followed by 5x5 conv with same padding
	followed by 3x3 conv with stride 2 and 	padding to make math work.

	What is the input volume receptive field of one neuron in the final conv layer.


**3. Params**

Compare (Receptive Field, Memory, #Params, Flops) for following architectures:
	`VGGNet`
	`AlexNet`


**4. Explain when you would use `{Group, Instance, Layer, Batch} Norm` and why?**




**4b.**

suppose output of layer `arr = np.random.randn((N,C,H,W))`
Write the numpy math for calculating the mean for each type of norm.



**5. Compare Resnet50 Bottleneck Block to Standard Resnet Block**



### Answers
1.
	For seq to seq, at train time, we use true y[t-1] as last hidden state instead of actual hidden state produced by model.
	At test time, we dont know y[t-1], so we just use model.
	Some algos slowly wean off using y[t-1] I think.
	Slides Char Level LM example:
		"At test-time sample characters one at a time, feed back to model"
		At train time condition on true past characters

2.TODO

3.TODO

4.

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
5. TODO

