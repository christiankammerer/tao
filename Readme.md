This github repository implements the tree alternating optimization algorithm (TAO) which has been published by [Carreira-Perpiñán and Tavallali in 2018](https://papers.nips.cc/paper_files/paper/2018/hash/185c29dc24325934ee377cfda20e414c-Abstract.html). 

## Why TAO?
Regular decision trees trained by the CART algorithm make splitting decisions based on local optimality criterias. While this is very fast and typically produces well-performing trees, it does not account for global optimality. 

## How does TAO work?
TAO optimizes the trees created by CART (or any other algorithm, even random trees) and optimizes for global optimality in multiple pass-throughs. We optimize each layer of the tree, one step at a time, with respect to the global loss-function.
After changing the nodes in a layer, the samples in the tree get re-routed to where they would end-up in the new tree, so that each optimization works only on the relevant samples.

## Important vocabulary

- **Pass:** One full optimization pass-through of the tree, similar to an epoch in neural networks. 
- **Node set:** The samples which pass-through a certain node. 
- **Care set:** A subset of the node-set, for which the 0-1 loss-function actually changes if we route to a different sub-tree. The split is only optimized w.r.t. the samples in the care set.
- **Axis-aligned split:** Split decision based on single feature compared to a threshold value.
- **Oblique-split:** Split decision based on linear decision function compared to a threshold value (bias), can capture more complex patterns than an axis-aligned split.
- **Depth-batching:** Instead of re-routing samples we only re-route after every $k$ layers for faster training times, slightly impacts the models final performance, but improves training times.
- **Selective re-route:** Only samples of nodes whose parameters changed significantly, and whose routing changed are re-routed. Improves computational efficiency.
- **TAOForest**: Bagging extension of TAO to a forest based on the `RandomForest`impl

 ## Important notes
 - Nodes on the same depth-level are independent of each other, and thus can be optimized in parallel. In practice, the parallelization overhead typically exceeds the performance gain, thus use at your own discretion.
 - This implementation only implements TAO for classification, although in theory this could be extended to regression tasks as well. 
   
