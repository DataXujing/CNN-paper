## R-CNN系列 & SPP-net
------

我们本章的学习路线为:R-CNN,SPP-net,fast R-CNN,faster R-CNN,Mask R-CNN.

### 1.R-CNN

R-CNN系列论文(R-CNN,fast R-CNN,faster R-CNN,mask R-CNN)是深度学习进行目标检测的鼻祖论文，都是沿用了R-CNN的思路，我们本节内容来自《Rich feature hierarchies for accurate object detection and semantic segmentation》(2014 CVRR)的R-CNN的论文。

其实在R-CNN之前，overfeat已经是用深度学习的方法在做目标检测(关于overfeat的相关学习资料，已经放在了我的Github的repo中),但是R-CNN是第一个可以真正以工业级应用的解决方案。(这也是我们为什么介绍R-CNN系列的主要原因),可以说改变了目标检测的主要研究思路，紧随其后的系列文章都沿用R-CNN。

<div align=center>
<img src="../img/R-CNN/pic1.png" /> 
</div>
**图1：CV中的主要问题:Classify,localization(单目标),detection(多目标)**

**0.摘要：**

过去几年，在权威数据集PASCAL上，物体检测的效果已经达到一个稳定水平。效果最好的方法是融合了多种低维图像特征和高维上下文环境的复杂融合系统。在这篇论文里，我们提出了一种简单并且可扩展的检测算法，可以将mAP在VOC2012最好结果的基础上提高30%以上——达到了53.3%。我们的方法结合了两个关键的因素：

1.在候选区域上自下而上使用大型卷积神经网络(CNNs)，用以定位和分割物体。

2.当带标签的训练数据不足时，先针对辅助任务进行有监督预训练，再进行特定任务的调优，就可以产生明显的性能提升。

因为我们把region proposal（定位）和CNNs结合起来，所以该方法被称为R-CNN： Regions with CNN features。把R-CNN效果跟OverFeat比较了下（OverFeat是最近提出的在与我们相似的CNN特征下采用滑动窗口进行目标检测的一种方法，Overfeat:改进了AlexNet，并用图像缩放和滑窗方法在test数据集上测试网络；提出了一种图像定位的方法；最后通过一个卷积网络来同时进行分类，定位和检测三个计算机视觉任务，并在ILSVRC2013中获得了很好的结果。），结果发现RCNN在200类ILSVRC2013检测数据集上的性能明显优于OVerFeat。项目地址:<https://github.com/rbgirshick/rcnn>(MatLab)

**1.介绍**

特征很重要。在过去十年，各类视觉识别任务基本都建立在对SIFT[29]和HOG[7]特征的使用。但如果我们关注一下PASCAL VOC对象检测[15]这个经典的视觉识别任务，就会发现，2010-2012年进展缓慢，取得的微小进步都是通过构建一些集成系统和采用一些成功方法的变种才达到的。 【描述现状】

SIFT和HOG是块方向直方图(blockwise orientation histograms)，两篇论文已经更新在Github的repo中，一种类似大脑初级皮层V1层复杂细胞的表示方法。但我们知道识别发生在多个下游阶段，（我们是先看到了一些特征，然后才意识到这是什么东西）也就是说对于视觉识别来说，更有价值的信息，是层次化的，多个阶段的特征。 【关于SIFT&HOG】

"神经认知机",一种受生物学启发用于模式识别的层次化、移动不变性模型，算是这方面最早的尝试,但神经认知机缺乏监督学习算法。Lecun等人的工作表明基于反向传播的随机梯度下降(SGD)对训练卷积神经网络（CNNs）非常有效，CNNs被认为是继承自neocognitron的一类模型。 【神经认知机】

CNNs在1990年代被广泛使用，但随即便因为SVM的崛起而淡出研究主流。2012年，Krizhevsky等人在ImageNet大规模视觉识别挑战赛(ILSVRC)上的出色表现重新燃起了世界对CNNs的兴趣（AlexNet）。他们的成功在于在120万的标签图像上使用了一个大型的CNN，并且对LeCUN的CNN进行了一些改造（比如ReLU和Dropout Regularization）。 【CNN的崛起】

这个ImangeNet的结果的重要性在ILSVRC2012 workshop上得到了热烈的讨论。提炼出来的核心问题是：ImageNet上的CNN分类结果在何种程度上能够应用到PASCAL VOC挑战的物体检测任务上？【CNN何时使用到目标检测】

我们通过连接图像分类和目标检测，回答了这个问题。本论文是第一个说明在PASCAL VOC的物体检测任务上CNN比基于简单类HOG特征的系统有大幅的性能提升。我们主要关注了两个问题：使用深度网络定位物体和在小规模的标注数据集上进行大型网络模型的训练。 【R-CNN解决的问题】

与图像分类不同的是检测需要定位一个图像内的许多物体。一个方法是将框定位看做是回归问题。但Szegedy等人的工作说明这种策略并不work（在VOC2007上他们的mAP是30.5%，而我们的达到了58.5%）。【将定位问题单纯作为回归解决效果并不好】

另一个可替代的方法是使用【滑动窗口探测器】，通过这种方法使用CNNs至少已经有20年的时间了，通常用于一些特定的种类如人脸，行人等。为了获得较高的空间分辨率，这些CNNs都采用了两个卷积层和两个池化层。我们本来也考虑过使用滑动窗口的方法，但是由于网络层次更深，输入图片有非常大的感受野（195×195）and 步长（32×32），这使得采用滑动窗口的方法充满挑战。【感受野大，滑动窗口出来的边界不准确】

我们是通过操作”recognition using regions”[21]范式，解决了CNN的定位问题。
+ 测试时，对这每张图片，产生了接近2000个与类别无关的region proposal,
+ 对每个CNN抽取了一个固定长度的特征向量，
+ 然后借助专门针对特定类别数据的线性SVM对每个区域进行分类。

我们不考虑region的大小，使用放射图像变形的方法来对每个不同形状的region proposal产生一个固定长度的作为CNN输入的特征向量（也就是把不同大小的proposal放到同一个大小）。图2展示了我们方法的全貌并突出展示了一些实验结果。由于我们结合了Region proposals[21]和CNNs，所以起名R-CNN：Regions with CNN features。【R-CNN的由来】

<div align=center>
<img src="../img/R-CNN/pic2.png" /> 
</div>
**图2：R-CNN目标检测系统过程. （1）获取一张输入图片，（2）产生2000个与类别无关的region proposal，（3）用大型的卷积计算备选区域的特征，（4）使用线性SVM对每一个定位进行分类**

检测中面对的第二个挑战是标签数据太少，现在可获得的数据远远不够用来训练一个大型卷积网络。传统方法多是采用无监督与训练，再进行有监督调优。本文的第二个核心贡献是在辅助数据集（ILSVRC）上进行有监督预训练，再在小数据集上针对特定问题进行调优。这是在训练数据稀少的情况下一个非常有效的训练大型卷积神经网络的方法。我们的实验中，针对检测的调优将mAP提高了8个百分点。调优后，我们的系统在VOC2010上达到了54%的mAP，远远超过高度优化的基于HOG的可变性部件模型（deformable part model，DPM）
【DPM:多尺度形变部件模型，连续获得07-09的检测冠军，2010年其作者Felzenszwalb Pedro被VOC授予”终身成就奖”。DPM把物体看成了多个组成的部件（比如人脸的鼻子、嘴巴等），用部件间的关系来描述物体，这个特性非常符合自然界很多物体的非刚体特征。DPM可以看做是HOG+SVM的扩展，很好的继承了两者的优点，在人脸检测、行人检测等任务上取得了不错的效果，但是DPM相对复杂，检测速度也较慢，从而也出现了很多改进的方法。】【挑战2及解决办法】

R-CNN计算高效： 原因都是小型矩阵的乘积，特征在不同类别间共享；HOG-like特征的一个优点是简单性：能够很容易明白提取到的特征是什么（可视化出来）。介绍技术细节之前，我们提醒大家由于R-CNN是在推荐区域上进行操作，所以可以很自然地扩展到语义分割任务上。只要很小的改动，我们就在PASCAL VOC语义分割任务上达到了很有竞争力的结果，在VOC2011测试集上平均语义分割精度达到了47.9%。【R-CNN的其他应用】

**2.用R-CNN做目标检测**

我们的物体检测系统有三个模块构成。

+ 第一个，产生类别无关的region proposal。这些推荐定义了一个候选检测区域的集合；
+ 第二个是一个大型卷积神经网络，用于从每个区域抽取特定大小的特征向量；
+ 第三个是一个指定类别的线性SVM。

本部分，将展示每个模块的设计，并介绍他们的测试阶段的用法，以及参数是如何学习的细节，最后给出在PASCAL VOC 2010-12和ILSVRC2013上的检测结果。

**2.1模块设计**

【region proposal：区域推荐】 近来有很多研究都提出了产生类别无关区域推荐的方法比如: objectness（物体性）[1]，selective search（选择性搜索）[39]，category-independent object proposals(类别无关物体推荐)[14]，constrained parametric min-cuts（受限参最小剪切, CPMC)[5]，multi-scal combinatorial grouping(多尺度联合分组)[3]，以及Ciresan[6]等人的方法,将CNN用在规律空间块裁剪上以检测有丝分裂细胞，也算是一种特殊的区域推荐类型。由于R-CNN对特定区域推荐算法是不关心的，所以我们采用了选择性搜索[39]以方便和前面的工作进行可控的比较。[region proposal方法，建议自行学习]

【Feature extraction: 特征提取】我们使用Krizhevsky等人所描述的CNN的一个Caffe实现版本[24]对每个推荐区域抽取一个4096维度的特征向量把一个输入为277*277大小的图片，通过五个卷积层和两个全连接层进行前向传播,最终得到一个4096-D的特征向量。读者可以参考AlexNet获得更多的网络架构细节。为了计算region proposal的特征，我们首先要对图像进行转换，使得它符合CNNC的输入（架构中的CNN只能接受固定大小：277*277）这个变换有很多办法，我们使用了最简单的一种。无论候选区域是什么尺寸和宽高比，我们都把候选框变形成想要的尺寸,。具体的，变形之前，我们先在候选框周围加上16的padding,再进行**各向异性缩放**。这种形变使得mAp提高了3到5个百分点。在补充材料中，作者对比了各向异性和各向同性缩放缩放方法。

**2.2测试阶段的物体检测**

测试阶段，在测试图像上使用selective search抽取2000个推荐区域（实验中，我们使用了选择性搜索的快速模式）（关于selective search我们在下文中会详细讲解）然后变形每一个推荐区域，再通过CNN前向传播计算出特征。然后我们使用对每个类别训练出的SVM给整个特征向量中的每个类别单独打分。【对每一个框使用每个类别的SVM进行打分】然后给出一张图像中所有的打分区域，然后使用NMS【贪婪非最大化抑制算法】（每个类别是独立进行的），拒绝掉一些和高分区域的IOU大于阈值的候选框。


【**运行时间的分析**】两个特性让检测变得很高效。首先，所有的CNN参数都是跨类别共享的。（参数共享）其次，通过CNN计算的特征向量相比其他通用方法（比如spatial pyramids with bag-of-visual-word encodings）维度是很低的。（低维特征）这种共享的结果就是计算推荐区域特征的耗时可以分摊到所有类别的头上（GPU：每张图13s，CPU：每张图53s）。

唯一的和具体类别有关的计算是特征向量和SVM权重和点积，以及NMS实践中，所有的点积都可以批量化成一个单独矩阵间运算。特征矩阵的典型大小是2000×4096，SVM权重的矩阵是4096xN，其中N是类别的数量。

分析表明R-CNN可以扩展到上千个类别，而不需要借用近似技术（如hashing）。及时有10万个类别，矩阵乘法在现代多核CPU上只需要10s而已。但这种高效不仅仅是因为使用了区域推荐和共享特征。

**2.3训练**

【**有监督的预训练 **】我们在大型辅助训练集ILSVRC2012分类数据集（没有约束框数据）上预训练了CNN。预训练采用了Caffe的CNN库。总体来说，我们的CNN十分接近krizhevsky等人的网络的性能，在ILSVRC2012分类验证集在top-1错误率上比他们高2.2%。差异主要来自于训练过程的简化。

【**特定领域的参数调优 **】为了让我们的CNN适应新的任务（即检测任务）和新的领域（变形后的推荐窗口）。我们只使用变形后的推荐区域对CNN参数进行SGD训练。我们替换掉了ImageNet专用的1000-way分类层，换成了一个随机初始化的21-way分类层，（其中20是VOC的类别数，1代表背景）而卷积部分都没有改变，我们对待所有的推荐区域，如果其和真实标注的框的IoU>= 0.5就认为是正例，否则就是负例，SGD开始的learning_rate为0.001（是初始化预训练时的十分之一），这使得调优得以有效进行而不会破坏初始化的成果。每轮SGD迭代，我们统一使用32个正例窗口（跨所有类别）和96个背景窗口，即每个mini-batch的大小是128。另外我们倾向于采样正例窗口，因为和背景相比他们很稀少。

【**目标种类分类器**】思考一下检测汽车的二分类器。很显然，一个图像区域紧紧包裹着一辆汽车应该就是正例。同样的，没有汽车的就是背景区域，也就是负例。较为不明确的是怎样标注哪些只和汽车部分重叠的区域。我们使用IoU重叠阈值来解决这个问题，低于这个阈值的就是负例。这个阈值我们选择了0.3，是在验证集上基于{0, 0.1, … 0.5}通过网格搜索得到的。我们发现认真选择这个阈值很重要。如果设置为0.5，可以提升mAP5个点，设置为0，就会降低4个点。正例就严格的是标注的框

一旦特征提取出来，并应用标签数据，我们优化了每个类的线性SVM。由于训练数据太大，难以装进内存，我们选择了标准的hard negative mining method【难负例挖掘算法，用途就是负例数据不平衡，而负例分赛代表性又不够的问题，hard negative就是每次把那些顽固的棘手的错误，在送回去训练，训练到你的成绩不在提升为止，这个过程叫做hard negative mining】

高难负例挖掘算法收敛很快，实践中只要在所有图像上经过一轮训练，mAP就可以基本停止增加了。 附录B中，讨论了，为什么在fine-tunning和SVM训练这两个阶段，我们定义得正负样例是不同的。【fine-tunning阶段是由于CNN对小样本容易过拟合，需要大量训练数据，故对IoU限制宽松： IoU>0.5的建议框为正样本，否则为负样本； SVM这种机制是由于其适用于小样本训练，故对样本IoU限制严格：Ground Truth为正样本，与Ground Truth相交IoU＜0.3的建议框为负样本。】

我们也会讨论为什么训练一个分类器是必要的，而不只是简单地使用来自调优后的CNN的最终fc8层的输出。【为什么单独训练了一个SVM而不是直接用softmax，作者提到，刚开始时只是用了ImageNet预训练了CNN，并用提取的特征训练了SVMs，此时用正负样本标记方法就是前面所述的0.3,后来刚开始使用fine-tuning时，也使用了这个方法，但是发现结果很差，于是通过调试选择了0.5这个方法，作者认为这样可以加大样本的数量，从而避免过拟合。然而，IoU大于0.5就作为正样本会导致网络定位准确度的下降，故使用了SVM来做检测，全部使用ground-truth样本作为正样本，且使用非正样本的，且IoU大于0.3的“hard negatives”，提高了定位的准确度】

**2.4在PASCAL VOC 2010-12上的结果**

在数据集： PASCAL 2010-12:

<div align=center>
<img src="../img/R-CNN/pic3.png" /> 
</div>
**原paper的Table1**

在数据集ILSVR2013数据集上得到了相似的结果

**3.可视化、消融、模型的错误**

**3.1可视化学习到的特征**（如何展示CNN每层学到的东西，了解）

直接可视化第一层filters非常容易理解，它们主要捕获方向性边缘和对比色。难以理解的是后面的层。Zeiler and Fergus提出了一种可视化的很棒的反卷积办法。我们则使用了一种简单的非参数化方法，直接展示网络学到的东西。这个想法是单一输出网络中一个特定单元（特征），然后把它当做一个正确类别的物体检测器来使用。 
方法是这样的，先计算所有抽取出来的推荐区域（大约1000万），计算每个区域所导致的对应单元的激活值，然后按激活值对这些区域进行排序，然后进行最大值抑制，最后展示分值最高的若干个区域。这个方法让被选中的单元在遇到他想激活的输入时“自己说话”。我们避免平均化是为了看到不同的视觉模式和深入观察单元计算出来的不变性。 
我们可视化了第五层的池化层pool5，是卷积网络的最后一层，feature_map(卷积核和特征数的总称)的大小是6 x 6 x 256 = 9216维。忽略边界效应，每个pool5单元拥有195×195的感受野，输入是227×227。pool5中间的单元，几乎是一个全局视角，而边缘的单元有较小的带裁切的支持。 
图4的每一行显示了对于一个pool5单元的最高16个激活区域情况，这个实例来自于VOC 2007上我们调优的CNN，这里只展示了256个单元中的6个（附录D包含更多）。我们看看这些单元都学到了什么。第二行，有一个单元看到狗和斑点的时候就会激活，第三行对应红斑点，还有人脸，当然还有一些抽象的模式，比如文字和带窗户的三角结构。这个网络似乎学到了一些类别调优相关的特征，这些特征都是形状、纹理、颜色和材质特性的分布式表示。而后续的fc6层则对这些丰富的特征建立大量的组合来表达各种不同的事物。

**3.2消融研究（Ablation studies）**

ablation study 就是为了研究模型中所提出的一些结构是否有效而设计的实验。如你提出了某某结构，但是要想确定这个结构是否有利于最终的效果，那就要将去掉该结构的网络与加上该结构的网络所得到的结果进行对比，这就是ablation study。也就是（控制变量法）

【**没有调优的各层性能**】

为了理解哪一层对于检测的性能十分重要，我们分析了CNN最后三层的每一层在VOC2007上面的结果。Pool5在3.1中做过剪短的表述。最后两层下面来总结一下。 

fc6是一个与pool5连接的全连接层。为了计算特征，它和pool5的feature map（reshape成一个9216维度的向量）做了一个4096×9216的矩阵乘法，并添加了一个bias向量。中间的向量是逐个组件的半波整流（component-wise half-wave rectified）【Relu（x<- max(0,x)）】 

fc7是网络的最后一层。跟fc6之间通过一个4096×4096的矩阵相乘。也是添加了bias向量和应用了ReLU。 

我们先来看看没有调优的CNN在PASCAL上的表现，没有调优是指所有的CNN参数就是在ILSVRC2012上训练后的状态。分析每一层的性能显示来自fc7的特征泛化能力不如fc6的特征。这意味29%的CNN参数，也就是1680万的参数可以移除掉，而且不影响mAP。更多的惊喜是即使同时移除fc6和fc7，仅仅使用pool5的特征，只使用CNN参数的6%也能有非常好的结果。可见CNN的主要表达力来自于卷积层，而不是全连接层。这个发现提醒我们也许可以在计算一个任意尺寸的图片的稠密特征图（dense feature map）时使仅仅使用CNN的卷积层。这种表示可以直接在pool5的特征上进行滑动窗口检测的实验。 

【**调优后的各层性能**】

我们来看看调优后在VOC2007上的结果表现。提升非常明显，mAP提升了8个百分点，达到了54.2%。fc6和fc7的提升明显优于pool5，这说明pool5从ImageNet学习的特征通用性很强，在它之上层的大部分提升主要是在学习领域相关的非线性分类器。

【**对比其他特征学习方法**】

R-CNN是最好的，我们的mAP要多大约20个百分点，61%的相对提升。

**3.3网络结构**
**3.4 检测错误分析**

两个直接省略！！！

**3.5Bounding-box回归**

基于错误分析，我们使用了一种简单的方法减小定位误差。受到DPM[17]中使用的约束框回归训练启发，我们训练了一个线性回归模型在给定一个选择区域的pool5特征时去预测一个新的检测窗口。详细的细节参考附录C。表1、表2和图4的结果说明这个简单的方法，修复了大量的错位检测，提升了3-4个百分点。

关于BoundingBox-Regression参考下文



**4.结论**

最近几年，物体检测陷入停滞，表现最好的检测系统是复杂的将多低层级的图像特征与高层级的物体检测器环境与场景识别相结合。本文提出了一种简单并且可扩展的物体检测方法，达到了VOC 2012数据集相对之前最好性能的30%的提升。 
我们取得这个性能主要通过两个方面：第一是应用了自底向上的候选框训练的高容量的卷积神经网络进行定位和分割物体。另外一个是使用在标签数据匮乏的情况下训练大规模神经网络的一个方法。我们展示了在有监督的情况下使用丰富的数据集（图片分类）预训练一个网络作为辅助性的工作是很有效的，然后采用稀少数据（检测）去调优定位任务的网络。我们猜测“有监督的预训练+特定领域的调优”这一范式对于数据稀少的视觉问题是很有效的。 
最后,我们注意到能得到这些结果，将计算机视觉中经典的工具和深度学习(自底向上的区域候选框和卷积神经网络）组合是非常重要的。而不是违背科学探索的主线，这两个部分是自然而且必然的结合。




------

###  2.PASCAL  & ILSVRC
 
> Pattern Analysis, Statical Modeling and Computational Learning  Visual Object Classes

[主页]<http://host.robots.ox.ac.uk/pascal/VOC/>

+ Provides standardised image data sets for object class recognition
+ Provides a common set of tools for accessing the data sets and annotations
+ Enables evaluation and comparison of different methods 
+ Ran challenges evaluating performance on object class recognition (from 2005-2012, now finished)

提供了2005-2012年的数据集，数据集的[参考格式]<https://www.cnblogs.com/whlook/p/7220105.html>

<div align=center>
<img src="../img/R-CNN/pic_voc.png" /> 
</div>

+ Large Scale Visual Recognition Challenge (ILSVRC)

Stanford Vison Lab

ImageNet比赛

[主页]<http://www.image-net.org/challenges/LSVRC/>

The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) evaluates algorithms for object detection and image classification at large scale. One high level motivation is to allow researchers to compare progress in detection across a wider variety of objects -- taking advantage of the quite expensive labeling effort. Another motivation is to measure the progress of computer vision for large scale image indexing for retrieval and annotation.

------

###   3. 目标检测中用到的一些评价指标

模型的好坏是相对的，什么样的模型好不仅取决于数据和算法，还取决于任务需求，因此选取一个合理的模型评价指标非常有必要。

+ IOU 

IOU是由预测的包围盒与地面真相包围盒的重叠区域（交集），除以他们之间的联合区域（并集），gt代表针织框

<div align=center>
<img src="../img/R-CNN/pic_IOU.png" /> 
</div>

<div align=center>
<img src="../img/R-CNN/pic_IOU1.png" /> 
</div>

+ Precision，Recall,......

一般模型常用的错误率(Error)和精度(accuracy)就能解决(一般的机器学习任务),精度和错误率虽然常用，但不能满足所有需求

<div align=center>
<img src="../img/R-CNN/pic_p.png" /> 
</div>

<div align=center>
<img src="../img/R-CNN/pic_p2.png" /> 
</div>

其他常用的：
ROC（AUC为ROC曲线下的面积)，P-R曲线，lift曲线，若当值，K-S值（二分类用的多一些），混淆矩阵，F1(F-score, F-Measure $\aplpha=1$ )

基于自己的学习任务，同时也可以修改(比如加一些惩罚)或自定义其他的评价指标。


+ AP & mAP

P: precision

AP: average precision,每一类别P值的平均值

mAP: mean average precision,对所有类别的AP取平均

目标检测中的模型的分类和定位都需要进行评估，每个图像都可能具有不同类别的不同目标。

** 计算mAP的过程**

[Ground Truth的定义]
对于任何算法，度量总是与数据的真实值(Ground Truth)进行比较。我们只知道训练、验证和测试数据集的Ground Truth信息。对于物体检测问题，Ground Truth包括图像，图像中的目标的类别以及图像中每个目标的边界框。 

下图给出了一个真实的图像(JPG/PNG)和其他标注信息作为文本(边界框坐标(X, Y, 宽度和高度)和类)，其中上图的红色框和文本标签仅仅是为了更好的理解，手工标注可视化显示。 

<div align=center>
<img src="../img/R-CNN/pic_mpa1.png" /> 
</div>
**标注图像：Ground Truth**

对于上面的例子，我们在模型在训练中得到了下图所示的目标边界框和3组数字定义的ground truth(假设这个图像是1000*800px，所有这些坐标都是构建在像素层面上的) 

<div align=center>
<img src="../img/R-CNN/pic_map2.png" /> 
</div>
**模型需要预测的：关于详细的一些模型预测label的设定建议学习吴恩达的deeplearning.ai关于卷积网络学习的网课**

开始计算mAP的步骤：

+ 1.假设原始图像和真实的标注信息(ground truth)如上所示，训练和验证数据以相同的方式都进行了标注。该模型将返回大量的预测，但是在这些模型中，大多数都具有非常低的置信度分数，因此我们只考虑高于某个置信度分数的预测信息。我们通过我们的模型运行原始图像，在置信阈值确定之后，下面是目标检测算法返回的带有边框的图像区域(bounding boxes)。 

<div align=center>
<img src="../img/R-CNN/pic_map3.png" /> 
</div>
**预测结果**

但是怎样在实际中量化这些检测区域的正确性呢？ 
首先我们需要知道每个检测的正确性。测量一个给定的边框的正确性的度量标准是loU-交幷比(检测评价函数)，这是一个非常简单的视觉量。 
下面给出loU的简单的解释。(我们在第一部分已经给出定义)

+  2.IoU计算

loU(交并比)是模型所预测的检测框和真实(ground truth)的检测框的交集和并集之间的比例。这个数据也被称为Jaccard指数。为了得到交集和并集值，我们首先将预测框叠加在ground truth实际框上面，如下图所示： 

<div align=center>
<img src="../img/R-CNN/pic_map4.png" /> 
</div>

现在对于每个类，预测框和真实框重叠的区域就是交集区域，预测框和真实框的总面积区域就是并集框。 
在上面的目标马的交集和联合看起来是这样的：

<div align=center>
<img src="../img/R-CNN/pic_map5.png" /> 
</div>

交集包括重叠区域(青色区域), 并集包括橙色和青色区域


+  3.识别正确的检测和计算精度 

我们使用loU看检测是否正确需要设定一个阈值，最常用的阈值是0.5，即如果loU>0.5，则认为是真实的检测(true detection)，否则认为是错误的检测(false detection)。我们现在计算模型得到的每个检测框的loU值。用计算出的loU值与设定的loU阈值(例如0.5)比较，就可以计算出每个图像中每个类的正确检测次数(A)。对于每个图像，我们都有ground truth的数据(即知道每个图像的真实目标信息),因此也知道了该图像中给定类别的实际目标(B)的数量。因此我们可以使用这个公式来计算该类模型的精度(A/B) 

<div align=center>
<img src="../img/R-CNN/pic_map6.png" /> 
</div>

即给定一张图像的类别C的Precision=图像正确预测(True Positives)的数量除以在该图像上这一类的总的目标数量。 

假如现在有一个给定的类，验证集中有100个图像，并且我们知道每个图像都有其中的所有类(基于ground truth)。所以我们可以得到100个精度值，计算这100个精度值的平均值，得到的就是该类的平均精度。

<div align=center>
<img src="../img/R-CNN/pic_map7.png" /> 
</div>

即一个C类的平均精度=在验证集上所有的图像对于类C的精度值的和/有类C这个目标的所有图像的数量。

+  4.计算最终mAP

现在假如我们整个集合中有20个类，对于每个类别，我们都先计算loU，接下来计算精度,然后计算平均精度。所有我们现在有20个不同的平均精度值。使用这些平均精度值，我们可以轻松的判断任何给定类别的模型的性能。 

但是问题是使用20个不同的平均精度使我们难以度量整个模型，所以我们可以选用一个单一的数字来表示一个模型的表现(一个度量来统一它们),我们可以取所有类的平均精度值的平均值，即mAP(均值平均精度)。

<div align=center>
<img src="../img/R-CNN/pic_map8.png" /> 
</div>

MAP=所有类别的平均精度求和除以所有类别 

使用MAP值时我们需要满足一下条件： 
(1) MAP总是在固定的数据集上计算 
(2)它不是量化模型输出的绝对度量，但是是一个比较好的相对度量。当我们在流行的公共数据集上计算这个度量时，这个度量可以很容易的用来比较不同目标检测方法 
(3)根据训练中类的分布情况，平均精度值可能会因为某些类别(具有良好的训练数据)非常高(对于具有较少或较差数据的类别)而言非常低。所以我们需要mAP可能是适中的，但是模型可能对于某些类非常好，对于某些类非常不好。因此建议在分析模型结果的同时查看个各类的平均精度(AP)，这些值也可以作为我们是不是需要添加更多训练样本的一个依据。


------

### 4.各向异性，各向同性缩放

R-CNN的论文中提到了各向同性，各向异性缩放的概念，这里做一个详细解释：

当我们输入一张图片时，我们要搜索出所有可能是物体的区域，R-CNN采用的就是Selective Search方法，通过这个算法我们搜索出2000个候选框。然后从R-CNN的总流程图中可以看到，搜出的候选框是矩形的，而且是大小各不相同。然而CNN对输入图片的大小是有固定的，如果把搜索到的矩形选框不做处理，就扔进CNN中，肯定不行。因此对于每个输入的候选框都需要缩放到固定的大小。

下面我们讲解要怎么进行缩放处理，为了简单起见我们假设下一阶段CNN所需要的输入图片大小是个正方形图片227*227。因为我们经过selective search 得到的是矩形框，paper试验了两种不同的处理方法：

**各向异性缩放：**
这种方法很简单，就是不管图片的长宽比例，管它是否扭曲，进行缩放就是了，全部缩放到CNN输入的大小227*227，如下图(D)所示；

**各向同性缩放：**
因为图片扭曲后，估计会对后续CNN的训练精度有影响，于是作者也测试了“各向同性缩放”方案。有两种办法：

+ 先扩充后裁剪

直接在原始图片中，把bounding box的边界进行扩展延伸成正方形，然后再进行裁剪；如果已经延伸到了原始图片的外边界，那么就用bounding box中的颜色均值填充；如下图(B)所示;

+ 先裁剪后扩充

先把bounding box图片裁剪出来，然后用固定的背景颜色填充成正方形图片(背景颜色也是采用bounding box的像素颜色均值),如下图(C)所示;

对于上面的异性、同性缩放，文献还有个padding处理，上面的示意图中第1、3行就是结合了padding=0, 第2、4行结果图采用padding=16的结果。经过最后的试验，作者发现采用各向异性缩放、padding=16的精度最高。（也就是最后一个图） 

<div align=center>
<img src="../img/R-CNN/pic_featureext.png" /> 
</div>


------

### 5.NMS:非极大值抑制

先假设有n个（假设有6个）候选框，根据分类器类别分类概率做排序，从小到大分别属于车辆的概率分别为A<=B<=C<=D<=E<=F。

（1）从最大概率的矩形框开始（F），分别判断A-E与F的IOU是否大于某个设定的阈值

（2）假设B,D与F的IOU超过F,那就扔掉B,D，并标记第一个矩形框F,是我们保留下来的

（3）从剩余矩形框A,C.E中选择概率最大的E，然后判断E与A,C的IOU(重叠度），重叠度大于一定的阈值，那么就扔掉，标记E是我们保留下来的第2个矩形框

（4）一直重复这个过程，找到所有被曾经保留下来的矩形框。

> 为什么需要NMS?

在测试过程完成到第4步之后[section7中的步骤]，获得2000×20维矩阵表示每个建议框是某个物体类别的得分情况，此时会遇到下图所示情况，同一个车辆目标会被多个建议框包围，这时需要非极大值抑制操作去除得分较低的候选框以减少重叠框。

<div align=center>
<img src="../img/R-CNN/pic_NMS1.png" /> 
</div>

------

### 6.边框回归：BoundingBox-Regression(BBR)

>首先考虑R-CNN中为什么要做BBR?

Bounding Boxregression是 RCNN中使用的边框回归方法，在RCNN的论文中，作者指出：主要的错误是源于mislocalization。为了解决这个问题，作者使用了bounding box regression。这个方法使得mAp提高了3到4个点。 

>BBR的输入 是什么？

<div align=center>
<img src="../img/R-CNN/pic_BBR1.png" /> 
</div>

对于预测框P,我们有一个ground truth是G：当0.1< IoU < 0.5时出现重复，这种情况属于作者说的poor localiazation, 但注意：我们使用的并不是这样的框进行BBR(网上很多地方都在这里出现了误导),作者是用iou>0.6的进行BBR,也就是IOU<0.6的Bounding Box会直接被舍弃，不进行BBR。这样做是为了满足线性转换的条件。否则会导致训练的回归模型不 work.
>（当 P跟 G 离得较远，就是复杂的非线性问题了，此时用线性回归建模显然不合理。)

至于为什么当IoU较大的时候，我们才认为是线性变化，我找到一个觉得解释的比较清楚的，截图在下面： 

<div align=center>
<img src="../img/R-CNN/pic_BBR2.png" /> 
</div>

线性回归就是给定输入的特征向量 X, 学习一组参数 W, 使得经过线性回归后的值跟真实值 Y(Ground Truth)非常接近. 即Y≈WX 。

边框回归的目的既是：给定(Px,Py,Pw,Ph)(Px,Py,Pw,Ph)寻找一种映射ff， 使得f(Px,Py,Pw,Ph)=(Gx^,Gy^,Gw^,Gh^)f(Px,Py,Pw,Ph)=(Gx^,Gy^,Gw^,Gh^) 并且(Gx^,Gy^,Gw^,Gh^)≈(Gx,Gy,Gw,Gh)



例如上图：我们现在要讲P框进行bbr,gt为G框，那么我们希望经过变换之后，P框能接近G框（比如，上图的G^框）。现在进行变换,过程如下： 

我们用一个四维向量（x,y,w,h）来表示一个窗口，其中x,y,w,h分别代表框的中心点的坐标以及宽，高。我们要从P得到G^，需要经过平移和缩放。 

<div align=center>
<img src="../img/R-CNN/pic_BBR3.png" /> 
</div>

其实这并不是真正的BBR，因为我们只是把P映射回G^,得到一个一般变换的式子，那为什么不映射回最优答案G呢？于是，P映射回G而不是G^，那我们就能得到最优变换（这才是最终的BBR）：

<div align=center>
<img src="../img/R-CNN/pic_BBR4.png" /> 
</div>

> 这里为什么会将tw,th写成exp形式？ 

是因为tw,th代表着缩放的尺寸，这个尺寸是>0的，所以使用exp的形式正好满足这种约束。 
也就是，我们将转换d换成转换t,就得到了P到G的映射。 di -> ti。 
现在我们只需要学习 这四个变换dx(P),dy(P),dw(P),dh(P)，然后最小化t和d之间的距离，最小化这个loss，即可。

注意：此时看起来我们只要输入P的四维向量，就可以学习,然后求出，但是，其实我们输入的是pool5之后的features，记做φ5，因为如果只是单纯的靠坐标回归的话，CNN根本就没有发挥任何作用，但其实，bb的位置应该有CNN计算得到的features来fine-tune。所以，我们选择将pool5的feature作为输入。 


<div align=center>
<img src="../img/R-CNN/pic_BBR5.png" /> 
</div>

loss为：

<div align=center>
<img src="../img/R-CNN/pic_BBR6.png" /> 
</div>

最后，我们只需要利用梯度下降或最小二乘求解w即可。
另外不要认为BBR和分类信息没有什么关系，是针对每一类都会训练一个BBR


------

### 7.R-CNN测试的一般步骤

+  1.输入一张多目标图像，采用selective search算法提取约2000个建议框；

+  2.先在每个建议框周围加上16个像素值为建议框像素平均值的边框，再直接变形为227×227的大小；

+ 3.先将所有建议框像素减去该建议框像素平均值后【预处理操作】，再依次将每个227×227的建议框输入AlexNet CNN网络获取4096维的特征【比以前的人工经验特征低两个数量级】，2000个建议框的CNN特征组合成2000×4096维矩阵；

+ 4.将2000×4096维特征与20个SVM组成的权值矩阵4096×20相乘【20种分类，SVM是二分类器，则有20个SVM】，获得2000×20维矩阵表示每个建议框是某个物体类别的得分；

+ 5.分别对上述2000×20维矩阵中每一列即每一类进行非极大值抑制剔除重叠建议框，得到该列即该类中得分最高的一些建议框；

+ 6.分别用20个回归器对上述20个类别中剩余的建议框进行回归操作，最终得到每个类别的修正后的得分最高的bounding box。

------

### 8.R-CNN的训练过程

+  1.有监督预训练

<div align=center>
<img src="../img/R-CNN/pic_train1.png" /> 
</div>

ILSVRC样本集上仅有图像类别标签，没有图像物体位置标注； 
采用AlexNet CNN网络进行有监督预训练，学习率=0.01； 
该网络输入为227×227的ILSVRC训练集图像，输出最后一层为4096维特征->1000类的映射，训练的是网络参数。

+ 2.特定样本下的微调

<div align=center>
<img src="../img/R-CNN/pic_train2.png" /> 
</div>

PASCAL VOC 2007样本集上既有图像中物体类别标签，也有图像中物体位置标签； 
采用训练好的AlexNet CNN网络进行PASCAL VOC 2007样本集下的微调，学习率=0.001【0.01/10为了在学习新东西时不至于忘记之前的记忆】； 
mini-batch为32个正样本和96个负样本【由于正样本太少】； 
该网络输入为建议框【由selective search而来】变形后的227×227的图像，修改了原来的1000为类别输出，改为21维【20类+背景】输出，训练的是网络参数。


+ 3.SVM训练

<div align=center>
<img src="../img/R-CNN/pic_train3.png" /> 
</div>

由于SVM是二分类器，需要为每个类别训练单独的SVM； 
SVM训练时输入正负样本在AlexNet CNN网络计算下的4096维特征，输出为该类的得分，训练的是SVM权重向量； 
由于负样本太多，采用hard negative mining的方法在负样本中选取有代表性的负样本，该方法具体见

+ 4.Bounding-box regression训练

<div align=center>
<img src="../img/R-CNN/pic_train4.png" /> 
</div>


结果怎么样?

PASCAL VOC 2010测试集上实现了53.7%的mAP；

PASCAL VOC 2012测试集上实现了53.3%的mAP；

计算Region Proposals和features平均所花时间：13s/image on a GPU；53s/image on a CPU


还存在什么问题?

很明显，最大的缺点是对一张图片的处理速度慢，这是由于一张图片中由selective search算法得出的约2k个建议框都需要经过变形处理后由CNN前向网络计算一次特征，这其中涵盖了对一张图片中多个重复区域的重复计算，很累赘；

知乎上有人说R-CNN网络需要两次CNN前向计算，第一次得到建议框特征给SVM分类识别，第二次对非极大值抑制后的建议框再次进行CNN前向计算获得Pool5特征，以便对建议框进行回归得到更精确的bounding-box，这里文中并没有说是怎么做的，个人认为也可能在计算2k个建议框的CNN特征时，在硬盘上保留了2k个建议框的Pool5特征，虽然这样做只需要一次CNN前向网络运算，但是耗费大量磁盘空间；

训练时间长，虽然文中没有明确指出具体训练时间，但由于采用RoI-centric sampling【从所有图片的所有建议框中均匀取样】进行训练，那么每次都需要计算不同图片中不同建议框CNN特征，无法共享同一张图的CNN特征，训练速度很慢；

整个测试过程很复杂，要先提取建议框，之后提取每个建议框CNN特征，再用SVM分类，做非极大值抑制，最后做bounding-box回归才能得到图片中物体的种类以及位置信息；同样训练过程也很复杂，ILSVRC 2012上预训练CNN，PASCAL VOC 2007上微调CNN，做20类SVM分类器的训练和20类bounding-box回归器的训练；这些不连续过程必然涉及到特征存储、浪费磁盘空间等问题。

------

### 9. Selective Search for Object Regognition（论文解读）

物体识别，在之前的做法主要是基于穷举搜索（Exhaustive Search）：选择一个窗口扫描整张图像（image），改变窗口的大小，继续扫描整张图像。这种做法是比较原始直观，改变窗口大小，扫描整张图像，非常耗时。若能过滤掉一些无用的box将会节省大量时间。这就是本文中Selective Search(选择性搜索)的优点。

选择性搜索（Selective Search)综合了穷举搜索（Exhausticve Search)和分割（Segmentation)的方法，意在找到一些可能的目标位置集合。作者将穷举搜索和分割结合起来，采取组合策略保证搜索的多样性，其结果达到平均最好重合率为0.879。能够大幅度降低搜索空间，提高程序效率，减小计算量。

*** Introduction**

图像（Image）包含的信息非常的丰富，其中的物体（Object）有不同的形状（shape）、尺寸（scale）、颜色（color）、纹理（texture），要想从图像中识别出一个物体非常的难，还要找到物体在图像中的位置，这样就更难了。下图给出了四个例子，来说明物体识别（Object Recognition）的复杂性以及难度。

（a）中的场景是一张桌子，桌子上面放了碗，瓶子，还有其他餐具等等。比如要识别“桌子”，我们可能只是指桌子本身，也可能包含其上面的其他物体。这里显示出了图像中不同物体之间是有一定的层次关系的。

（b）中给出了两只猫，可以通过纹理（texture）来找到这两只猫，却又需要通过颜色（color）来区分它们。

（c）中变色龙和周边颜色接近，可以通过纹理（texture）来区分。

（d）中的车辆，我们很容易把车身和车轮看做一个整体，但它们两者之间在纹理（texture）和颜色（color）方面差别都非常地大。

<div align=center>
<img src="../img/R-CNN/pic_SS1.png" /> 
</div>

 上面简单说明了一下在做物体识别（Object Recognition）过程中，不能通过单一的策略来区分不同的物体，需要充分考虑图像物体的多样性（diversity）。另外，在图像中物体的布局有一定的层次（hierarchical）关系，考虑这种关系才能够更好地对物体的类别（category）进行区分。

 在深入介绍Selective Search之前，先说说其需要考虑的几个问题：

 1.适应不同尺度（Capture All Scales）：穷举搜索（Exhaustive Selective）通过改变窗口大小来适应物体的不同尺度，选择搜索（Selective Search）同样无法避免这个问题。算法采用了图像分割（Image Segmentation）以及使用一种层次算法（Hierarchical Algorithm）有效地解决了这个问题。
 
2.多样化（Diversification）：单一的策略无法应对多种类别的图像。使用颜色（color）、纹理（texture）、大小（size）等多种策略对（【1】中分割好的）区域（region）进行合并。

3.速度快（Fast to Compute）：算法，就像功夫一样，唯快不破！

研究了一下，没有研究完，需要重要的参考文献：

[1].Selective Search for Object Recognition

[2].Efficient Graph-Based Image Segmentation

[3].SIFT: Distance image features from scale-invariant keypoints

[4].HUG: Histograms of oriented gradients for human detection

[5].DPM: Object detection with discriminatively trained part based models



------

### Reference
<https://blog.csdn.net/v1_vivian/article/details/78599229>

<https://github.com/broadinstitute/keras-rcnn>

<https://blog.csdn.net/Katherine_hsr/article/details/79266880>

<https://blog.csdn.net/v1_vivian/article/details/80245397>

<https://blog.csdn.net/bryant_meng/article/details/78613881?utm_source=blogxgwz1>

<https://blog.csdn.net/v1_vivian/article/details/80292569>

<https://blog.csdn.net/zijin0802034/article/details/77685438?utm_source=blogxgwz0>

<https://blog.csdn.net/mao_kun/article/details/50576003>


### 2.SPP-net


