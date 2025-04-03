
Ensemble Detection and “Segment Anything” Pseudo-Labeling Pipeline
Ensemble Zero-Shot Detectors for Reliable Pseudo-Boxes
Leveraging multiple open-vocabulary detectors in committee can greatly improve the reliability of pseudo-labels. In fact, ensemble methods like Weighted Boxes Fusion (WBF) have shown that merging predictions from different models yields more accurate bounding boxes than any single model​
ARXIV.ORG
. WBF uses the confidence and overlap of all detectors’ boxes to produce a consensus box, effectively a weighted average of highly overlapping predictions​
ARXIV.ORG
. In your context, models such as Grounding DINO, Florence-2, OWL-ViT, or GLIP/Detic can each propose candidate boxes for all objects; by measuring their agreement (high IoU overlap and consistent labels), you can filter for boxes that multiple models concur on. This concept is supported by recent evaluations of zero-shot detectors: for example, a study on automating eyeglasses labeling compared six foundation models (Grounding DINO, Detic, OWL-ViT/v2, “YOLO World”, and Florence-2) and found significant variance in their performance​
MDPI.COM
. Detic achieved the highest AP, but Grounding DINO and OWL-ViT v2 excelled in high-recall scenarios​
MDPI.COM
 – suggesting an ensemble could cover each other’s blind spots. The authors also emphasize prompt engineering (i.e. the text queries given to open-vocab models) as crucial for optimal detection​
MDPI.COM
, which aligns with your idea of ensuring label consistency (e.g. mapping synonyms or descriptions from different models to a common label space). In practice, a consensus filter might use IoU thresholding and CLIP-based text similarity to fuse predictions – an approach analogous to ensemble box NMS but accounting for label semantics. Such multi-detector agreement strategies have precedent in multi-model pseudo-labeling: Qiu et al. (2024) propose a dual-uncertainty multi-model framework where multiple segmentation models vote on pixel labels, yielding higher-quality pseudo masks by trusting regions of low uncertainty agreement​
AIMSPRESS.COM
. Likewise, in domain adaptation, ensembling multiple models to generate pseudo-labels has been shown to produce more robust, “interpretable” training data​
ARXIV.ORG
. 

Example of combining a language-based detector with SAM: Grounding DINO (center) detects objects from a text prompt, and Segment Anything (right) refines these into precise masks​
DOCS.AUTODISTILL.COM
. In your pipeline, multiple detectors could similarly vote on the most confident box proposals before segmentation. Beyond academic studies, open-source tools have embraced detector ensembles for auto-labeling. The Grounded-SAM pipeline (IDEA Research, 2023) explicitly encourages swapping in different detectors as “experts” to detect & segment “anything”​
GITHUB.COM
​
GITHUB.COM
. They demonstrated that Grounding DINO + SAM could achieve state-of-the-art zero-shot segmentation results on the Segmentation in the Wild benchmark​
GITHUB.COM
. Similarly, Roboflow’s Autodistill framework recommends using Grounding DINO as a base detector and then Grounded SAM (GroundingDINO + SAM together) to generate segmentation masks for unlabeled images​
DOCS.AUTODISTILL.COM
. This approach automatically labels datasets: Grounding DINO first “identifies a wide range of objects” zero-shot, and SAM then produces mask annotations for those detections​
DOCS.AUTODISTILL.COM
. If one model misses an object, Autodistill suggests trying a broader detector like Detic (20k classes) as an alternative​
DOCS.AUTODISTILL.COM
. In essence, your committee-based strategy aligns well with these existing pipelines – using an ensemble of zero-shot detectors, possibly with box fusion or voting, can yield high-confidence, “promptable” boxes that are ideal to feed into SAM.
Prompt-Based Segmentation and Stability Checking with SAM
Once reliable bounding boxes are obtained, the next step is to generate segmentation masks using the Segment Anything Model (SAM) by feeding those boxes as prompts. SAM was designed to produce high-quality masks from various prompt types (boxes, points, etc.), and it often excels given a precise prompt​
OPENREVIEW.NET
. However, a key insight from recent research is that SAM’s output stability can vary with prompt quality. Stable-SAM (Fan et al., 2025) provides a comprehensive analysis of SAM’s behavior under imperfect prompts​
OPENREVIEW.NET
. They found that with imprecise or shifted boxes, SAM’s mask decoder might latch onto background or only part of the object, causing inconsistent results​
OPENREVIEW.NET
. To address this, they introduced a lightweight plugin that adaptively shifts SAM’s attention to the intended region, dramatically improving stability for “casual” (imperfect) prompts​
OPENREVIEW.NET
. This underscores the importance of your perturbation test: if a slight shift or scale of the box causes SAM to segment a very different region, the mask is not reliable. Perturbation-based stability testing is reminiscent of consistency regularization in semi-supervised learning – here used as a quality signal. In fact, Rahman et al. (2024) explicitly leverage such perturbations in PP-SAM, a method to fine-tune SAM for medical images. They found that training SAM with perturbed bounding box prompts (BBP) made it far more robust at inference – e.g. a 50-pixel box jitter caused a 20-37% drop in DICE for the vanilla model, but after PP-SAM fine-tuning, the model’s output hardly changed​
ARXIV.ORG
. This validates your idea: robust masks remain stable under prompt perturbation, whereas flimsy masks will “break” with slight changes. By selecting masks that survive random shifts/expansions of the prompt box, you effectively filter for outputs that likely capture the true object well. Another way to assess mask quality is to use confidence or IoU measures internally. For instance, the Pseudo-label Aligning work (PAIS, ICCV 2023) notes that a mask’s true IoU can be misaligned with the model’s class confidence​
OPENACCESS.THECVF.COM
. They propose using a separate mask-quality prediction branch to weigh loss terms​
OPENACCESS.THECVF.COM
. In your setting, although there is no ground truth, you could estimate mask confidence by overlapping the mask from the original prompt and the mask from a perturbed prompt – a high overlap (IoU close to 1.0) implies the segmentation is stable and thus likely accurate. This is conceptually similar to the consistency checks used in semi-supervised training (e.g. Mean Teacher models enforce that a student and an EMA teacher predict the same masks under image augmentations​
OPENACCESS.THECVF.COM
). Here, the “augmentations” are prompt perturbations on SAM. Any large disagreement would flag the pseudo-mask as suspect. We haven’t seen a paper that does exactly this prompt-jitter check on SAM for self-training, but it fits naturally with the above findings. Stable-SAM’s improvements also hint that if your boxes are very precise (e.g. tight to object bounds), SAM tends to be stable; issues arise with loose or slightly off-target boxes​
OPENREVIEW.NET
. Thus, using the high-agreement ensemble boxes from step 1 as prompts already gives you a head start: those boxes are likely tight and correct, and then the perturbation test is a final sanity check to confirm SAM’s segmentation isn’t brittle.
Pseudo-Labeling at Scale for Segmentation
With robust SAM masks in hand, you can treat them as pseudo-ground-truth for training a new segmentation model (or fine-tuning SAM itself). This strategy connects to a rich history of pseudo-labeling and self-training in computer vision. Notably, Xie et al.’s Noisy Student (2020) demonstrated that iteratively training on self-generated labels can surpass state-of-the-art – they trained a student model on 300M pseudo-labeled images (generated by a teacher) and achieved a new record on ImageNet​
PAPERSWITHCODE.COM
. The key was to use a large model for labeling and to inject noise/augmentation during student training to refine the pseudo-label quality​
PAPERSWITHCODE.COM
​
PAPERSWITHCODE.COM
. Your proposed pipeline is a modern, segmentation-centric take on this: the ensemble of zero-shot detectors + SAM serves as a powerful teacher (a “committee teacher” of sorts) to label millions of images. Because SAM’s masks are high-quality and your consensus filter ensures high precision, the pseudo dataset will be quite reliable. This setup is very much in line with recent semi-supervised segmentation works. For example, in semi-supervised instance segmentation, a method like STAC (2020) first generates masks from a pre-trained model then trains a student with strong augmentations on those masks​
OPENACCESS.THECVF.COM
. Newer approaches like Unbiased Teacher (2021) go further and update the teacher model continuously (EMA) so that pseudo-labels improve as the student learns​
OPENACCESS.THECVF.COM
. In a large-scale scenario, you might similarly fine-tune your SAM (or a custom segmentation network) in iterations: label a batch of images with the current model ensemble, fine-tune on those pseudo-labels, then update the pseudo-labels for the next round, and so on – gradually expanding and improving the mask annotations. This self-training loop could eventually allow the model to surpass the initial quality of SAM itself, especially on domains or object classes that SAM+detectors initially struggled with. There is early evidence that using SAM and detectors for automatic mask labeling is effective. Ren et al. (Grounded SAM, 2024) describe a fully automated annotation pipeline that needs only images as input​
ARXIV.ORG
. In their setup, a vision-language model like BLIP first generates text captions for an image, then Grounding DINO uses those texts to detect corresponding regions, and SAM produces precise masks – yielding labeled segments without any manual work​
ARXIV.ORG
. They even incorporate a “Recognize Anything Model” (RAM) to assign category names to the segments, achieving a true open-world annotation system​
ARXIV.ORG
. Your use of multiple detectors with overlap agreement could be seen as an alternative way to get high-quality proposals without captions – essentially focusing on object consensus rather than language prompts. In another applied study, Lim et al. (2024) used Grounding DINO + SAM to label a specialty dataset (tracking sweet peppers in farming) and only had humans refine a subset of those labels​
ARXIV.ORG
​
ARXIV.ORG
. With this weakly-supervised data, they trained a YOLOv8 model for pepper detection and segmentation, greatly reducing manual effort​
ARXIV.ORG
​
ARXIV.ORG
. This shows the feasibility of scaling up with minimal human intervention – exactly the promise of your pipeline. The authors reported that their largely pseudo-labeled model achieved high precision (90%+ in detecting peppers)​
ARXIV.ORG
, underlining that carefully filtered SAM masks can serve as ground truth for training. To further improve pseudo-label quality at scale, researchers have looked into confidence and uncertainty heuristics. One idea is to use only masks above a certain confidence threshold or IoU agreement. In Pseudo-Label Alignment for Instance Segmentation (PAIS, 2023), Hu et al. argue that many pseudo-masks that are actually good get discarded by naive thresholding because their class score was low​
OPENACCESS.THECVF.COM
. They devise a dynamic loss that down-weights a pseudo-mask if either its class confidence or mask quality seems low​
OPENACCESS.THECVF.COM
, rather than hard-dropping it. In your case, since the detectors supply a class label (possibly with a confidence) and SAM can provide a mask IoU confidence (perhaps estimated via stability as discussed), you could adopt a similar soft-weighting: e.g. give higher training weight to pseudo-masks where both the detection confidence and SAM stability are high, but still use masks that are, say, very accurate shape-wise even if the detector was less certain of the category. Moreover, a recent medical imaging paper by Qiu et al. (2024) trains an ensemble of two segmentation models and only keeps pseudo-label pixels where both models agree with high certainty​
AIMSPRESS.COM
​
AIMSPRESS.COM
. This dual-uncertainty approach improved Dice scores by 4–6% in their semi-supervised segmentation task​
AIMSPRESS.COM
. Such an approach could inspire an extension to your committee: not only must multiple detectors agree on the box, but one could run multiple SAM variants (e.g. SAM-H, SAM-L, or other segmentation models like Mask2Former) on the prompt and require mask agreement as well. Agreement across different segmenters would be a very strict criterion, but would ensure extremely high-quality masks (this is akin to self-ensembling on the segmentation side). While this might be overkill, it highlights the general principle: consensus and consistency are key to trustworthy pseudo-labels.
Fine-Tuning SAM and Related Models
Finally, improving a custom SAM (or SAM-like model) via supervised fine-tuning on the collected dataset is a logical next step. There is active research on adapting foundation models like SAM to new domains using pseudo or limited labels. For instance, MedSAM (2023) and other works successfully fine-tune SAM on medical images by using relatively small expert-labeled datasets, achieving strong results in that domain​
NATURE.COM
​
GITHUB.COM
. In a self-training spirit, you could bootstrap SAM on your pseudo-labeled data then iterate. No work to date has explicitly reported re-training SAM on millions of purely self-generated masks, likely because SAM was just introduced in 2023. However, the massive original SAM dataset (SA-1B) was itself gathered with a model-in-the-loop data engine​
AR5IV.ORG
​
AR5IV.ORG
. The authors of SAM note that their model powered the annotation of 1 billion masks – humans provided prompts and verified masks, but much of the heavy lifting was automatic​
AR5IV.ORG
. This demonstrates the scalability of such approaches. Your proposal removes the human prompt step by using zero-shot detectors as prompt generators, which is a novel twist. If successful, it could dramatically cut down the need for human annotators even further than SAM’s assisted labeling did. In the meantime, there are related open-source efforts and benchmarks you might draw on. The Label Studio team, for example, published a guide on integrating Grounding DINO with SAM to perform zero-shot labeling in an annotation UI​
LABELSTUD.IO
. This could be useful if you plan any manual verification stage. Benchmarks like LVIS and ODinW (Open Detection in the Wild) have open-vocabulary test sets where a model fine-tuned on your pseudo-labels could be evaluated to quantify how well the self-training worked. And as mentioned, the Segmentation in the Wild challenge (CVPR 2023 workshop) is specifically aimed at testing zero-shot segmentation performance across a variety of object categories​
GITHUB.COM
 – a fine-tuned SAM with your pseudo-labeled corpus might establish a new state-of-the-art there. Furthermore, research on open-vocabulary segmentation is booming: one work introduces Open-Vocabulary SAM, which augments SAM with CLIP’s image/text knowledge so it can recognize up to 22k classes while segmenting​
ECVA.NET
. Their approach uses a two-way knowledge transfer (SAM → CLIP and CLIP → SAM) to make a unified model that can segment and name anything interactively​
ECVA.NET
​
ECVA.NET
. While this is more about model architecture than data pipeline, it underscores a similar goal of “segmenting anything and assigning labels”. In your case, the ensemble detectors already provide class labels for the pseudo-masks (and these detectors are grounded in vision-language models like CLIP or Florence), so your fine-tuned model could inherit an open-vocabulary ability. In fact, Florence and OWL-ViT were designed for broad vocab detection; by training on their outputs, your SAM-derived model might naturally become an open-vocabulary segmenter. This is exactly what the community is moving toward – combining detection, segmentation, and recognition in a unified network trained on vast, automatically-labeled data.
Resources and Further Reading
Grounded-SAM (Ren et al., 2024) – ArXiv paper and GitHub repo demonstrating how to combine Grounding DINO, SAM, BLIP, and more for automatic segmentation and even image editing​
ARXIV.ORG
. They report 48.7 mAP on a zero-shot segmentation benchmark using SAM + GroundingDINO​
ARXIV.ORG
, and their repo’s Highlight Extension Projects list many community techniques (e.g. using Stable Diffusion or ChatGPT in the loop). This is a great reference for assembling “foundation modules” as you propose.
Autodistill by Roboflow (2023) – Open-source tool that enables “training with zero annotations.” Their docs recommend Grounding DINO + Grounded-SAM for detection/segmentation labeling​
DOCS.AUTODISTILL.COM
. Autodistill supports plugging in custom base models (including ensembles) and can output COCO-format datasets, which might be handy for managing your millions of pseudo-labeled images.
Weighted Boxes Fusion (Solovyev et al., 2021) – Paper introducing the WBF algorithm for ensembling detectors​
ARXIV.ORG
. The WBF GitHub provides code that you could use to merge the bounding boxes from Grounding DINO, Florence, OWL-ViT, etc. into a single set of high-confidence boxes. This could simplify your consensus logic.
Stable-SAM (Fan et al., ICLR 2025) – Research that adds a Deformable Sampling Plugin to SAM for stable results with imperfect prompts​
OPENREVIEW.NET
​
OPENREVIEW.NET
. Their code is available, and while their focus was on inference-time robustness, the ideas could inspire metrics for your perturbation test. They quantify stability across a “spectrum of prompt qualities,” which might help in designing a prompt perturbation schedule to test your masks.
PP-SAM (Rahman et al., CVPRW 2024) – Workshop paper on fine-tuning SAM for perturbation robustness​
ARXIV.ORG
. They share an approach for augmenting training with random box shifts. This could be directly relevant if you plan to fine-tune your SAM model on pseudo-labels – you might incorporate their perturbation augmentation to ensure the model doesn’t overfit to exact boxes.
Pseudo-Labeling in Detection/Segmentation – For theoretical background, see Pseudo Labeling for Semi-supervised Learning (a 2023 survey) and works like STAC​
OPENACCESS.THECVF.COM
 and Unbiased Teacher​
OPENACCESS.THECVF.COM
 (semi-supervised object detection), or PAIS​
OPENACCESS.THECVF.COM
​
OPENACCESS.THECVF.COM
 (semi-supervised instance segmentation). These illustrate the evolution from using static pseudo-labels to iterative refinement and uncertainty-based filtering. Your method can be seen as an extension of these ideas powered by foundation models.
In summary, your proposed pipeline is well-aligned with the cutting edge of computer vision research. By fusing an ensemble of zero-shot detectors (to maximize recall and precision of object proposals) with a powerful promptable segmenter (SAM), and then using stability and consensus checks as quality control, you are essentially building an automatic annotation engine. This engine can produce a trove of training data to improve itself (via self-training) or to train specialized segmentation models. Recent papers and projects have explored each piece of this puzzle – from ensemble detection and open-vocabulary models to SAM’s capabilities and limitations – which provides a strong foundation for your approach. If executed at the scale of millions of images, this could lead to a new state-of-the-art in open-world segmentation, much like how Noisy Student achieved state-of-the-art in classification through massive self-training​
PAPERSWITHCODE.COM
. Your intuition to require multi-model agreement and prompt perturbation stability will help ensure the pseudo-labels remain high-quality even as quantity grows, addressing the usual pitfalls of self-training (confirmation bias and noise). In effect, you would be segmenting anything with greater confidence than ever, pushing SAM and its kin to the next level. Good luck, and happy annotating! Sources: The details and methods discussed are drawn from recent literature on ensemble detection​
ARXIV.ORG
​
MDPI.COM
, open-world segmentation pipelines​
ARXIV.ORG
​
DOCS.AUTODISTILL.COM
, robustness analyses of SAM​
OPENREVIEW.NET
​
ARXIV.ORG
, and advances in semi-supervised learning for vision​
OPENACCESS.THECVF.COM
​
ARXIV.ORG
, among others, as cited above.
