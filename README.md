# Flash-To-Ambient Model

In the procces of generating digital images from scences sometimes the level of light very low and insufficient to get a properly digitization of the image. Thus, noisy and blurry areas are produced in the image. We can handle this situation with an external illumination device such as a camera flash, but here is when other challenges are generated. The camera flash can be blinding and too strong for a scene, so instead of enhancing the low light image, sometimes, it causes very bright and very dark areas. Another problem in the flash image is the shadows. These shadows sometimes cover considerable areas of the scene depending on the direction of the camera flash. And finally, getting the correct tone of the scene objects in a flash image becomes very difficult, because the color of objects changes due the flash illumination. In contrast, in an ambient image, the illumination of the objects not depends so much of their position, because the available light can be more evenly distributed.

![Screenshot](imgs/generator-model.png)

The architecture has two CNNs, the generator, generates synthetic ambient images, and the discriminator network classifies if their input images are authentic. The generator network has as an encoder the [VGG-16](https://arxiv.org/abs/1409.1556). The generator models the translation from flash images($I_f$) to ambient images($\hat{I}_a$). The real image($I_a$) is classified by the discriminator as a real, while the synthetic image($\hat{I}_a$) is classified by the discriminator as a fake. The discriminator is the same proposed by Isola et al. in the [pix2pix](https://arxiv.org/abs/1611.07004) framework.

## Qualitative results

| Flash image | Synthetic ambient image | Ambient image |
|:---:|:---:|:---:|
|![](imgs/flash_it_105.png)|![Synthetic ambient image](imgs/fake_it_105.png)|![Ambient image](imgs/real_it_105.png)|

<p align="left">
    <img src="imgs/flash_it_105.png" alt="drawing" style="width:100px; margin:1px;"/>
    <img src="imgs/fake_it_105.png" alt="drawing" style="width:100px; margin:1px;"/>
    <img src="imgs/real_it_105.png" alt="drawing" style="width:100px; margin:1px;"/><br>
</p>
<img src="imgs/flash_it_89.png" alt="drawing" style="width:200px; margin:1px; float:left;"/>
<img src="imgs/fake_it_89.png" alt="drawing" style="width:200px; margin:1px;"/>
<img src="imgs/real_it_89.png" alt="drawing" style="width:200px; margin:1px;"/><br>

<img src="imgs/flash_it_112.png" alt="drawing" style="width:200px; margin:1px; float:left;"/>
<img src="imgs/fake_it_112.png" alt="drawing" style="width:200px; margin:1px;"/>
<img src="imgs/real_it_112.png" alt="drawing" style="width:200px; margin:1px;"/><br>

<img src="imgs/flash_it_4.png" alt="drawing" style="width:200px; margin:1px; float:left;"/>
<img src="imgs/fake_it_4.png" alt="drawing" style="width:200px; margin:1px;"/>
<img src="imgs/real_it_4.png" alt="drawing" style="width:200px; margin:1px;"/><br>

<img src="imgs/flash_it_15.png" alt="drawing" style="width:200px; margin:1px; float:left;"/>
<img src="imgs/fake_it_15.png" alt="drawing" style="width:200px; margin:1px;"/>
<img src="imgs/real_it_15.png" alt="drawing" style="width:200px; margin:1px;"/><br>

<img src="imgs/flash_it_40.png" alt="drawing" style="width:200px; margin:1px; float:left;"/>
<img src="imgs/fake_it_40.png" alt="drawing" style="width:200px; margin:1px;"/>
<img src="imgs/real_it_40.png" alt="drawing" style="width:200px; margin:1px;"/><br>



Figure 1. Some results of our model based on the pix2pix framework. Flash image (left), image generated through the generator network (middle) and the ground truth, the ambient image(right).