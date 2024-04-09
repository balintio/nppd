# Partitioning Pyramids

<p align="center">
<img src=assets/header.png width="100%" />
<a href="https://balint.io/nppd/">Website</a> &emsp;|&emsp; <a href="#citation">BibTeX</a> 
</p>

[**Neural Partitioning Pyramids for Denoising Monte Carlo Renderings**](https://balint.io/nppd/nppd_paper.pdf)<br/>
[Martin Bálint](https://orcid.org/0000-0001-6689-4770),
[Krzysztof Wolski](https://orcid.org/0000-0003-2290-0299),
[Karol Myszkowski](https://orcid.org/0000-0002-8505-4141),
[Hans-Peter Seidel](https://orcid.org/0000-0002-1343-8613),
[Rafał Mantiuk](https://orcid.org/0000-0003-2353-0349)<br/>

Also check out [Noisebase](https://github.com/balintio/noisebase)!

## Installation

Use the following commands to clone and set up our repos:

```bash
# Clone this repo as well as Noisebase
git clone https://github.com/balintio/noisebase
git clone https://github.com/balintio/nppd

# Make environment
cd nppd
conda env create -f environment.yaml
conda activate nppd

# Add NPPD and Noisebase to your Python path
# by installing them as editable
pip install -e ../noisebase
pip install -e .
```

## Data

Download the [Zip archive](https://neural-partitioning-pyramids.mpi-inf.mpg.de/data/nppd_pretrained.zip) including our pretrained models and unpack them in the `nppd` folder.
```bash
wget https://neural-partitioning-pyramids.mpi-inf.mpg.de/data/nppd_pretrained.zip
unzip nppd_pretrained.zip
```

Download our 8 spp test data (320 GB):
```bash
nb-download sampleset_test8_v1
```

Optionally download our 32 spp test data (this has shorter sequences) (255 GB):
```bash
nb-download sampleset_test32_v1
```

If you want to train NPPD download our training data (1.84 TB):
```bash
nb-download sampleset_v1
```

Be patient, these will take a *while*.

## Evaluation

Export the reference sequences for the datasets you just downloaded and run inference:
```bash
nb-save-reference sampleset_test8_v1
nb-save-reference sampleset_test32_v1

python src/test.py --config-name=small_2_spp
python src/test.py --config-name=small_4_spp
python src/test.py --config-name=large_8_spp
python src/test.py --config-name=large_32_spp
```

Compute metrics for the inferred images:
```bash
nb-compute-metrics sampleset_test8_v1 outputs/small_2_spp
nb-compute-metrics sampleset_test8_v1 outputs/small_4_spp
nb-compute-metrics sampleset_test8_v1 outputs/large_8_spp
nb-compute-metrics sampleset_test32_v1 outputs/large_32_spp
```

Finally, export a table with all the results:
```bash
nb-result-table sampleset_test8_v1 \
outputs/small_2_spp,outputs/small_4_spp,outputs/large_8_spp \
psnr,ssim,msssim,fvvdp,flip --sep=" | "
nb-result-table sampleset_test32_v1 outputs/large_32_spp \
psnr,ssim,msssim,fvvdp,flip --sep=" | "
```

| name         | psnr    | ssim    | msssim  | fvvdp   | flip    |
| -            | -       | -       | -       | -       | -       |
| small_2_spp  | 28.3952 | 0.87283 | 0.95905 | 7.35612 | 0.10702 |
| small_4_spp  | 29.3461 | 0.88300 | 0.96597 | 7.66470 | 0.09624 |
| large_8_spp  | 30.3686 | 0.89657 | 0.97347 | 7.96943 | 0.08451 |
| large_32_spp | 32.5832 | 0.92474 | 0.98520 | 8.51527 | 0.06371 |

<details>
<summary>Comparison to reported results</summary>

These results differ slightly from the ones reported in our paper. This is mostly within the expected run-to-run variance and has zero impact on our original conclusions. 

Most noticeable is the scaling of FovVideoVDP on long test sequences was wrong in the original implementation. We fixed this issue with the release of Noisebase, including our metrics scripts. Each compared method was equally affected, so our original conclusions still hold.

Our original differentiable PU21 implementation proved numerically unstable, likely due to floating point and optimisation differences between Pytorch and Tensorflow. We replaced it with a log plus one operation that behaves very similarly.

We also compressed and tidied up our test dataset, shortening and lengthening some sequences. Again, this has a measurable but insignificant impact on the results.
</details>

## Training

```bash
python src/train.py --config-name=small_2_spp
```

You can train other configurations similarly. Our code uses Lightning and should seamlessly use multiple GPUs if available.

[More details on the perceptual loss.](src/loss/README.md)

## Citation

```bibtex
@inproceedings{balint2023nppd,
    author = {Balint, Martin and Wolski, Krzysztof and Myszkowski, Karol and Seidel, Hans-Peter and Mantiuk, Rafa\l{}},
    title = {Neural Partitioning Pyramids for Denoising Monte Carlo Renderings},
    year = {2023},
    isbn = {9798400701597},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3588432.3591562},
    doi = {10.1145/3588432.3591562},
    booktitle = {ACM SIGGRAPH 2023 Conference Proceedings},
    articleno = {60},
    numpages = {11},
    keywords = {upsampling, radiance decomposition, pyramidal filtering, kernel prediction, denoising, Monte Carlo},
    location = {<conf-loc>, <city>Los Angeles</city>, <state>CA</state>, <country>USA</country>, </conf-loc>},
    series = {SIGGRAPH '23}
}
```