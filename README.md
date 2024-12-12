# Dial2MSA-Verified

Welcome to the Dial2MSA-Verified repository. Dial2MSA-Verified is an extension of the Dial2MSA dataset that includes verified translations for Gulf, Egyptian, Levantine, and Maghrebi dialects. This corpus is an evaluation dataset of multiple Arabic dialects in social media

## Data Split

| **Dataset**      | **EGY** | **GLF** | **LEV** | **MGR** |
|------------------|---------|---------|---------|---------|
| Dial2MSA-V-train | 9,099   | 6,575   | 4,101   | 3,312   |
| Dial2MSA-V-dev   | 200     | 200     | 200     | 200     |
| Dial2MSA-V-test  | 2000 3-R| 2000 3-R| 2000 2-R| 2000 2-R|

"R" indicates the number of available MSA references: 2,000 tweets in EGY and GLF have three MSA references each, and 2,000 tweets in LEV and MGR have two MSA references each.

## Multi-Reference Evaluation 

Evaluation was conducted using the Bilingual Evaluation Understudy (BLEU) and chrF++ metrics, employing the multi-reference SacreBLEU implementation [post-2018-call](https://aclanthology.org/W18-6319/).

This is an example code for model evaluation using multi-reference translations in the [`Dial2MSA_Evaluation.ipynb`](https://github.com/khered20/Dial2MSA-Verified/blob/main/Dial2MSA_Evaluation.ipynb) . Both `test_pred_egy.txt` and `test_pred_allDialects8000.txt` are fake prediction files for usage demonstrations. 

### Requirements

```
- sacrebleu
- sacremoses
- pandas
- numpy
```

## Additional Corpora

 Additional Corpora we used in the training:
1. **PADIC** - Covers six Arabic cities from the Levant and Maghrebi regions.
   > Reference: [meftouh2018padic](https://sourceforge.net/projects/padic/).
2. **MADAR** -  Multilingual parallel dataset of 25 Arabic city-specific dialects and MSA.
   > Reference: [bouamor-etal-2018-madar](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/).
3. **Arabic STS** - Provides MSA, Egyptian, and Saudi dialect translations for English sentences.
   > Reference: [alsulaiman2022semantic](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0272991).
4. **Emi-NADI** - Our dataset to address the scarcity of Emirati dialect parallel corpora.
   > Reference: [khered2023Emi-NADI](https://github.com/khered20/UniManc_NADI2023_ArabicDialectToMSA_MT/blob/main/datasets/Emi-NADI.csv)

| **Dataset**      | **EGY** | **GLF** | **LEV** | **MGR** |
|------------------|---------|---------|---------|---------|
| PADIC            | 0       | 0       | 12,824  | 25,648  |
| MADAR-train      | 13,800  | 15,400  | 18,600  | 29,200  |
| Arabic STS       | 2,758   | 2,758   | 0       | 0       |
| Emi-NADI         | 0       | 2,712   | 0       | 0       |

## Usage

This corpus is for research purposes only. Please cite the relevant publications if you use Dial2MSA-Verified.

## Citation

If you find this work or the provided dataset useful in your research or projects, please cite our paper:

```bib

==

```