# Robustness, Consistency and Credibility Eval for LLMs

In recent few months, the **Large Language Models(LLMs)** --- in particular ChatGPT --- have swept the world by causing a huge influence on numerous domains. Assisted by the [WebUI](https://chat.openai.com), open-source models, [APIs](https://platform.openai.com/docs/api-reference), or the [ecosystem Plugins](https://openai.com/blog/chatgpt-plugins), these LLMs successfully immerse into everyone's life, worldwide. 

This success of LLMs is truly unprecedented. However, at the current stage, LLMs are not perfect and pose numerous potential risks, including difficulties in identifying noisy inputs. These risks, particularly challenging in large-scale applications and deployments,  urgently require evaluation. Additionally, the massive and unknown training data of LLMs present a pressing issue in selecting trustworthy evaluation data.

Driven by the above two points, we introduce this automated framework to conduct a systematic evaluation of LLMs and the data. We hope this can be helpful to build a more reliable evaluation system.



## Table of Contents

* [Motivation](## Motivation)
* [Repo Structure](## Repo Structure)
* [Pre-process](## Pre-process)
* [Evaluation on Benchmarks](## Evaluation on benchmarks)
* [Evaluation on Your Datasets](## Evaluation on your datasets)
* [Benchmarks](## Benchmarks)
* [Contribution](## Contribution)
* [Paper](## Paper)
* [Citation](## citation)



## Motivation

We open-source this tool and framework to assess LLM's performance and identify potential limitations. Our works cover three new terms of **Robustness, Consistency, and Credibility**. 

- **Robustness** focuses on examining LLMs' ability to handle adversarial examples, aligning with real-world deployment scenarios.
- **Consistency** aims to measure the distinction in LLMs' responses to semantically similar inputs.
- **$$\textsf{RTI}$$ (Credibility)** provides insights into the datasets used to train LLMs. This $$\textsf{RTI}$$ score reflects the relative probability that the datasets have been memorized by LLMs.

More details are shown in our [paper](## Assessing Hidden Risks of LLMs: An Empirical Study on Robustness, Consistency, and Credibility).



## Repo Structure

```
- Analysor: scripts for pattern analysis of attacked samples
    |- analysis_dep_tag.py
    |- analysis_pos_parser.py
- API: script to fetch responses from the ChatGPT API in bulk
    |- api.py
- Components
	|- attacker.py
	|- converter.py
	|- generator.py
	|- interpreter.py
- DA
	|- eda.py
	|- token_level_attack.py
	|- visual_letter_map.json
- Datasets: benchmark datasets
	|- Attacked_Data_Primitive
	|- Data_Primitive
	|- Raw_Data
- Evals: scripts for evaluting from different aspects
	|- conEval.py
    |- creEval.py
    |- get_path.py
    |- robEval.py
- Preprocess: scripts for pre-processing the benchmark datasets
	|- ecqa.py
    |- esnli.py
    |- format.py
    |- gsm8k_clean.py
    |- noah_clean.py
    |- qasc.py
- eval.py: script for evaluation of the output responses of the LLMs
- processor.py: script for generating a questions list as the input of the LLMs
```



## Installation

To install our tool, simply run this command:

```python
pip install -e .
```



## Data Format

> If you want to evaluate models on our custom dataset, please read this section carefully, otherwise you may choose to skip it.

To facilitate auto-interpretation and auto-evaluation, we define a unified data format --- **data primitive** (see paper for more details):
$$
D=\{\mathbf{x}_i\}_{i=1}^n=\{(\mathbf{prompt}_i,\mathbf{p}_i,q_i,o_i,a_i)\}_{i=1}^n
$$
An example is as follows:

```json
{
    "id" : {
        "passage" : "...",
        "sub-qa" : [
            {
                "question" : "...",
                "answer" : "..."
            }
        ]
    }
}
```

where `passage` denotes the conditions, and `sub-qa` denotes the sub-questions and corresponding answers connected to the `passage`. 

For datasets with options provided, the data format should be:

```json
{
    "id" : {
        "passage" : "...",
        "sub-qa" : [
            {
                "question" : "...",
                "answer" : "...",
                "options" : [
                    {
                        "option_type": "...",
                        "option_describe": "..."
                    }
                ]
            }
        ]
    }
}
```

All the datasets in this repository have been converted to this format. **If you need to evaluate models on your custom dataset, please convert your dataset to the above format.** You can refer to `converter.py` for converting raw dataset to the data primitive format (take GSM8K dataset as the example here) :

```
python converter.py --dataset 'GSM8K'
```



## Evaluation on Benchmarks

### Get Input Questions List 

To evaluate on our benchmarks, first obtain a list of questions using the following command. (change `dataset` parameter according to different [benchmarks](## Benchmarks)) :

1. Robustness

   ```
   python processor.py --dataset GSM8K --type robustness 
   ```

2. Consistency

   ```
   python processor.py --dataset GSM8K --type consistency \
   	--prompt_dir 'Datasets\Attacked data primitive\Consistency\prompt_template.json'
   ```

3. credibility

   ```
   python processor.py --dataset GSM8K --type credibility 
   ```

Then, feed the question list into a language model, which can be your own model or any publicly available model. And, you need to format the reponse of model as a list: `[['response to Q1 of sample1', 'response to Q2 of sample1', ...], ['response to Q1 of sample2', ...], ...]`.

### Evaluation

Next, evaluate model outputs (formated) using the following command:

```Shell
python eval.py \
	--response_dir '' \ # the path of the formated reponse list
	--indir '' \ # the path of the data primitive format file.
	--type robustness
```


Also, `ChatGPT` inference is provided for direct evaluation :

```Shell
python processor.py --useChatGPT --api_key ['', '']
# provide multiple keys `api_key` to speed up the evaluation process
```



## Evaluation on Custom Datasets

1. Format custom dataset into [data primitive](## Data Format).
2. Use the following command to obtain a question list, processed from the dataset you uploaded through the file path `indir`:

    ```shell
    python processor.py --indir '' --type robustness 
    ```

The remaining steps are the same as the [benchmark testing](## Evaluation on Benchmarks).



## Benchmarks

`Datasets/*` directory stores all datasets as benchmarks tests :

- `Raw_Data`: Raw format datasets;
- `Data_Primitive`: Data primitives format datasets;
- `Attacked_Data_Primitive`: Attacked data primitives format datasets;

Related repos:
  - API: [Reverse engineered ChatGPT API](https://github.com/acheong08/ChatGPT)
  - Datasets: [AQuA](https://github.com/deepmind/AQuA/blob/master/test.json) | [creak](https://github.com/yasumasaonoe/creak/blob/main/data/convert_faviq.py) | [NoahQA](https://github.com/Don-Joey/NoahQA) | [GSM8K](https://github.com/openai/grade-school-math) | [bAbI-tasks](https://github.com/facebookarchive/bAbI-tasks) | [Sen-Making](https://github.com/wangcunxiang/Sen-Making-and-Explanation/tree/master) | [strategyqa](https://github.com/eladsegal/strategyqa) | 
  - Data augment: [eda_nlp: Data augmentation for NLP, presented at EMNLP 2019](https://github.com/jasonwei20/eda_nlp)



## Contribution

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.



## Paper

More details and benchmark results are shown in the paper "Assessing Hidden Risks of LLMs: An Empirical Study on Robustness, Consistency, and Credibility".

[ArXiv](https://arxiv.org/pdf/2305.10235.pdf)|[Lab Page](http://jakezhao.net/)

<img src="https://s2.loli.net/2023/06/09/V34xfDP9qTBsYKO.png" alt="Framework" style="zoom:80%;" />



## Citation

Please use the below BibTeX entry to cite this dataset:

```latex
@misc{ye2023assessing,
      title={Assessing Hidden Risks of LLMs: An Empirical Study on Robustness, Consistency, and Credibility}, 
      author={Wentao Ye and Mingfeng Ou and Tianyi Li and Yipeng chen and Xuetao Ma and Yifan Yanggong and Sai Wu and Jie Fu and Gang Chen and Haobo Wang and Junbo Zhao},
      year={2023},
      eprint={2305.10235},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
