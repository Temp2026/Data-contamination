# Contamination Means Overestimation? A Fine-grained Empirical Study in Code Intelligence

## Experimental Procedure

Below is the research methodology overview for RQ1-RQ5
![alt text](overviewQ1-Q4.jpg)
![alt text](overviewQ5.jpg)

## Prompt

Here is the prompt we used for large language model inference, along with a one-shot example.

Code Translation:

```
# Java->C#

Please translate the following Java function into equivalent C# code. End your answer with 'END OF CASE'.
Java:
private void injectBundleContext(BundleContext bundleContext) {
    this.bundleContext = bundleContext;
    this.resourceLoader = new OsgiBundleResourceLoader(bundleContext.getBundle());
}         
C#:
private void InjectBundleContext(BundleContext bundleContext) {
    this.bundleContext = bundleContext;
    this.resourceLoader = new OsgiBundleResourceLoader(bundleContext.getBundle());
}
END OF CASE
Java:
{entry\['Java\_function']}
C#:
```

```
# Python->jJava

Please translate the following Python code into equivalent Java code. End your answer with 'END OF CASE'.
Python:
class Counter:
    def \_\_init\_\_(self):
        self.count = 0
    def increment(self, delta):
        self.count += delta
        return self.count
Java:
public class Counter {{
    private int count;
    public Counter() {{
        this.count = 0;
    }}
    public int increment(int delta) {{
        this.count += delta;
        return this.count;
    }}
}}
END OF CASE

Python code:
{entry\['question']}
Java code:
```



Code Summarization:

```
# Java->NL
Please summarize the following Java function. End your answer with 'END OF CASE'.
Function:
private void injectBundleContext(BundleContext bundleContext) {
    this.bundleContext = bundleContext; this.resourceLoader = new OsgiBundleResourceLoader(bundleContext.getBundle());
}
Summary:
This Java function injects a `BundleContext` object, stores it, and initializes a `ResourceLoader` with the associated bundle.
END OF CASE
Function:
{entry\['Function']}
Summary:
```

```
# Python>NL
Please summarize the following Java function to natural language. End your answer with 'END OF CASE'.
        Function:
        public boolean saveConfig(Map<String, Object> data, String filename) {{
            try (FileWriter writer = new FileWriter(filename)) {{
                new Gson().toJson(data, writer);
                return true;
            }} catch (IOException e) {{
                return false;
            }}
        }}
        Summary:
        This function writes a map of configuration data to a JSON file and returns whether the save operation was successful.
        END OF CASE

        Java function:
        {entry\['question']}
        Natural language:
```

Code Generation:

```
# NL->Java

Please implement the following Java function. End your answer with 'END OF CASE'.
Instruction:
Write a Java method that sets a `name` field to the provided parameter value.
Function:
public void setName(String name) {
    this.name = name;
}
END OF CASE
Instruction:
{entry\['Instruction']}
Function:
```

```
# NL->Python
Please implement the Python function based on the description. End your answer with 'END OF CASE'.

Description:
Write a procedure, clamp, which takes two integers, x and limit, and returns x if x is between -limit and limit, otherwise returns the nearest of -limit or limit.

Function:
def clamp(x, limit): 
    if x < -limit: 
        return -limit 
    elif x > limit: 
        return limit 
    else: 
        return x
END OF CASE

Description:
{entry\['question']}

Python function:

```

## Complete Result

Please refer to the [result](./result.md) for the complete experimental results (RQ1-RQ5).

## Pretrained Language Model

The Java data and Python data used for pretraining can be obtained from [CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net/blob/main/data/java.zip). The dataset for the Java->C# code translation task and Python->Java code translation task is available at [CodeTrans](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/data) and [AVATAR](https://huggingface.co/datasets/ziwenyd/AVATAR) respectively, and the dataset for the NL->Java code generation task and NL->Python code generation task can be found at [Concode](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code/dataset/concode) and [Text2Python](https://huggingface.co/datasets/gretelai/gretel-text-to-python-fintech-en-v1) respectively, and the dataset for the Java->NL code summarization task and Python->NL code summarization task can be found at [TL-codesum](https://github.com/xing-hu/TL-CodeSum) and [PySuma](https://huggingface.co/datasets/teven/code_docstring_corpus/tree/main/data) respectively.



#### RoBERTa-base

Run `pretrain.sh` to perform model pretraining. Note that you may need to modify the paths in the script to point to your specific dataset and model locations.

```shell
cd roberta/
bash pretrain.sh
```

Use the pretrained model for fine-tuning on downstream tasks and evaluate it on the test set:

```shell
bash run.sh
bash score.sh
```

Note: Ensure you have configured the paths in `run.sh` and `score.sh` correctly before running.

#### GPT2-small

Pre-training and fine-tuning code for different languages and different code-related tasks can be found in the `gpt2` directory.
We have updated the scripts to support command-line arguments for easier configuration. You can use the provided `run\_gpt2.sh` script as a starting point.

```shell
# Make sure to update paths in run\_gpt2.sh before running
bash run\_gpt2.sh
```

Or run the Python scripts directly with arguments:

**Pretraining:**

```shell
cd gpt2/python/code\_translation
python pretrain\_python2java.py \\
    --train\_file /path/to/train.jsonl \\
    --model\_name\_or\_path gpt2 \\
    --output\_dir /path/to/save/model
```

**Fine-tuning:**

```shell
python fine\_python2java.py \\
    --train\_file /path/to/train.jsonl \\
    --validation\_file /path/to/valid.jsonl \\
    --model\_name\_or\_path /path/to/pretrained/model \\
    --output\_dir /path/to/save/finetuned
```

**Inference:**

```shell
python infer\_python2java.py \\
    --model\_path /path/to/finetuned/model \\
    --test\_file\_path /path/to/test.jsonl \\
    --output\_file /path/to/output.jsonl
```

**Evaluation:**

```shell
python eval\_python2java.py \\
    --tokenizer\_path /path/to/finetuned/model \\
    --json\_file /path/to/output.jsonl
```

## Large Language Model

### Data construction

The Java and C# data used in StarCoder's pretraining can be obtained from [bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack), while the Java and C# data used in LLaMA's pretraining can be accessed via [bigquery](https://console.cloud.google.com/bigquery?ws=!1m4!1m3!3m2!1sbigquery-public-data!2sgithub_repos).

We provide `extract\_data.sh` as an example to run data extraction scripts. You need to provide the `tree-sitter` library path.

```shell
bash extract\_data.sh
```

Or run individual scripts:

```shell
cd extract\_data
# Extract unpaired data
python filter-unpaired.py --csharp\_file csharp.jsonl --java\_file java.jsonl --output\_dir ./unpaired --tree\_sitter\_lib ./build/my-languages.so

# Match unpaired data
python matched-unpaired.py --input\_dir ./unpaired --output\_file matched.jsonl --tree\_sitter\_lib ./build/my-languages.so

# Extract paired summary
python extract\_paired-summary.py --input\_file java.jsonl --output\_dir ./summary --tree\_sitter\_lib ./build/my-languages.so

# Extract paired generation
python extract-paired-generation.py --input\_file java.jsonl --output\_dir ./generation --tree\_sitter\_lib ./build/my-languages.so
```

We have provided samples in the [dataset](./dataset)

### Infer

The large models used for inference are obtained from [Starcoder](https://huggingface.co/bigcode/starcoderbase) and [Llama](https://huggingface.co/alexl83/LLaMA-33B-HF).

We have updated the scripts to support command-line arguments. You can use `run\_llama.sh` as a template.

```shell
# Make sure to update paths in run\_llama.sh
bash run\_llama.sh
```

**Inference:**

```shell
cd llama/python
python infer\_translation.py \\
    --model\_name\_or\_path /path/to/llama-model \\
    --input\_file /path/to/input.jsonl \\
    --output\_file /path/to/output.jsonl
```

**Evaluation:**

```shell
# Clean output first
python clean\_translate.py --input\_file /path/to/output.jsonl --output\_file /path/to/cleaned.jsonl

# Evaluate
python eval\_translate.py --tokenizer\_path /path/to/llama-model --input\_file /path/to/cleaned.jsonl --output\_dir /path/to/eval\_results
```

