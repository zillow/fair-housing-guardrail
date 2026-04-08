# Fair Housing Guardrail
Fair Housing Guardrail is a tool designed to help housing providers, real estate professionals, and related organizations review compliance of textual content with fair housing and fair lending laws in the United States, specifically the federal Fair Housing Act and the Equal Credit Opportunity Act. By leveraging a combination of stop list and fine-tuned BERT-based classifier which offers low latency and ease of fine-tuning compared to LLMs, Fair Housing Guardrail is intended to identify content that may potentially violate these laws, allowing users to make necessary adjustments to help maintain compliance.

## Overview
For a comprehensive overview of Fair Housing Guardrail, including its features, usage instructions, and implementation details, please visit our [blog](https://www.zillow.com/tech/navigating-fair-housing-guardrails-in-llms/)

In use cases which use an LLM, we recommend using Fair Housing Guardrail in combination with a fair housing specific prompt as we found that it yielded better coverage in violation detection and prevention. Refer to the 'Prompt Engineering' section [here](https://www.zillow.com/tech/navigating-fair-housing-guardrails-in-llms/) for more details. 

## Features
**Compliance Detection**: Fair Housing Guardrail utilizes advanced natural language processing techniques to identify content that may violate fair housing and fair lending laws.

**Customizable Stop List**: The tool demonstrates incorporating a curated list of terms and phrases commonly associated with discriminatory language, helping to flag potentially non-compliant content. Users may modify the sample stop list to add phrases relevant for their scenario.

**Customizable Classifier (Binary and Multi-Class)**:
- **Binary Model**: Uses a fine-tuned BERT-base-uncased model to classify text as either `compliant` or `non-compliant`.
- **Multi-Class Model**: Uses a fine-tuned RoBERTa-large model to classify text as `compliant` or into one of several specific non-compliant categories (e.g., `non-compliant-age`, `non-compliant-disability`, etc.). The full list of classes is present in `data/constants.py`.
- Users can adapt this framework to a different domain by providing their specific training data with compliant and non-compliant examples.

**Fair Housing Policy**: Specifically, the model was trained to address the risk of illegal “steering” in real estate. In the traditional brick and mortar sense, this is when a real estate agent takes into account the legally-protected characteristics of their client when determining which listings to show them. A well publicized example of this was documented in a three-year investigation by [Newsday](https://projects.newsday.com/long-island/real-estate-agents-investigation/), published in 2019. Extensive fair housing testing or “secret shopping” of real estate agents on Long Island, New York was conducted, finding that many would direct different clients to different neighborhoods, depending on the race or ethnicity of the client, and the racial and ethnic demographics of the neighborhood.  

**Fine-tuned model and labeled dataset**: Refer to the ‘Contact Us’ section to request this.


## How It Works
Fair Housing Guardrail works by taking input text and running it through both the stop list and a fine-tuned classifier. The classifier architecture and output depend on the mode:

- **Binary Classification (BERT-based):**
  - Uses a BERT-base-uncased model.
  - Predicts either `compliant` or `non-compliant` for each input.
  - Uses a sigmoid activation and a configurable threshold (default: 0.5) to determine the label.

- **Multi-Class Classification (RoBERTa-based):**
  - Uses a RoBERTa-large model.
  - Predicts one of several classes: `compliant` or a specific non-compliant category (e.g., `non-compliant-age`, `non-compliant-disability`, etc.).
  - Uses a softmax activation and selects the class with the highest probability (no threshold needed).

If the text contains any flagged terms from the stop list or if the classifier determines it to be non-compliant with the fair housing and fair lending guidelines that it was trained on, the tool will label the content accordingly, providing an opportunity to review and revise the content as necessary.

Note that this project was designed to provide low latency, standalone and easy-to-tune guardrails for Fair Housing and Fair Lending compliance of conversational or text based experiences. The fine-tuned model (provided upon request) is an ongoing work in progress and we will be iteratively improving its accuracy.  We invite you - the developers and AI practitioners - to help us identify gaps, submit improvements via pull requests as well as share your feedback directly with us via email at [fair-housing-guardrail-oss-support@zillowgroup.com](mailto:fair-housing-guardrail-oss-support@zillowgroup.com).
 
## Usage
After cloning the repo to a local directory, you can install the necessary dependencies with poetry.

Make sure you have Poetry installed. If not, you can install it by following the instructions at https://python-poetry.org/docs/#installation.

Navigate to the project directory that you cloned.

Run the following command to install the project dependencies using Poetry: 

`poetry install`

The project includes an `examples/` folder that contains Jupyter notebooks and config files for both **binary** and **multi-class** classification:

### Binary Classification (Compliant vs. Non-Compliant)
- **Model:** BERT-base-uncased
- **Config files:**
  - `examples/configs/train-config-binary.yaml`
  - `examples/configs/test-config-binary.yaml`
- **Sample data:**
  - `examples/datasets/sample_data_binary.jsonl`
- **How it works:**
  - The model predicts either `compliant` or `non-compliant` for each input.
  - The `fairhousing` section in the config allows you to set a threshold for the binary classifier (default: 0.5).

### Multi-Class Classification (Compliant + Multiple Non-Compliant Classes)
- **Model:** RoBERTa-large
- **Config files:**
  - `examples/configs/train-config-multiclass.yaml`
  - `examples/configs/test-config-multiclass.yaml`
- **Sample data:**
  - `examples/datasets/sample_data_multiclass.jsonl`
- **How it works:**
  - The model predicts one of several classes: `compliant` or a specific non-compliant category (e.g., `non-compliant-age`, `non-compliant-disability`, etc.).
  - No threshold is needed; the model uses softmax and selects the most likely class.

### Adding Your Own Data
- For **binary**: Provide a JSONL file with `label` values of `compliant` or `non-compliant`.
- For **multi-class**: Provide a JSONL file with `label` values matching the multi-class categories (see `sample_data_multiclass.jsonl`).
- Update the appropriate config file to point to your data file.

### Running the Notebooks
- Open `examples/train_model.ipynb` or `examples/test_model.ipynb` depending on your use case.
- Set the `BINARY` parameter at the top of the notebook to `True` for binary classification, or `False` for multi-class classification.
- Update the config file path if needed.
- Run the notebook cells to train or test the model.

## Contributing
See the [Contributing](https://github.com/zillow/fair-housing-guardrail/blob/main/CONTRIBUTING.md) file for instructions on how to submit a PR.

## License
See the [License](https://github.com/zillow/fair-housing-guardrail/blob/main/LICENSE) file.

## Disclaimer
Fair Housing Guardrail is provided for informational purposes only and should not be considered legal advice. We recognize that users will interpret fair housing and fair lending requirements based on their own understanding and risk appetite, and are responsible for ensuring compliance with all applicable laws and regulations. By using Fair Housing Guardrail, you acknowledge that you understand these risks and agree to use the software responsibly and at your own risk. See [Security](https://github.com/zillow/fair-housing-guardrail/blob/main/SECURITY.md)

## Contact Us
If you are interested in obtaining the training data and/or trained model, kindly contact us at 
[fair-housing-guardrail-oss-support@zillowgroup.com](mailto:fair-housing-guardrail-oss-support@zillowgroup.com). In your message, provide a brief paragraph outlining your intended use case and how you plan to utilize both the model and dataset. Information shared will be governed as described in the [Zillow Group Privacy Notice](https://www.zillowgroup.com/zg-privacy-policy/)
