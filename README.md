# Fair Housing Guardrail
Fair Housing Guardrail is a tool designed to help real estate professionals and organizations ensure compliance with fair housing and fair lending laws in the United States. By leveraging a combination of stop list and fine-tuned BERT-based classifier, Fair Housing Guardrail identifies content that may potentially violate these laws, allowing users to make necessary adjustments to maintain compliance.

## Overview
For a comprehensive overview of Fair Housing Guardrail, including its features, usage instructions, and implementation details, please visit our [blog] (https://www.zillow.com/tech/navigating-fair-housing-guardrails-in-llms/) 

## Features
**Compliance Detection**: Fair Housing Guardrail utilizes advanced natural language processing techniques to identify content that may violate fair housing and fair lending laws.

**Customizable Stop List**: The tool demonstrates incorporating a curated list of terms and phrases commonly associated with discriminatory language, helping to flag potentially non-compliant content. Users may modify the sample stop list to add phrases relevant for their scenario.

**Customizable BERT-based Classifier**: Fair Housing Guardrail employs a fine-tuned BERT-based classifier to analyze text and determine whether it aligns with fair housing and fair lending regulations. Users can finetune the classifier to fit their specific needs and requirements on their own data, allowing for tailored compliance checks.

## How It Works
Fair Housing Guardrail works by taking input text and running it through both the stop list and the fine-tuned BERT-based classifier. If the text contains any flagged terms or if the classifier determines it to be non-compliant, the tool will label the content accordingly, providing an opportunity to review and revise the content as necessary.
 
## Usage
After cloning the repo to a local directory, you can install the necessary dependencies with poetry.

Make sure you have Poetry installed. If not, you can install it by following the instructions at https://python-poetry.org/docs/#installation.

Navigate to the project directory that you cloned.

Run the following command to install the project dependencies using Poetry: 

`poetry install`

The project includes an `examples/` folder that contains two Jupyter notebooks:

**test_model**: This notebook uses the sample `test-config.yaml` file to create a test dataset. Then, it runs predictions and returns the results.
**train_model**: This notebook uses the sample `train-config.yaml` file to create train and test datasets. Then, it runs training and plots the training and validation losses and saves the pretrained model.

# TODO: add usage description based on whether we release model or not

## Contributing
See the [Contributing](https://github.com/zillow/fair-housing-guardrail/blob/main/CONTRIBUTING.md) file for instructions on how to submit a PR.

## License
See the [License](https://github.com/zillow/fair-housing-guardrail/blob/main/LICENSE) file.

## Disclaimer
Fair Housing Guardrail is provided for informational purposes only and should not be considered legal advice. Users are responsible for ensuring compliance with all applicable laws and regulations.