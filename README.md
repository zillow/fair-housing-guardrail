# Fair Housing Guardrail
Fair Housing Guardrail is a tool designed to help housing providers, real estate professionals, and related organizations review compliance of textual content with fair housing and fair lending laws in the United States, specifically the federal Fair Housing Act and the Equal Credit Opportunity Act. By leveraging a combination of stop list and fine-tuned BERT-based classifier which offers low latency and ease of fine-tuning compared to LLMs, Fair Housing Guardrail identifies content that may potentially violate these laws, allowing users to make necessary adjustments to help maintain compliance.

## Overview
For a comprehensive overview of Fair Housing Guardrail, including its features, usage instructions, and implementation details, please visit our [blog](https://www.zillow.com/tech/navigating-fair-housing-guardrails-in-llms/).

In use cases which use an LLM, we recommend using Fair Housing Guardrail in combination with a fair housing specific prompt as we found that it yielded better coverage in violation detection and prevention. Refer to the ‘Prompt Engineering’ section [here](https://www.zillow.com/tech/navigating-fair-housing-guardrails-in-llms/) for more details.

## Features
**Compliance Detection**: Fair Housing Guardrail utilizes advanced natural language processing techniques to identify content that may violate fair housing and fair lending laws.

**Customizable Stop List**: The tool demonstrates incorporating a curated list of terms and phrases commonly associated with discriminatory language, helping to flag potentially non-compliant content. Users may modify the sample stop list to add phrases relevant for their scenario.

**Customizable BERT-based Classifier**: Fair Housing Guardrail employs a fine-tuned BERT-based classifier to analyze text and determine whether it aligns with fair housing and fair lending regulations. Users can adapt this framework to a different domain by providing their specific training data with compliant and non-compliant examples.

**Fair Housing Policy**: Specifically, the model was trained to address the risk of illegal “steering” in real estate. In the traditional brick and mortar sense, this is when a real estate agent takes into account the legally-protected characteristics of their client when determining which listings to show them. A well publicized example of this was documented in a three-year investigation by [Newsday](https://www.google.com/url?q=https://projects.newsday.com/long-island/real-estate-agents-investigation/&sa=D&source=docs&ust=1712163974278248&usg=AOvVaw3U1PPg4BEXVJm_kQ17UB5f), published in 2019. Extensive fair housing testing or “secret shopping” of real estate agents on Long Island, New York was conducted, finding that many would direct different clients to different neighborhoods, depending on the race or ethnicity of the client, and the racial and ethnic demographics of the neighborhood.  

**Fine-tuned model and labeled dataset**: Refer to ‘Contact Us’ section to request this.


## How It Works
Fair Housing Guardrail works by taking input text and running it through both the stop list and the fine-tuned BERT-based classifier. If the text contains any flagged terms from the stop list or if the classifier determines it to be non-compliant with the fair housing and fair lending guidelines that it was trained on, the tool will label the content accordingly, providing an opportunity to review and revise the content as necessary.
 
## Usage
After cloning the repo to a local directory, you can install the necessary dependencies with poetry.

Make sure you have Poetry installed. If not, you can install it by following the instructions at https://python-poetry.org/docs/#installation.

Navigate to the project directory that you cloned.

Run the following command to install the project dependencies using Poetry: 

`poetry install`

The project includes an `examples/` folder that contains two Jupyter notebooks:

**train_model**: This notebook uses the sample `train-config.yaml` file to load train and test datasets. Then, it runs training and plots the training and validation losses and saves the pretrained model.

**test_model**: This notebook uses the sample `test-config.yaml` file to load a test dataset. Ensure that paths to dataset and trained model are updated in test-config.yaml. Then, it runs predictions and returns the results. 

## Contributing
See the [Contributing](https://github.com/zillow/fair-housing-guardrail/blob/main/CONTRIBUTING.md) file for instructions on how to submit a PR.

## License
See the [License](https://github.com/zillow/fair-housing-guardrail/blob/main/LICENSE) file.

## Disclaimer
Fair Housing Guardrail is provided for informational purposes only and should not be considered legal advice. We recognize that users will interpret fair housing and fair lending requirements based on their own understanding and risk appetite, and are responsible for ensuring compliance with all applicable laws and regulations. 

## Contact Us
If you are interested in obtaining the training data and/or trained model, kindly contact us at 
[fair-housing-guardrail-oss-support@zillowgroup.com](mailto:fair-housing-guardrail-oss-support@zillowgroup.com). In your message, provide a brief paragraph outlining your intended use case and how you plan to utilize both the model and dataset. 
