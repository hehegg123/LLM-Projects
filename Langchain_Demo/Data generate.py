# Using AI to generate data

# advantage:
# 1. privacy and safe to use (no real data)
# 2. expand data for machine learning
# 3. flexibility (create specific or unique circumstance)
# 4. low cost compare to RL data
# 5. fast PROTOTYPE
# 6. controlled experiment
# 7. Sub solution when real life data is not available

from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_PREFIX, SYNTHETIC_FEW_SHOT_SUFFIX
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, FewShotPromptTemplate
import os

from pydantic import BaseModel

os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')

# 1. define data structure
class MedicalBilling(BaseModel):
    patient_id: int
    patient_name: str
    diagnosis_code: str
    procedure_code: str
    total_charge: float
    insurance_claim_amount: float

# 2. provide sample data to the model

examples = [
    {
        "example": "Patient ID: 123456, Patient Name: Sam, Diagnosis Code: J20.9, Procedure Code 20_i, Total_Charge: 500, Insurance_claim_amount: 400"
    },
    {
        "example": "Patient ID: 123421, Patient Name: Adam, Diagnosis Code: J13.1, Procedure Code 12_d, Total_Charge: 35400, Insurance_claim_amount: 30000"
    }
]

# 3. create prompt template
Gemini_template = PromptTemplate(input_variables= ['example'], template = "{example}")

prompt_template = FewShotPromptTemplate(
    prefix = SYNTHETIC_FEW_SHOT_PREFIX,
    suffix = SYNTHETIC_FEW_SHOT_SUFFIX,
    examples=examples,
    example_prompt = Gemini_template,
    input_variables = ['subject', 'extra']
)

generator_chain = prompt_template | model.with_structured_output(schema=MedicalBilling)

num_runs = 10
results = []
for i in range(num_runs):
    print(f"Generating run {i+1}...")
    try:
        # Pass the input variables as a dictionary to the invoke method
        response = generator_chain.invoke(
            {
                'subject': 'medical billing',
                'extra': 'name can be random, use more common names at the beginning.'
            }
        )
        results.append(response)
        print(f"Generated: {response}")
    except Exception as e:
        print(f"Error during generation for run {i+1}: {e}")


print(results)

# chain = create_data_generate
# on_chain(model)
#
# # generate data
# result = chain.invoke(
#     {
#         "fields": ['blue','yellow'],
#         "preferences": {}
#     }
# )
#
# print(result)