from langchain.chains import TransformChain
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from ExtractionChain.image_perception import chat_completion


def extraction_chain(input, data_model, prompt_template, reasoning_model):
    """
    input: user-provided prompt
    """

    # (1) using langchain's default message to enforce GPT to output structured info
    parser = PydanticOutputParser(pydantic_object=data_model)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input"],
        partial_variables={"format_instructions": parser.get_format_instructions()})
    
    # (2) here is calling GPT

    # puts the 1) input (user-provided input), 2) system message, 3) format instructions into one single string
    prompt_str = prompt.invoke({"input":input}).to_string()

    response = chat_completion(prompt_str, reasoning_model)
    
    # (3) this line takes the plain text output into json dict
    # fix_parser = OutputFixingParser(parser=parser, retry_chain=fix_chain, max_retries=1)
    return parser.invoke(response).dict()