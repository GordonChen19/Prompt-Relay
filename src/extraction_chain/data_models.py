from pydantic import BaseModel, Field

#For text-to-video generation
class AugmentedPrompt(BaseModel):
    global_prompt: str = Field(description = '''global_prompt: A detailed description of the overall video based on the input text prompt. This should include inferred details that are not explicitly mentioned in the text prompt.''')
    scene_prompts: list[str] = Field(description = '''scene_prompts: A list of detailed descriptions for each distinct scene or segment within the video, capturing the key elements and atmosphere of each part.''')
    mapping: dict[int, list[int]] = Field(description = '''mapping: A dictionary mapping scene indices to their corresponding local prompt indices, indicating which local prompts are associated with each scene.''')
    local_prompts: list[str] = Field(description = '''local_prompts: A list of detailed, realistic, movie-like descriptions of the sub-scenes or actions within the video, that evoke visuals with strong detail and composition cues.''')
    

