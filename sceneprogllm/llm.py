import os
import time
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel, create_model
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from .template import SceneProgTemplate
from .image_helper import ImageHelper
from .text2x import text2img, text2speech, text2embeddings
from .tracker import UsageTracker

class ListResponse(BaseModel):
    response: list[str]

class DefaultJsonResponse(BaseModel):
    response: str

class LLM:
    def __init__(self,
                 system_desc="You are a helpful assistant.",
                 response_format="text",
                 response_params=None,
                 model_name='gpt-5-nano',
                 reasoning_effort="medium",
                 seed=124,
                 temperature=0.8,
                 name=None,
                 tracker=None,
                 ):
        assert response_format in ['text', 'list', 'code', 'json', 'pydantic','image','speech','embedding'], "Invalid response format, must be one of 'text', 'list', 'code', 'json', 'image', 'pydantic', 'speech', 'embedding'"
        self.response_format = response_format
        self.response_params = response_params

        if self.response_format == "json":
            if self.response_params is not None:
                self.json_keys = [x for x in self.response_params.items()]
            else:
                raise ValueError("json_keys must be provided when response_format is 'json'")
        
        self.model_name = model_name
        self.system_desc = system_desc
        self.temperature = temperature
        self.seed = seed
        self.model = ChatOpenAI(model_name=model_name, reasoning_effort=reasoning_effort, api_key=os.getenv("OPENAI_API_KEY"),seed=self.seed, temperature=self.temperature)
        self.last_usage = None
        self.name = name or response_format
        self.tracker = tracker

    def set_system_desc(self, system_desc_keys=None):
        if '$' in self.system_desc and system_desc_keys is None:
            raise ValueError("system_desc_keys must be provided when system_desc contains placeholders. Detected $ in system_desc. Is that a mistake?")
        
        if system_desc_keys is not None:
            system_desc = SceneProgTemplate.format(self.system_desc, system_desc_keys)
            assert '$' not in system_desc, "Incomplete set of system_desc_keys. Please provide all keys to fill the placeholders."

        else:
            system_desc = self.system_desc

        ## sanitize the system description
        system_desc = system_desc.replace('{', '{{').replace('}', '}}')

        self.image_helper = ImageHelper(system_desc)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_desc),
            ("human", "{input}")
        ])
        self.msg_header = f"""
"system": "{system_desc}",
"""
        return system_desc
    
    def __call__(self, query, image_paths=None, pydantic_object=None, system_desc_keys=None):
        system_desc = self.set_system_desc(system_desc_keys)
        # sanitize the query and system description
        try:
            query = query.replace('{', '{{').replace('}', '}}')
        except:
            pass

        if self.response_format == "image":
            img, usage = text2img(
                text=query,
                image_paths=image_paths,
                **(self.response_params or {})
            )
            self.last_usage = usage
            if self.tracker is not None:
                self.tracker.record(self.last_usage, self.name, self.response_format)
            return img
        elif self.response_format == "speech":
            audio, usage = text2speech(
                text=query,
                **(self.response_params or {})
            )
            self.last_usage = usage
            if self.tracker is not None:
                self.tracker.record(self.last_usage, self.name, self.response_format)
            return audio
        elif self.response_format == "embedding":
            embedding, usage = text2embeddings(
                texts=query,
                **(self.response_params or {})
            )
            self.last_usage = usage
            if self.tracker is not None:
                self.tracker.record(self.last_usage, self.name, self.response_format)
            return embedding

        if self.response_format == "pydantic":
            assert pydantic_object is not None, "pydantic_object must be provided when response_format is 'pydantic'"
        elif self.response_format == "list":
            pydantic_object = ListResponse
        elif self.response_format == "json":
            if self.json_keys is not None:
                CustomJSONModel = create_model(
                    'CustomJSONModel',
                    **{key: (type, ...) for key, type in self.json_keys}
                )
                pydantic_object = CustomJSONModel
            else:
                pydantic_object = DefaultJsonResponse

        full_prompt = self.msg_header + f"""
"human": "{query}",
"""
        if pydantic_object is not None:
            parser = PydanticOutputParser(pydantic_object=pydantic_object)
            self.prompt_template = PromptTemplate(
                template="Answer the user query.\n{format_instructions}\n{input}\n",
                input_variables=["input"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
        else:
            parser = StrOutputParser()

        if self.response_format == "code":
            full_prompt += """Return only python code in Markdown format, e.g.:
```python
....
```"""

        if image_paths is not None:
            if pydantic_object is not None:
                format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
            else:
                format_instructions = "Responsed as plain text."

            self.prompt_template = ChatPromptTemplate.from_messages(
                messages = self.image_helper.prepare_image_prompt_template(image_paths, format_instructions),
            )
            chain = self.prompt_template | self.model | parser
            t0 = time.time()
            with get_openai_callback() as cb:
                result = self.image_helper.invoke_image_prompt_template(chain, full_prompt, image_paths)
            self.last_usage = {
                'prompt_tokens': cb.prompt_tokens,
                'completion_tokens': cb.completion_tokens,
                'total_tokens': cb.total_tokens,
                'cost_usd': cb.total_cost,
                'latency_s': round(time.time() - t0, 4),
            }
            if self.tracker is not None:
                self.tracker.record(self.last_usage, self.name, self.response_format)
        else:
            chain = self.prompt_template | self.model | parser
            t0 = time.time()
            with get_openai_callback() as cb:
                result = chain.invoke({"input": full_prompt})
            self.last_usage = {
                'prompt_tokens': cb.prompt_tokens,
                'completion_tokens': cb.completion_tokens,
                'total_tokens': cb.total_tokens,
                'cost_usd': cb.total_cost,
                'latency_s': round(time.time() - t0, 4),
            }
            if self.tracker is not None:
                self.tracker.record(self.last_usage, self.name, self.response_format)
        
        if self.response_format == "code":
            result = self._sanitize_output(result)

        if self.response_format == "json":
            result = result.model_dump()
            tmp = {}
            for key, type in self.json_keys:
                if type == 'bool':
                    tmp[key] = bool(result[key])
                elif type == 'str':
                    tmp[key] = str(result[key])
                elif type == 'int':
                    tmp[key] = int(result[key])
                elif type == 'float':
                    tmp[key] = float(result[key])
                else:
                    raise ValueError(f"Unsupported type: {type}")
            result = tmp
        if self.response_format == "list":
            result = result.response
        return result
    
    def _sanitize_output(self, text: str):
        _, after = text.split("```python")
        return after.split("```")[0]