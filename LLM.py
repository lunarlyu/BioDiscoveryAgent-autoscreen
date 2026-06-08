""" This file contains the code for calling all LLM APIs. """

import os
from functools import partial
from pathlib import Path
import tiktoken
# from schema import TooLongPromptError, LLMError

enc = tiktoken.get_encoding("cl100k_base")
REPO_ROOT = Path(__file__).resolve().parent
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_PROVIDER = os.environ.get("BIODISCOVERY_LLM_PROVIDER", "openrouter").lower()
OPENROUTER_DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL")
HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"

try:
    auth = None
    service = None
    if OPENROUTER_PROVIDER != "openrouter":
        from helm.common.authentication import Authentication
        from helm.common.request import Request, RequestResult
        from helm.proxy.accounts import Account
        from helm.proxy.services.remote_service import RemoteService
        # setup CRFM API
        auth = Authentication(api_key=open("crfm_api_key.txt").read().strip())
        service = RemoteService("https://crfm-models.stanford.edu")
        account: Account = service.get_account(auth)
except Exception as e:
    if OPENROUTER_PROVIDER != "openrouter":
        print(e)
        print("Could not load CRFM API key crfm_api_key.txt.")

try:   
    import anthropic
    HUMAN_PROMPT = getattr(anthropic, "HUMAN_PROMPT", HUMAN_PROMPT)
    AI_PROMPT = getattr(anthropic, "AI_PROMPT", AI_PROMPT)
    anthropic_client = None
    if OPENROUTER_PROVIDER != "openrouter":
        #setup anthropic API key
        anthropic_client = anthropic.Anthropic(api_key=open("claude_api_key.txt").read().strip())
except Exception as e:
    if OPENROUTER_PROVIDER != "openrouter":
        print(e)
        print("Could not load anthropic API key claude_api_key.txt.")

try:
    import openai
    from openai import OpenAI
    client = None
    if OPENROUTER_PROVIDER != "openrouter":
        try:
            organization, api_key  =  open("openai_api_key.txt").read().strip().split(":")    
            os.environ["OPENAI_API_KEY"] = api_key 
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except Exception:
            client = None
except Exception as e:
    print(e)
    print("Could not import OpenAI client.")


def read_openrouter_api_key():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    key_file = os.environ.get("OPENROUTER_API_KEY_FILE")
    candidate_files = []
    if key_file:
        candidate_files.append(Path(key_file).expanduser())
    candidate_files.extend([REPO_ROOT / ".openrouter_api_key", Path.home() / ".openrouter_api_key"])

    for candidate_file in candidate_files:
        if candidate_file.exists():
            return candidate_file.read_text().strip()
    return None


def get_openrouter_client():
    api_key = read_openrouter_api_key()
    if not api_key:
        raise ValueError(
            "Set OPENROUTER_API_KEY, OPENROUTER_API_KEY_FILE, "
            "or save the key in BioDiscoveryAgent/.openrouter_api_key or ~/.openrouter_api_key."
        )

    default_headers = {}
    site_url = os.environ.get("OPENROUTER_SITE_URL")
    app_name = os.environ.get("OPENROUTER_APP_NAME", "BioDiscoveryAgent")
    if site_url:
        default_headers["HTTP-Referer"] = site_url
    if app_name:
        default_headers["X-Title"] = app_name

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=default_headers or None,
    )


def to_openrouter_model(model):
    if OPENROUTER_DEFAULT_MODEL:
        return OPENROUTER_DEFAULT_MODEL
    if "/" in model:
        return model

    model_map = {
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4": "openai/gpt-4",
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
        "claude-3-5-sonnet-20240620": "anthropic/claude-3.5-sonnet",
        "claude-3-5-sonnet": "anthropic/claude-3.5-sonnet",
        "claude-3-opus-20240229": "anthropic/claude-3-opus",
        "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet",
        "claude-3-haiku-20240307": "anthropic/claude-3-haiku",
    }
    if model in model_map:
        return model_map[model]
    if model.startswith("claude"):
        return f"anthropic/{model}"
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return f"openai/{model}"
    return model


def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample):
    """ Log the prompt and completion to a file."""
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}")
        num_prompt_tokens = len(enc.encode(f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}"))
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        num_sample_tokens = len(enc.encode(completion))
        f.write("\n===================tokens=====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")


def complete_text_claude(prompt, stop_sequences=None, model="claude-v1", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, **kwargs):
    """ Call the Claude API to complete a prompt."""
    global anthropic_client
    if anthropic_client is None:
        anthropic_client = anthropic.Anthropic(api_key=open("claude_api_key.txt").read().strip())

    if stop_sequences is None:
        stop_sequences = [HUMAN_PROMPT]
    ai_prompt = AI_PROMPT
    if "ai_prompt" in kwargs is not None:
        ai_prompt = kwargs["ai_prompt"]
        del kwargs["ai_prompt"]
    # model = "claude-2"
    if model.startswith("claude-3"):
        messages = [
            {'role': 'user', 'content': f"{HUMAN_PROMPT} {prompt}"}
        ]
        rsp = anthropic_client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens_to_sample
        )
        completion = rsp.content[0].text
        if log_file is not None:
            log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
        return completion
    try:
        rsp = anthropic_client.completions.create(
            prompt=f"{HUMAN_PROMPT} {prompt} {ai_prompt}",
            stop_sequences=stop_sequences,
            model=model,
            temperature=temperature,
            max_tokens_to_sample=max_tokens_to_sample,
            **kwargs
        )
    except anthropic.APIStatusError as e:
        print(e)
        exit()
        raise TooLongPromptError()
    except Exception as e:
        exit()
        raise LLMError(e)

    completion = rsp.completion
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion


def get_embedding_crfm(text, model="openai/gpt-4-0314"):
    global auth, service
    if auth is None or service is None:
        from helm.common.authentication import Authentication
        from helm.common.request import Request
        from helm.proxy.services.remote_service import RemoteService
        auth = Authentication(api_key=open("crfm_api_key.txt").read().strip())
        service = RemoteService("https://crfm-models.stanford.edu")
    request = Request(model="openai/text-similarity-ada-001", prompt=text, embedding=True)
    request_result: RequestResult = service.make_request(auth, request)
    return request_result.embedding 

def complete_text_crfm(prompt=None, stop_sequences = None, model="openai/gpt-4-0314",  max_tokens_to_sample=2000, temperature = 0.5, log_file=None, messages = None, **kwargs):
    global auth, service
    if auth is None or service is None:
        from helm.common.authentication import Authentication
        from helm.common.request import Request, RequestResult
        from helm.proxy.services.remote_service import RemoteService
        auth = Authentication(api_key=open("crfm_api_key.txt").read().strip())
        service = RemoteService("https://crfm-models.stanford.edu")

    random = log_file
    if messages:
        request = Request(
                prompt=prompt, 
                messages=messages,
                model=model, 
                stop_sequences=stop_sequences,
                temperature = temperature,
                max_tokens = max_tokens_to_sample,
                random = random
            )
    else:
        print("model", model)
        print("max_tokens", max_tokens_to_sample)
        request = Request(
                prompt=prompt, 
                model=model, 
                stop_sequences=stop_sequences,
                temperature = temperature,
                max_tokens = max_tokens_to_sample,
                random = random
        )

    try:      
        request_result: RequestResult = service.make_request(auth, request)
    except Exception as e:
        # probably too long prompt
        print(e)
        exit()
        # raise TooLongPromptError()

    if request_result.success == False:
        print(request.error)
        # raise LLMError(request.error)
    completion = request_result.completions[0].text
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion


def complete_text_openai(prompt, stop_sequences=[], model="gpt-3.5-turbo", max_tokens_to_sample=2000, temperature=0.5, log_file=None, **kwargs):

    """ Call the OpenAI API to complete a prompt."""
    global client
    if client is None:
        organization, api_key  =  open("openai_api_key.txt").read().strip().split(":")
        os.environ["OPENAI_API_KEY"] = api_key
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    raw_request = {
          "model": model,
        #   "temperature": temperature,
        #   "max_completion_tokens": max_tokens_to_sample,
        #   "stop": stop_sequences or None,  # API doesn't like empty list
          **kwargs
    }
    if model.startswith("gpt-3.5") or model.startswith("gpt-4") or model.startswith("o1"):
        # Requires openai==1.42.0
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(**{"messages": messages,**raw_request})
        completion = response.choices[0].message.content
    else:
        response = client.completions.create(**{"prompt": prompt,**raw_request})
        completion = response.choices[0].text
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion

def complete_text_openrouter(prompt, stop_sequences=["Observation:"], model="gpt-4o", max_tokens_to_sample=2000, temperature=0.5, log_file=None, messages=None, **kwargs):
    """Call OpenRouter's OpenAI-compatible chat completion API."""
    openrouter_model = to_openrouter_model(model)
    openrouter_client = get_openrouter_client()

    request = {
        "model": openrouter_model,
        "messages": messages or [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens_to_sample,
    }
    if stop_sequences:
        request["stop"] = stop_sequences
    request.update(kwargs)
    request.pop("ai_prompt", None)

    response = openrouter_client.chat.completions.create(**request)
    completion = response.choices[0].message.content
    if log_file is not None:
        log_to_file(log_file, prompt or "", completion, openrouter_model, max_tokens_to_sample)
    return completion

def complete_text(prompt, log_file, model, **kwargs):
    """ Complete text using the specified model with appropriate API. """

    provider = os.environ.get("BIODISCOVERY_LLM_PROVIDER", OPENROUTER_PROVIDER).lower()
    if provider == "openrouter":
        completion = complete_text_openrouter(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    elif model.startswith("claude"):
        # use anthropic API
        completion = complete_text_claude(prompt, stop_sequences=[HUMAN_PROMPT, "Observation:"], log_file=log_file, model=model, **kwargs)
    elif "/" in model:
        # use CRFM API since this specifies organization like "openai/..."
        completion = complete_text_crfm(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    else:
        # use OpenAI API
        completion = complete_text_openai(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    return completion

# specify fast models for summarization etc
FAST_MODEL = "claude-v1"
def complete_text_fast(prompt, **kwargs):
    return complete_text(prompt = prompt, model = FAST_MODEL, temperature =0.01, **kwargs)
