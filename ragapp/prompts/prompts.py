from constants import dummy_full_list_of_devices_lower


retriever_system_prompt = """You are an assistant for answering questions. Use the retrieved context to respond. If the answer is unknown (no context provided), say so. Return the response in the provided structure, including the used context and answer."""

retriever_user_prompt = "Question: {question} \n\nContext: {context}"

device_extractor_system_prompt = """
< Role >
You are a medical device assistant.
</ Role >

< Instructions >
You have access to the following device list:  
{devices}

Users ask questions about these devices. Your primary task is to identify the exact device name from the list for accurate search filtering.
Return the exact device name only if it is clearly specified, including the series number if applicable. If the name is unclear or multiple devices share the same base name without a series number, do not specify device and explain your choice in reasoning section.
</ Instructions >
"""

device_extractor_system_prompt_formatted = device_extractor_system_prompt.format(
    devices=",".join(dummy_full_list_of_devices_lower),
)
