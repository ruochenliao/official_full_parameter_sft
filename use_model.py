from typing import Any


from transformers import AutoModelForCausalLM, AutoTokenizer

# online_model_name = "Qwen/Qwen3-0.6B"
local_model_name = "./sft_output/checkpoint-200"
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(local_model_name)
model = AutoModelForCausalLM.from_pretrained(
    local_model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "抽取出文本中的关键词：\n标题：人工神经网络在猕猴桃种类识别上的应用\n文本：在猕猴桃介电特性研究的基础上,将人工神经网络技术应用于猕猴桃的种类识别.该种类识别属于模式识别,其关键在于提取样品的特征参数,在获得特征参数的基础上,选取合适的网络通过训练来进行识别.猕猴桃种类识别的研究为自动化识别果品的种类、品种和新鲜等级等提供了一种新方法,为进一步研究果品介电特性与其内在品质的关系提供了一定的理论与实践基础."
messages = [
    {"role": "user", "content": prompt}
]

# 把 messages 拼接成字符串
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,  # 这里不分词，后面才分词
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)

print("text="+text)

# 模型只能接受数字张量作为输入，不能直接处理文本。这一行就是把人类可读的文字，翻译成模型能理解的张量。
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print("model_inputs=")
print(model_inputs)

# conduct text completion
generated_ids: Any = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
print("output_ids=")
print(output_ids)


# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
"""
未经过SFT训练的模型，输出的content如下：
content: 关键词：人工神经网络、猕猴桃、种类识别、模式识别、特征参数、果品、介电特性、品质、研究方法


经过SFT训练后的模型，输出的content如下：
content: 猕猴桃;人工神经网络;果品;品种;新鲜等级
"""

