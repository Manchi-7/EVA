# 批量运行本地模型作为victim的collection实验

# UI-TARS-1.5-7B（最小模型，推荐先运行测试）
python run_collection.py --attacker gpt-5-nano --victim ui-tars-1.5-7b --evolution

# Qwen2.5-VL-7B-Instruct
python run_collection.py --attacker gpt-5-nano --victim qwen2.5-vl-7b-instruct --evolution

# Qwen2-VL-7B  
python run_collection.py --attacker gpt-5-nano --victim qwen2-vl-7b --evolution

# OS-Atlas-Pro-7B
python run_collection.py --attacker gpt-5-nano --victim os-atlas-pro-7b --evolution

# GUI-Owl-7B
python run_collection.py --attacker gpt-5-nano --victim gui-owl-7b --evolution
