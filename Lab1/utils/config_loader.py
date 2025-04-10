import yaml
import re
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    env_var_pattern = re.compile(r'\${env:(.*?)}')

    def env_constructor(loader, node):
        value = node.value
        match = env_var_pattern.match(value)
        if match:
            var_name = match.group(1)
            return os.environ.get(var_name, f"${{{var_name}}}")
        return os.environ.get(node.value, node.value)

    try:
        with open(Path(config_path).resolve(), 'r') as f:
            loader = yaml.SafeLoader
            loader.add_constructor('!env', env_constructor)
            loader.add_implicit_resolver('!env', env_var_pattern, None)
            
            config = yaml.load(f, loader)
        
        # 关键修改：匹配你的配置键名
        required_keys = ['training', 'model', 'data']  # 原为'train'，现改为'training'
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件缺少必填项: {key}")
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    except Exception as e:
        raise RuntimeError(f"配置加载失败: {str(e)}")
