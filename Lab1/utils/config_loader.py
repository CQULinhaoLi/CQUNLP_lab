import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    加载配置文件，支持环境变量替换和YAML锚点引用
    """
    try:
        with open(Path(config_path).resolve(), 'r') as f:
            # 支持!env标签解析环境变量
            loader = yaml.SafeLoader
            loader.add_implicit_resolver(
                '!env',
                yaml.regex.Pattern(r'\${env:.*}'),
                None
            )
            loader.add_constructor('!env', lambda loader, node: 
                os.environ.get(node.value[6:-1], node.value[6:-1])
            )
            config = yaml.load(f, loader)
        
        # 验证必填项（示例）
        required_keys = ['train', 'model', 'data']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件缺少必填项: {key}")
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    except Exception as e:
        raise RuntimeError(f"配置加载失败: {str(e)}")