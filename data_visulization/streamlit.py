import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(layout="wide")
st.title("📄 智能数据查看器")

# --- 辅助函数 ---

def parse_nested_json(data):
    """
    递归遍历数据结构（字典或列表），尝试将所有字符串解析为JSON。
    这是实现“要求4”的核心。
    """
    if isinstance(data, dict):
        # 遍历字典的键值对
        for key, value in data.items():
            data[key] = parse_nested_json(value)
    elif isinstance(data, list):
        # 遍历列表的元素
        for i in range(len(data)):
            data[i] = parse_nested_json(data[i])
    elif isinstance(data, str):
        try:
            # 尝试将字符串加载为JSON
            loaded_json = json.loads(data)
            # 如果加载成功，递归地解析这个新加载的结构
            return parse_nested_json(loaded_json)
        except (json.JSONDecodeError, TypeError):
            # 如果不是有效的JSON字符串，保持原样
            return data
    
    # 返回非（字典、列表、字符串）类型的数据
    return data


@st.cache_data(show_spinner="正在加载数据...")
def load_data(file_path):
    """
    根据文件路径和后缀名加载数据，统一返回 list[dict] 格式。
    """
    try:
        p = Path(file_path)
        if not p.exists():
            st.error(f"文件未找到: {file_path}")
            return None

        ext = p.suffix.lower()
        if ext == '.parquet':
            df = pd.read_parquet(file_path)
        elif ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext == '.json':
            # 假设json是 [{}, {}] 或 { "0": {}, "1": {} } 格式
            df = pd.read_json(file_path, orient='records')
        elif ext == '.jsonl':
            df = pd.read_json(file_path, lines=True)
        else:
            st.error(f"不支持的文件类型: {ext} (仅支持 .parquet, .csv, .json, .jsonl)")
            return None
        
        # 将DataFrame转换为字典列表，这是最灵活的格式
        return df.to_dict('records')

    except Exception as e:
        st.error(f"加载文件时出错: {e}")
        return None

# --- Streamlit Session State 初始化 ---

# data: 存储加载的数据 (list[dict])
if 'data' not in st.session_state:
    st.session_state.data = None
# current_index: 存储当前查看的数据索引 (0-based)
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
# last_file_path: 用于检测文件路径是否变更
if 'last_file_path' not in st.session_state:
    st.session_state.last_file_path = ""

# --- 侧边栏 UI ---

st.sidebar.header("数据加载")
file_path = st.sidebar.text_input(
    "输入文件路径", 
    placeholder="/path/to/your/file.jsonl"
)

# 当用户输入了文件路径
if file_path:
    # 仅在文件路径改变时才重新加载数据
    if file_path != st.session_state.last_file_path:
        st.session_state.data = load_data(file_path)
        st.session_state.current_index = 0  # 重置索引
        st.session_state.last_file_path = file_path
else:
    # 如果清空了路径，也清空数据
    st.session_state.data = None
    st.session_state.current_index = 0
    st.session_state.last_file_path = ""


# --- 导航和数据展示 (仅在数据加载成功时显示) ---

if st.session_state.data:
    
    total_items = len(st.session_state.data)
    
    # 确保索引不会越界 (例如，在加载一个更短的新文件后)
    if st.session_state.current_index >= total_items:
        st.session_state.current_index = total_items - 1

    st.sidebar.divider()
    st.sidebar.header("数据导航")
    st.sidebar.write(f"总共: **{total_items}** 条数据")

    # --- 同步控件 (要求2 和 3) ---
    
    # 回调函数：当滑块变化时，更新 session_state.current_index
    def update_from_slider():
        # st.session_state.slider_nav 的值是 1-based
        st.session_state.current_index = st.session_state.slider_nav - 1

    # 回调函数：当数字输入变化时，更新 session_state.current_index
    def update_from_num_input():
        # st.session_state.num_input_nav 的值是 1-based
        st.session_state.current_index = st.session_state.num_input_nav - 1

    # 处理只有一条数据的边界情况 (要求3)
    is_disabled = (total_items <= 1)
    
    # UI 显示使用 1-based 索引，更符合直觉
    display_index = st.session_state.current_index + 1

    # 1. 进度条/滑块
    st.sidebar.slider(
        label="进度条跳转",
        min_value=1,
        max_value=total_items,
        value=display_index,
        key="slider_nav",
        on_change=update_from_slider,
        disabled=is_disabled
    )
    
    # 2. 序号输入框
    st.sidebar.number_input(
        label="序号跳转",
        min_value=1,
        max_value=total_items,
        value=display_index,
        key="num_input_nav",
        on_change=update_from_num_input,
        disabled=is_disabled
    )

    # --- 侧边栏原始数据展示 (要求5) ---
    st.sidebar.divider()
    if st.sidebar.button("在侧边栏展示原始JSON"):
        raw_item = st.session_state.data[st.session_state.current_index]
        st.sidebar.caption(f"第 {display_index} 条的原始数据")
        st.sidebar.json(raw_item)

    # --- 主页面 JSON 格式化展示 (要求4) ---
    st.subheader(f"数据索引: {display_index} / {total_items}")
    
    try:
        # 获取当前索引的原始数据
        raw_item = st.session_state.data[st.session_state.current_index]
        
        # 关键：创建一个深拷贝，避免修改st.session_state中的缓存数据
        # json.loads(json.dumps(x)) 是一个快速实现深拷贝的技巧
        item_to_format = json.loads(json.dumps(raw_item))
        
        # 应用递归的嵌套JSON解析
        formatted_item = parse_nested_json(item_to_format)
        
        # 使用 st.json 展示最终格式化的结果
        st.json(formatted_item)
        
    except Exception as e:
        st.error(f"格式化JSON时出错: {e}")
        st.write("展示原始数据作为备用:")
        st.json(st.session_state.data[st.session_state.current_index])

elif file_path and not st.session_state.data:
    # 路径已输入，但加载失败（错误信息已在load_data中显示）
    st.info("数据加载失败，请检查文件路径和文件内容。")
else:
    # 初始状态
    st.info("请在左侧侧边栏输入有效的文件路径以开始浏览数据。")