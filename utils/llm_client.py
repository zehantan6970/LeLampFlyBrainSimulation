# -*- coding: utf-8 -*-
"""L0 容错场景引擎：DeepSeek-V3 -> Qwen-Max -> 本地 Mock 三级降级。

契约：返回 dict，含 "scenario" 与 "timeline"（事件列表），字段见 mock_scenarios.json。
解析失败 / 超时 / 网络错误 / key 缺失 均自动降级，绝不抛出中断仿真。
"""
import json
import os
import time

from dotenv import load_dotenv

from configs.default_config import (
    LLM_TIMEOUT_S, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    QWEN_BASE_URL, QWEN_MODEL, MOCK_SCENARIO_PATH,
)

SYSTEM_PROMPT = """你是一个智能台灯 (LeLamp) 仿真系统的场景生成器。
根据场景描述生成时序 JSON 驱动数据。

果蝇内部状态 (fruit_fly_state) 必须是以下 6 种之一：
"Approach_Reward", "Fear_Arousal", "Vigilance_Defense",
"Exploration_Seeking", "Grooming_Rest", "Emotional_Comfort"

严格输出如下结构的 JSON（不要 Markdown 代码块标记）：
{
  "scenario": "场景名称",
  "timeline": [
    {
      "timestamp": 0.0,
      "duration_s": 6.0,
      "user_event": "用户动作描述",
      "fruit_fly_state": "Emotional_Comfort",
      "neuromodulation": {"dopamine_DA": 0.3, "octopamine_OA": 0.1, "serotonin_5HT": 0.9},
      "environmental_obstacles": [{"id": "water_cup", "pos": [0.20, 0.10, 0.06], "size": [0.07, 0.07, 0.12]}],
      "output_response": {"speak_text": "I am here for you.", "led_rgb": [1.0, 0.6, 0.8], "led_mode": "slow_breathing"}
    }
  ]
}
生成 3-5 个时间片段，总时长 20-40 秒。"""

REQUIRED_EVENT_KEYS = {"timestamp", "fruit_fly_state", "neuromodulation"}


def _validate(data):
    if not isinstance(data, dict) or "timeline" not in data:
        raise ValueError("missing timeline")
    for ev in data["timeline"]:
        if not REQUIRED_EVENT_KEYS.issubset(ev.keys()):
            raise ValueError(f"event missing keys: {ev.keys()}")
    return data


def _call_openai_compatible(base_url, model, api_key, scene_description):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=LLM_TIMEOUT_S)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请为以下场景生成仿真事件序列：{scene_description}"},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    return _validate(json.loads(resp.choices[0].message.content))


def load_mock_scenarios(project_root):
    path = os.path.join(project_root, MOCK_SCENARIO_PATH)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_mock(scenarios, scene_description=""):
    """按关键词粗匹配 mock 场景，默认返回第一条。"""
    text = scene_description.lower()
    for sc in scenarios:
        for kw in sc.get("keywords", []):
            if kw.lower() in text:
                return {"scenario": sc["scenario"], "timeline": sc["timeline"]}
    sc = scenarios[0]
    return {"scenario": sc["scenario"], "timeline": sc["timeline"]}


def generate_scenario(scene_description, project_root, force_mock=False, verbose=True):
    """三级降级链。返回 (data, source)，source in {deepseek, qwen, mock}。"""
    load_dotenv(os.path.join(project_root, ".env"))

    if not force_mock:
        chain = [
            ("deepseek", DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, os.getenv("DEEPSEEK_API_KEY")),
            ("qwen", QWEN_BASE_URL, QWEN_MODEL, os.getenv("DASHSCOPE_API_KEY")),
        ]
        for name, url, model, key in chain:
            if not key:
                if verbose:
                    print(f"[llm] {name}: 未配置 key，跳过")
                continue
            t0 = time.time()
            try:
                data = _call_openai_compatible(url, model, key, scene_description)
                if verbose:
                    print(f"[llm] {name} 成功，耗时 {time.time() - t0:.2f}s")
                return data, name
            except Exception as e:  # noqa: BLE001 - 容错链设计：任何失败都降级
                if verbose:
                    print(f"[llm] {name} 失败（{type(e).__name__}: {e}），降级")

    scenarios = load_mock_scenarios(project_root)
    if verbose:
        print("[llm] 使用本地 Mock 事件流")
    return pick_mock(scenarios, scene_description), "mock"
