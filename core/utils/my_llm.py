# llm_client.py
from __future__ import annotations

import random
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterable, List, Optional, Protocol, Union, Callable, Tuple

try:
    from loguru import logger
except Exception:  # 兜底到标准 logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("llm_client")

try:
    import tiktoken
except Exception:
    tiktoken = None  # 允许无 tiktoken 环境

from openai import OpenAI


# =========================
# 工具/协议/错误类型
# =========================

class CacheProtocol(Protocol):
    """最小缓存协议。"""
    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None: ...


class LLMError(Exception):
    """通用 LLM 异常基类。"""
    pass


class OutOfBalanceError(LLMError):
    """余额不足类错误。"""
    pass


class PostProcessError(LLMError):
    """post_process 处理阶段异常（不做内部重试）。"""
    pass


@dataclass(frozen=True)
class BackoffConfig:
    """重试退避策略配置。"""
    base_seconds: float = 1.0
    max_seconds: float = 30.0
    jitter: float = 0.1  # 相对抖动比例
    factor: float = 2.0  # 指数增长因子

    def sleep(self, attempt: int) -> None:
        delay = min(self.base_seconds * (self.factor ** attempt), self.max_seconds)
        jitter_val = delay * self.jitter * (random.random() * 2 - 1.0)
        time.sleep(max(0.0, delay + jitter_val))


@dataclass
class EndpointConfig:
    """单个端点（key/base_url/模型集）配置。"""
    api_key: str
    api_base_url: str
    candidate_models: List[str] = field(default_factory=lambda: ["gpt-4o-mini", "gpt-4o"])
    max_attempts: int = 5
    default_system_prompt: str = "You are a natural language processing expert, proficient in handling various natural semantic processing tasks."


@dataclass
class ClientConfig:
    """Client 层配置。"""
    temperature: float = 0.3
    top_p: Optional[float] = None
    stream: bool = False
    backoff: BackoffConfig = field(default_factory=BackoffConfig)
    cache: Optional[CacheProtocol] = None
    enable_cache_for_stream: bool = False  # 一般不建议对流式结果做缓存


# =========================
# Token 统计辅助
# =========================

def _token_len(text: str, model_hint: str = "gpt-4") -> int:
    """安全 token 估算：优先 tiktoken，兜底粗略估算。"""
    if not text:
        return 0
    if tiktoken is None:
        return max(1, int(len(text) / 4))  # ~4 chars/token
    try:
        enc = tiktoken.encoding_for_model(model_hint)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# =========================
# 基础请求
# =========================

def _request_llm(
    client: OpenAI,
    prompt: str,
    model: str = 'gpt-4o',
    system: str = "",
    stream: bool = False,
    **kwargs: Any,
) -> Union[str, Iterable[str]]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        stream=stream,
        **kwargs,
    )

    if not stream:
        if completion.choices and completion.choices[0].message:
            return completion.choices[0].message.content
        return None

    def generate() -> Generator[str, None, None]:
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
    return generate()


# =========================
# 单端点封装
# =========================

class SingleEndpointLLM:
    """
    - 候选模型优先/回退
    - 重试 + 指数退避（post_process 不做内部重试）
    - 缓存（仅存 RAW，key 绑定当前 model）
    - 流式采用 finalized streaming（先缓冲 RAW，再一次性输出处理后文本）
    - 语义：若多次 LLM 尝试 + 单次 post_process 都失败，最终 raise 错误
    """

    def __init__(self, ep: EndpointConfig, cfg: ClientConfig):
        self._ep = ep
        self._cfg = cfg
        self._client = OpenAI(api_key=ep.api_key, base_url=ep.api_base_url)

    @staticmethod
    def _hash_key(*parts: str) -> str:
        import hashlib as _hashlib
        h = _hashlib.md5("||".join(parts).encode()).hexdigest()
        return "LLM_" + h[-15:]

    def _build_cache_key(
        self,
        model: str,                      # 绑定当前实际使用的模型
        sys_msg: str,
        prompt: str,
        request_overrides: Dict[str, Any],
        cache_token: Optional[str],
    ) -> str:
        parts = [
            f"model:{model}",
            f"sys:{sys_msg or ''}",
            f"prompt:{prompt}",
            f"temp:{self._cfg.temperature}",
            f"top_p:{self._cfg.top_p}",
            f"ro:{sorted(request_overrides.items()) if request_overrides else ''}",
            f"stream:{self._cfg.stream}",
            f"ctok:{cache_token or ''}",
        ]
        return self._hash_key(*parts)

    def _apply_post_process(self, raw: str, post_process: Callable, **kwargs) -> str:
        """单次 post_process，不做内部重试。失败抛 PostProcessError。"""
        try:
            processed = post_process(raw, **kwargs)
            return processed
        except Exception as e:
            raise PostProcessError(f"post_process failed: {e}") from e

    def call_llm(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        stream: Optional[bool] = None,
        token_counter: Optional[Dict[str, Optional[int]]] = None,
        post_process: Optional[Callable[[str], str]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        cache_token: Optional[str] = None,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Returns:
            - 非流式：str
            - 流式：生成器（finalized streaming：一次性吐出处理后的文本）
        """
        sys_msg = self._ep.default_system_prompt if system is None else system
        do_stream = self._cfg.stream if stream is None else stream
        post_process = post_process or (lambda s: s.strip() if isinstance(s, str) else s)
        ro = request_overrides or {}

        if token_counter is not None:
            token_counter.clear()
            token_counter.update({
                "input_token": _token_len((sys_msg or "") + prompt, self._ep.candidate_models[0]),
                "output_token": None
            })

        allow_cache = (self._cfg.cache is not None) and (not do_stream or self._cfg.enable_cache_for_stream) and use_cache

        last_err: Optional[Exception] = None
        for attempt in range(self._ep.max_attempts):
            model = self._select_model(attempt)
            cache_key = self._build_cache_key(model, sys_msg or "", prompt, ro, cache_token)

            try:
                # 先尝试命中缓存（非流式）
                if not do_stream and allow_cache:
                    try:
                        cached_raw = self._cfg.cache.get(cache_key)
                        if cached_raw:
                            processed = self._apply_post_process(cached_raw, post_process, **kwargs)
                            if token_counter is not None:
                                token_counter["output_token"] = _token_len(cached_raw, model)
                            return processed
                    except PostProcessError as e:
                        # 本次 attempt 失败，进入下一次
                        last_err = e
                        logger.warning(f"[PP from cache failed] attempt={attempt}, model={model}: {e}")
                        if attempt < self._ep.max_attempts - 1:
                            self._cfg.backoff.sleep(attempt)
                            continue
                    except Exception as e:
                        # 缓存异常不终止；继续走实际请求
                        last_err = e
                        logger.warning(f"[Cache error ignored] attempt={attempt}, model={model}: {e}")

                if do_stream:
                    # finalized streaming：先拿 RAW，再统一 post_process，再一次性吐出
                    try:
                        return self._stream_call_finalized(
                            model=model,
                            prompt=prompt,
                            sys_msg=sys_msg or "",
                            token_counter=token_counter,
                            request_overrides=ro,
                            post_process=post_process,
                            cache_token=cache_token,
                            allow_cache=allow_cache,
                        )
                    except PostProcessError as e:
                        last_err = e
                        logger.warning(f"[PP from stream failed] attempt={attempt}, model={model}: {e}")
                        if attempt < self._ep.max_attempts - 1:
                            self._cfg.backoff.sleep(attempt)
                            continue
                else:
                    # 非流式：实际请求
                    raw = _request_llm(
                        client=self._client,
                        model=model,
                        prompt=prompt,
                        system=sys_msg or "",
                        stream=False,
                        temperature=self._cfg.temperature,
                        top_p=self._cfg.top_p,
                        **ro,
                    )
                    if not isinstance(raw, str):
                        raw = "" if raw is None else str(raw)

                    if allow_cache and raw:
                        try:
                            self._cfg.cache.set(cache_key, raw)
                        except Exception:
                            pass

                    try:
                        processed = self._apply_post_process(raw, post_process, **kwargs)
                    except PostProcessError as e:
                        last_err = e
                        logger.warning(f"[PP failed] attempt={attempt}, model={model}: {e}")
                        if attempt < self._ep.max_attempts - 1:
                            self._cfg.backoff.sleep(attempt)
                            continue
                        break

                    if token_counter is not None:
                        token_counter["output_token"] = _token_len(raw, model)
                    return processed

            except OutOfBalanceError:
                # 透传给上层聚合器，让其移除此端点
                raise
            except Exception as e:
                # 这次尝试其他错误，继续下一次
                last_err = e
                self._handle_error(attempt, e, model, self._client)
                if attempt < self._ep.max_attempts - 1:
                    self._cfg.backoff.sleep(attempt)
                    continue
                break

        # 所有尝试均失败
        assert last_err is not None
        raise last_err

    def _stream_call_finalized(
        self,
        model: str,
        prompt: str,
        sys_msg: str,
        token_counter: Optional[Dict[str, Optional[int]]],
        request_overrides: Dict[str, Any],
        post_process: Callable,
        cache_token: Optional[str],
        allow_cache: bool,
        kwargs
    ) -> Generator[str, None, None]:
        """流式（finalized）：先完整缓冲 RAW，再统一 post_process，最后一次性 yield 处理后文本。"""
        gen = _request_llm(
            client=self._client,
            model=model,
            prompt=prompt,
            system=sys_msg,
            stream=True,
            temperature=self._cfg.temperature,
            top_p=self._cfg.top_p,
            **request_overrides,
        )
        buf: List[str] = []
        try:
            for piece in gen:
                if piece:
                    buf.append(piece)
        finally:
            raw = "".join(buf)

        if allow_cache and raw:
            try:
                cache_key = self._build_cache_key(
                    model=model,
                    sys_msg=sys_msg,
                    prompt=prompt,
                    request_overrides=request_overrides,
                    cache_token=cache_token,
                )
                self._cfg.cache.set(cache_key, raw)
            except Exception:
                pass

        processed = self._apply_post_process(raw, post_process, **kwargs)

        if token_counter is not None:
            token_counter["output_token"] = _token_len(raw, model)

        def _iter() -> Generator[str, None, None]:
            yield processed
        return _iter()

    def _select_model(self, attempt: int) -> str:
        """回退策略：前 N-1 次用第一个，最后一次用最后一个。"""
        if attempt == self._ep.max_attempts - 1 and len(self._ep.candidate_models) > 1:
            return self._ep.candidate_models[-1]
        return self._ep.candidate_models[0]

    def _handle_error(self, attempt: int, err: Exception, model: str, client) -> None:
        """统一错误处理与判别（可扩展错误码/文案）。"""
        msg = str(err)
        tb = traceback.format_exc()
        if "30001" in msg or "余额不足" in msg or "insufficient_quota" in msg:
            logger.error(f"[OutOfBalance] model={model}, endpoint: [{client.api_key[:20]}], attempt={attempt}: {msg}")
            raise OutOfBalanceError(msg)

        if "429" in msg or "Rate limit" in msg or "Too Many Requests" in msg:
            logger.warning(f"[RateLimit] model={model}, endpoint: [{client.api_key[:20]}], attempt={attempt}: {msg}")
            time.sleep(5)
        elif any(code in msg for code in ("502", "503", "504", "Service Unavailable", "Bad Gateway", "Gateway Timeout")):
            logger.warning(f"[ServiceUnavailable] model={model}, endpoint: [{client.api_key[:20]}], attempt={attempt}: {msg}")
        else:
            logger.warning(f"[LLMError] model={model}, endpoint: [{client.api_key[:20]}], attempt={attempt}: {msg}\n{tb}")


# =========================
# 多端点聚合 Client
# =========================

class LLMClient:
    def __init__(self, endpoints: List[Dict[str, Any]], client_cfg: Optional[ClientConfig] = None):
        self._cfg = client_cfg or ClientConfig()
        self._endpoint_cfgs = list(endpoints)  # 可能会 pop
        self._engines: Dict[int, SingleEndpointLLM] = {}  # 延迟初始化
        logger.info(f"Configured endpoints: {len(endpoints)}")

    def _get_engine(self, idx: int) -> SingleEndpointLLM:
        if idx not in self._engines:
            self._engines[idx] = SingleEndpointLLM(EndpointConfig(**self._endpoint_cfgs[idx]), self._cfg)
        return self._engines[idx]

    def call_llm(self, prompt: str, **kwargs: Any):
        if not self._endpoint_cfgs:
            raise LLMError("没有可用的 LLM 端点")

        while self._endpoint_cfgs:
            idx = random.randint(0, len(self._endpoint_cfgs) - 1)
            engine = self._get_engine(idx)
            try:
                return engine.call_llm(prompt, **kwargs)
            except OutOfBalanceError:
                logger.warning(f"移除余额不足端点，剩余: {len(self._endpoint_cfgs) - 1}")
                self._endpoint_cfgs.pop(idx)
                self._engines.pop(idx, None)
                if not self._endpoint_cfgs:
                    raise OutOfBalanceError("所有端点余额不足")
            except Exception:
                # 其它错误直接上抛（包括多次尝试后仍失败）
                raise


# =========================
# 使用示例
# =========================

if __name__ == "__main__":
    import redis

    endpoints = [
        {
            "api_key": 'sk-xxxxx',  # <-- 替换你的 key
            "api_base_url": 'https://your-proxy-or-openai-endpoint/v1',
            "candidate_models": ['gpt-4o', "gpt-4o"],
            "max_attempts": 4,
        }
    ]

    cache = redis.Redis(host="127.0.0.1", port=6579, decode_responses=True)

    def my_post_process(s: str, **kwargs) -> str:
        # 硬编码的确定性后处理：失败就抛异常（不做内部重试）
        s = (s or "").strip()
        if not s:
            raise ValueError("empty output after strip")
        return s

    # 非流式
    client = LLMClient(endpoints=endpoints, client_cfg=ClientConfig(cache=cache, stream=False))
    counter: Dict[str, Optional[int]] = {}
    text = client.call_llm("用三句话介绍一下量子计算。", token_counter=counter, post_process=my_post_process)
    print("=== 非流式输出 ===")
    print(text)
    print("Token统计:", counter)

    # 流式（finalized）
    client_stream = LLMClient(
        endpoints=endpoints,
        client_cfg=ClientConfig(temperature=0.2, stream=True, cache=cache, enable_cache_for_stream=True)
    )
    scounter: Dict[str, Optional[int]] = {}
    print("\n=== 流式输出（finalized） ===")
    gen = client_stream.call_llm("请用要点列出人工智能的三大主要领域。", token_counter=scounter, post_process=my_post_process)
    for piece in gen:
        print(piece, end="", flush=True)
    print("\nToken统计:", scounter)
