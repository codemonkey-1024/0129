from typing import List, Tuple
import tiktoken
import re
class TextSplitter:
    def __init__(self, model_name: str = 'gpt-3.5-turbo'):
        self.encoding = tiktoken.encoding_for_model(model_name)

    def split_text(self, text: str,
                   overlap: int = 16,
                   max_chunk_size: int = 128,
                   min_chunk_size: int = 100,
                   padding: str = " ...") -> List[str]:
        tokens = self.encoding.encode(text)

        step_size = max_chunk_size - overlap
        pos = 0
        chunks = []

        while pos < len(tokens):
            end_pos = pos + max_chunk_size

            if end_pos >= len(tokens):
                chunk = tokens[pos:len(tokens)]
                if len(chunk) < min_chunk_size and chunks:
                    chunks[-1].extend(chunk)
                else:
                    chunks.append(chunk)
                break
            else:
                chunk = tokens[pos:end_pos]
                chunks.append(chunk)
                pos += step_size

        texts = [self.encoding.decode(chunk) for chunk in chunks]

        padded_texts = []
        num_chunks = len(texts)

        if num_chunks <= 1:
            return texts

        for i, chunk_text in enumerate(texts):
            if i == 0:
                padded_chunk = chunk_text + padding
            elif i == num_chunks - 1:
                padded_chunk = padding + chunk_text
            else:
                padded_chunk = padding + chunk_text + padding
            padded_texts.append(padded_chunk)
        return padded_texts



class SemanticTextSplitter:
    def __init__(self, model_name='gpt-3.5-turbo'):
        self.encoding = tiktoken.encoding_for_model(model_name)

    def split_text(self, text: str, max_tokens=128, overlap=16):
        sentences = re.split(r'(?<=[.!?。！？\n])\s+', text)
        chunks, current_chunk = [], []
        current_length = 0

        for sent in sentences:
            sent_len = len(self.encoding.encode(sent))
            if current_length + sent_len > max_tokens:
                chunks.append(" ".join(current_chunk))
                overlap_tokens = self.encoding.encode(" ".join(current_chunk))[-overlap:]
                overlap_text = self.encoding.decode(overlap_tokens)
                current_chunk = [overlap_text, sent]
                current_length = len(self.encoding.encode(overlap_text + sent))
            else:
                current_chunk.append(sent)
                current_length += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks


class SemanticTextSplitterV2:
    """
    句子优先 + 动态打包 + 兜底再切分 + 事后合并 的语义切分器。

    关键参数：
        max_tokens:   单块 token 上限，硬约束（例如 256/512）
        min_tokens:   单块 token 下限，鼓励块不要太短（例如 96/128）
        target_tokens: 贪心打包时的“理想长度”（介于 min 和 max 之间）
        overlap_tokens: 相邻块之间的 token 级重叠，增强上下文连续性
        model_name:   tiktoken 模型名（决定分词方式）

    设计要点：
    1) 先段落再句子，尽量保持语义边界。
    2) 如果某一句本身超长：优先按次级标点分，再不行则 token 级切分兜底。
    3) 贪心装箱：尽量贴近 target_tokens，但绝不超过 max_tokens。
    4) 事后合并：将过小的块优先并入前块；不行再尝试并入后块。
    5) 重叠控制：给下一块预置上一块的尾部 overlap_tokens（必要时裁剪以不超过 max）。
    """

    # 句子终止标点（中英混合）；含换行。尽量覆盖常见场景
    SENT_END_REGEX = r'(?<=[。！？!?;\.;:：\n\r])'
    # 次级切分（当一句太长时再细分），优先中文逗号/顿号/分号/英文逗号分号
    SUB_SENT_SPLITTER = r'([，、；;,:：])'

    def __init__(
        self,
        max_tokens: int = 512,
        min_tokens: int = 48,
        target_tokens: int = 256,
        overlap_tokens: int = 0,
        model_name: str = "gpt-4",
    ):
        assert min_tokens < target_tokens <= max_tokens, "需满足 min < target <= max"
        assert 0 <= overlap_tokens < min_tokens, "overlap_tokens 应小于 min_tokens，避免产生过小有效内容"
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.encoding_for_model(model_name)

    # ---------- 基础编码工具 ----------
    def _encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def _decode(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)

    def _len(self, text: str) -> int:
        return len(self._encode(text))

    # ---------- 文本分段/分句 ----------
    def _split_paragraphs(self, text: str) -> List[str]:
        # 按空行拆段，保留段内换行
        parts = re.split(r'\n\s*\n', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _split_sentences(self, paragraph: str) -> List[str]:
        # 基于句末标点切分；再清洗空白
        # 让标点保留在句子末尾，避免丢失语气
        pieces = re.split(self.SENT_END_REGEX, paragraph)
        # re.split 会保留分隔符后的空字符串，需要清洗
        sents = []
        buf = ""
        for piece in pieces:
            if not piece:
                continue
            buf += piece
            # 若 piece 以句末边界结束，则结束一句
            if re.search(self.SENT_END_REGEX + r'$', piece):
                sents.append(buf.strip())
                buf = ""
        if buf.strip():
            sents.append(buf.strip())
        # 兜底：若上述逻辑因正则差异导致没切开，就退化为按中文/英文常见断句
        if len(sents) <= 1:
            sents = re.split(r'(?<=[。！？!?])', paragraph)
            sents = [s.strip() for s in sents if s.strip()]
        return sents

    # ---------- 过长句子的再切分 ----------
    def _split_overlong_sentence(self, sent: str) -> List[str]:
        """当单句超过 max_tokens：先按次级标点进一步切；再不行转 token 级兜底"""
        if self._len(sent) <= self.max_tokens:
            return [sent]

        # 1) 优先按次级标点切
        parts = re.split(self.SUB_SENT_SPLITTER, sent)
        # 重新拼合：保持分隔符在前一片尾部
        merged = []
        cur = ""
        for p in parts:
            if re.match(self.SUB_SENT_SPLITTER, p):
                cur += p
            else:
                if cur:
                    merged.append(cur)
                    cur = p
                else:
                    cur = p
        if cur:
            merged.append(cur)

        # 将较大的片段再检查一次长度，必要时继续细分（递归式保证不会超长）
        results = []
        for m in merged:
            if m.strip():
                if self._len(m) > self.max_tokens:
                    results.extend(self._split_by_tokens(m))
                else:
                    results.append(m.strip())
        return results

    def _split_by_tokens(self, text: str) -> List[str]:
        """彻底兜底：按 token 粗切，保证每段<=max_tokens"""
        ids = self._encode(text)
        chunks = []
        pos = 0
        while pos < len(ids):
            end = min(pos + self.max_tokens, len(ids))
            chunks.append(self._decode(ids[pos:end]).strip())
            pos = end
        return [c for c in chunks if c]

    # ---------- 装箱与合并 ----------
    def _pack_sentences(self, sentences: List[str]) -> List[str]:
        """贪心装箱：尽量靠近 target，不超 max；不足 min 则尝试继续填充"""
        chunks: List[List[int]] = []
        current: List[int] = []
        current_len = 0
        tail_overlap: List[int] = []

        def flush_current():
            nonlocal current, current_len, tail_overlap
            if not current:
                return
            # 记录尾部重叠
            if self.overlap_tokens > 0:
                tail_overlap = current[-self.overlap_tokens:].copy()
            chunks.append(current)
            current = []
            current_len = 0

        i = 0
        while i < len(sentences):
            s = sentences[i]
            s_ids = self._encode(s)

            # 过长句子先行拆解
            if len(s_ids) > self.max_tokens:
                sub_sents = self._split_overlong_sentence(s)
                # 将分解结果插入到 sentences 序列中（原地展开）
                sentences = sentences[:i] + sub_sents + sentences[i + 1:]
                continue  # 重新处理插入后的元素

            # 预装：如有重叠，放在新块开头（但不能超 max）
            if not current and tail_overlap:
                # 注意：不重复计入重叠的语义“新内容预算”
                pre = tail_overlap
                if len(pre) >= self.max_tokens:
                    pre = pre[-self.max_tokens + 1:]  # 保底至少留1个 token 给新内容
                current = pre.copy()
                current_len = len(current)

            # 贪心：如果还能放得下就放
            if current_len + len(s_ids) <= self.max_tokens:
                current.extend(s_ids)
                current_len += len(s_ids)
                # 若已接近 target，尝试收束
                if current_len >= self.target_tokens:
                    flush_current()
                i += 1
            else:
                # 放不下：若当前块太小（<min），仍需换块（避免无限循环）
                if current_len < self.min_tokens:
                    # 当前块过小，但又放不下 s，说明 s 接近 max；直接换块，避免卡死
                    flush_current()
                    # 下一轮会先预填 overlap，再尝试放 s
                else:
                    flush_current()

        # 收尾
        if current:
            chunks.append(current)

        # 事后合并：把过小的块尽量并到前/后块
        chunks = self._merge_small_chunks(chunks)
        return [self._decode(c).strip() for c in chunks]

    def _merge_small_chunks(self, chunks: List[List[int]]) -> List[List[int]]:
        if not chunks:
            return chunks
        merged: List[List[int]] = []
        i = 0
        while i < len(chunks):
            cur = chunks[i]
            if len(cur) >= self.min_tokens or i == 0:
                merged.append(cur)
                i += 1
                continue

            # 当前块过小（且不是首块）
            prev = merged[-1]
            # 1) 优先尝试并入前块
            if len(prev) + len(cur) <= self.max_tokens:
                merged[-1] = prev + cur
                i += 1
                continue

            # 2) 再尝试并入后块
            if i + 1 < len(chunks) and len(cur) + len(chunks[i + 1]) <= self.max_tokens:
                chunks[i + 1] = cur + chunks[i + 1]
                i += 1  # 当前块并到后块，跳过
                continue

            # 3) 实在不行，就保留现状（通常因为相邻都接近上限）
            merged.append(cur)
            i += 1

        return merged

    # ---------- 对外接口 ----------
    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        paragraphs = self._split_paragraphs(text)
        all_sentences: List[str] = []
        for p in paragraphs:
            all_sentences.extend(self._split_sentences(p))

        # 清理空句
        sentences = [s for s in all_sentences if s and s.strip()]
        if not sentences:
            return []

        return self._pack_sentences(sentences)