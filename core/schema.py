import copy
from typing import Dict, Optional, List


class GraphPath:
    """
    表示知识图谱中的检索路径
    """

    def __init__(self, start_node: Dict):
        # 核心状态数据
        self.nodes = [start_node]
        self.relations = []
        self.scores = []
        self.is_finished = False

        # 缓存或临时变量
        self._diversity = None
        self._score = 0.0

    @property
    def score(self) -> float:
        valid_scores = [score for score in self.scores if score != 0.0]
        if valid_scores:
            self._score = sum(valid_scores) / (len(valid_scores) + 1e-8)
        else:
            self._score = 0.0
        return round(self._score, 5)

    @property
    def current_node(self) -> Dict:
        """获取路径末尾节点"""
        return self.nodes[-1]

    @property
    def diversity(self) -> float:
        """计算路径多样性（基于节点类型分布的辛普森指数）"""
        type_counts = {}
        for n in self.nodes:
            t = n.get('type', 'Unknown')
            type_counts[t] = type_counts.get(t, 0) + 1
        self._diversity = 1 - sum((v / len(self.nodes)) ** 2 for v in type_counts.values())
        return self._diversity

    def add_node(self, node: Dict, relation: Optional[Dict] = None) -> 'GraphPath':
        self.nodes.append(node)
        if relation:
            self.relations.append(relation)
        # 状态改变，重置缓存
        self._diversity = None
        return self

    def pop_node(self) -> 'GraphPath':
        self.nodes.pop()
        self.relations.pop()
        self._diversity = None
        return self

    def copy(self) -> 'GraphPath':
        """
        优化后的浅拷贝方法。

        机制：
        1. 列表容器 (nodes, relations) 是全新的对象 -> 保证路径分支互不影响。
        2. 列表内的元素 (dict) 是引用 -> 避免内存复制，极大提升速度。
        """
        # 1. 创建新实例
        # 这里的 start_node 参数只为了满足 __init__ 签名，稍后会被覆盖
        # 此时 new_path.nodes 只有 [self.nodes[0]]
        new_path = GraphPath(self.nodes[0])

        # 2. 核心优化：使用切片 [:] 进行列表的浅拷贝
        # new_path.nodes 成为一个新的 list 对象，但在内存中存储的是指向原节点 dict 的指针
        # 操作 new_path.nodes.append() 不会影响 self.nodes
        new_path.nodes = self.nodes[:]

        # 同理处理 relations 和 scores
        new_path.relations = self.relations[:]
        new_path.scores = self.scores[:]

        # 3. 复制值类型状态 (bool, float, int 等不可变类型直接赋值即可)
        new_path.is_finished = self.is_finished

        # 缓存状态也直接复制（因为是不可变类型或 None）
        new_path._diversity = self._diversity
        new_path._score = self._score

        return new_path

    def __len__(self):
        return len(self.nodes)

    @property
    def format_string(self, path_format="link"):
        # 注意：这里原代码中的 path_format 参数在 property 中无法直接传递参数。
        # 如果需要支持参数，建议改为普通方法 get_format_string(self, path_format="link")
        # 此处保留原逻辑，默认为 "link" 风格的逻辑

        # 默认实现 link 风格
        if not self.nodes:
            return ""

        format_string = f"{self.nodes[0].get('mention', 'UNK')}"
        for relation, node in zip(self.relations, self.nodes[1:]):
            # 简单的防御性编程，防止 KeyError
            r_val = relation.get('r', 'unk')
            n_mention = node.get('mention', 'UNK')

            if relation.get('end', {}).get('id') == node.get('id'):
                format_string += f" - {r_val} -> {n_mention}"
            else:
                format_string += f" <- {r_val} - {n_mention}"
        return format_string

    # -----------------------------------------------------------
    # 核心修改部分：to_dict 和 from_dict
    # -----------------------------------------------------------

    def to_dict(self) -> Dict:
        """
        将路径对象序列化为字典。
        包含恢复对象所需的所有原始数据，以及用于展示的计算数据。
        """
        return {
            # --- 核心状态数据 (用于 from_dict 还原) ---
            "nodes": self.nodes,
            "relations": self.relations,
            "scores": self.scores,
            "is_finished": self.is_finished,

            # --- 展示/计算数据 (用于前端展示或日志) ---
            # 这些数据是导出的快照，还原时会由 property 重新计算
            "score": self.score,
            "diversity": round(self.diversity, 5),
            "format_string": self.format_string
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'GraphPath':
        """
        从字典还原 GraphPath 对象。

        :param data: 由 to_dict 生成的字典
        :return: 还原后的 GraphPath 对象
        """
        if not data or 'nodes' not in data or not data['nodes']:
            raise ValueError("无法从字典还原：缺少 'nodes' 数据")

        # 1. 使用起始节点初始化对象
        # 使用 deepcopy 确保还原的对象与原字典数据解耦
        start_node = copy.deepcopy(data['nodes'][0])
        path = cls(start_node)

        # 2. 覆盖并恢复核心列表数据
        # 注意：nodes[0] 已经在 init 中添加了，这里直接覆盖整个列表最简单且不易出错
        path.nodes = copy.deepcopy(data['nodes'])
        path.relations = copy.deepcopy(data.get('relations', []))
        path.scores = copy.deepcopy(data.get('scores', []))

        # 3. 恢复布尔状态
        path.is_finished = data.get('is_finished', False)

        # 4. 缓存数据不需要显式恢复，
        # 因为访问 .score 或 .diversity 属性时会根据上面的 nodes/scores 自动重新计算

        return path