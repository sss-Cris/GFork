from typing import Dict, List, Optional, Any, Union
from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout, Embedding, LogSoftmax
import torch.nn.functional as F
from transformers import BertModel
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import Metric

import sys

@Model.register("chunk_model")
class Chunk_Model(Model):
    """
    Chunk-Based POS（词性标注）模型，支持通过BERT嵌入与依存关系建模。
    """
    def __init__(self,
                 vocab: Vocabulary,  # 词汇表
                 bert_model: Union[str, BertModel],  # BERT 模型
                 embedding_dropout: float = 0.0,  # 嵌入层的Dropout比例
                 dependency_label_dim: int = 400,  # 依存标签的维度
                 initializer: InitializerApplicator = InitializerApplicator(),  # 参数初始化
                 regularizer: Optional[RegularizerApplicator] = None,  # 正则化器
                 label_smoothing: float = None,  # 标签平滑
                 tuple_metric: Metric = None) -> None:  # 用于评估的指标
        super(Chunk_Model, self).__init__(vocab, regularizer)

        # 初始化 BERT 模型
        if isinstance(bert_model, str):
            self.bert_model = BertModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model

        # 边界分类数和标签分类数
        self.num_bound_classes = 2
        self.num_type_classes = self.vocab.get_vocab_size("labels")

        # 线性投影层（预测边界和类型）
        self.tag_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_type_classes)
        self.bound_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_bound_classes)

        # POS 嵌入
        self.num_pos_labels = self.vocab.get_vocab_size("pos_labels")
        self.pos_embedding = Embedding(self.num_pos_labels, self.bert_model.config.hidden_size, padding_idx=0)

        # Dropout层
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self._token_based_metric = tuple_metric
        self.LogSoftmax = LogSoftmax()
        initializer(self)

    def forward(
        self,
        tokens: Dict[str, torch.Tensor],
        metadata: List[Any],
        pos_tags: torch.Tensor,
        dep_nodes: Dict[str, torch.Tensor] = None,
        dep_edges: Dict[str, torch.Tensor] = None,
        bound_tags: torch.Tensor = None,
        token_tags: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播逻辑。
        参数：
        - tokens: 输入文本的张量化表示。
        - metadata: 附加的元信息，如词偏移量等。
        - pos_tags: 词性标签张量。
        - dep_nodes: 依存节点的张量表示。
        - dep_edges: 依存边的张量表示。
        - bound_tags: 边界标签（目标）。
        - token_tags: 类型标签（目标）。
        返回：
        - output_dict: 包含预测概率、损失等的字典。
        """

        # 获取输入文本的mask
        mask = get_text_field_mask(tokens)

        # 通过BERT获取嵌入表示
        # print('1111',tokens["tokens"]["tokens"])
        # print('1111',tokens["tokens"]["tokens"].size())
        # print('2222',mask)
        # print('2222',mask.size())
        bert_outputs = self.bert_model(
            input_ids=tokens["tokens"]["tokens"],
            attention_mask=mask
        )
        bert_embeddings = bert_outputs["last_hidden_state"]
        # print('tokens["tokens"]["tokens"][0]',tokens["tokens"]["tokens"][0])

        # 词性嵌入与BERT嵌入结合
        embed_pos = self.pos_embedding(pos_tags["pos_tags"]["tokens"])
        embedded_text_input = self.embedding_dropout(bert_embeddings + embed_pos)

        # 边界和类型的预测
        bound_logits = self.bound_projection_layer(embedded_text_input)
        bound_probabilities = F.softmax(bound_logits, dim=-1)

        type_logits = self.tag_projection_layer(embedded_text_input)
        type_probabilities = F.softmax(type_logits, dim=-1)

        # 输出字典
        output_dict = {
            "bound_logits": bound_logits,
            "bound_probabilities": bound_probabilities,
            "type_logits": type_logits,
            "type_probabilities": type_probabilities,
            "mask": mask,
            "wordpiece_embedding": embedded_text_input
        }

        # 计算损失（仅在训练阶段）
        if not metadata[0]['validation']:
            bound_loss = sequence_cross_entropy_with_logits(bound_logits, bound_tags, mask, label_smoothing=self._label_smoothing)
            tag_loss = sequence_cross_entropy_with_logits(type_logits, token_tags, mask, label_smoothing=self._label_smoothing)
            output_dict["loss"] = bound_loss + 0.8 * tag_loss

        # 保留掩码信息，用于后续解码
        output_dict["mask"] = mask
        # 添加偏移信息，用于解码非 WordPiece 标签
        words, offsets = zip(*[(x["words"], x["offsets"]) for x in metadata])
        output_dict["words"] = list(words)
        output_dict["wordpiece_offsets"] = list(offsets)

        # 验证阶段，调用解码方法并计算指标
        if metadata[0]['validation']:
            output_dict = self.decode(output_dict)
            sent_ids = []
            if bound_tags is not None:
                bound_tags_list, type_tags_list = [], []
                for item in metadata:
                    bound_tags_list.append(item['bound_tags'])
                    type_tags_list.append(item['token_tags'])
                    sent_ids.append(item['sent_id'])

                self._token_based_metric(output_dict, bound_tags_list, type_tags_list, sent_ids)

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        解码方法，将模型的概率输出转为具体的边界和标签。
        """
        # 边界预测和标签预测
        bound_predictions = output_dict['bound_probabilities']
        type_predictions = output_dict['type_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        # 将预测结果转换为列表
        bound_predictions_list = [bound_predictions[i].detach().cpu() for i in range(bound_predictions.size(0))]
        type_predictions_list = [type_predictions[i].detach().cpu() for i in range(type_predictions.size(0))]

        # Viterbi解码所需的转移矩阵和起始转移
        transition_matrix = self.get_viterbi_pairwise_potentials()
        start_transitions = self.get_start_transitions()

        wordpiece_bound_tags, word_bound_tags = [], []
        wordpiece_type_tags, word_type_tags = [], []

        # 遍历每个样本，解码边界和标签
        word_embeddings = []
        for i, (bound_predictions, type_predictions, length, offsets, embedding) in enumerate(zip(
                bound_predictions_list, type_predictions_list, sequence_lengths, 
                output_dict["wordpiece_offsets"], output_dict["wordpiece_embedding"])):
            # 解码边界
            bound_tags = torch.argmax(bound_predictions[:length], dim=1).tolist()
            wordpiece_bound_tags.append(bound_tags)
            word_bound_tags.append([bound_tags[idx] for idx in offsets])

            # 使用 Viterbi 解码标签
            type_ints, _ = viterbi_decode(type_predictions[:length], transition_matrix, allowed_start_transitions=start_transitions)
            type_tags = [self.vocab.get_token_from_index(x, namespace="labels") for x in type_ints]
            wordpiece_type_tags.append(type_tags)
            word_type_tags.append([type_tags[idx] for idx in offsets])

            # 获取单词级别的嵌入
            # print("Number of 1s in output_dict['mask'][0]:", (output_dict["mask"][0] == 1).sum().item())
            # print('output_dict["words"][0]',output_dict["words"][0])
            # print('len(output_dict["words"][0])',len(output_dict["words"][0]))
            # print("wordpiece_embedding", output_dict["wordpiece_embedding"].size())
            # print('len(bound_tags)',len(bound_tags))
            # print('len([bound_tags[idx] for idx in offsets])',len([bound_tags[idx] for idx in offsets]))
            # print('offsets',offsets)
            # 构造 word_embedding
            word_embedding = []
            for i in range(len(offsets)):
                if i == 0:
                    word_embedding.append(embedding[1:offsets[i]+1].mean(dim=0))
                else:
                    start, end = offsets[i-1]+1, offsets[i]+1
                    word_embedding.append(embedding[start:end].mean(dim=0))
            # 将结果转换为张量
            word_embedding = torch.stack(word_embedding)
            word_embeddings.append(word_embedding)

        # 保存解码后的结果
        output_dict['wordpiece_bound_tags'] = wordpiece_bound_tags
        output_dict['bound_tags'] = word_bound_tags
        output_dict['wordpiece_type_tags'] = wordpiece_type_tags
        output_dict['type_tags'] = word_type_tags
        output_dict['word_embedding'] = word_embeddings
        return output_dict



    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        获取模型的评价指标。
        参数：
        - reset: 是否重置指标。
        返回：
        - all_metrics: 包含所有指标的字典。
        """
        all_metrics = {}
        if not self.training:
            if self._token_based_metric:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))
        return all_metrics
    
    def get_viterbi_pairwise_potentials(self):
        """
        生成 BIO 标签的配对潜能矩阵，用于 Viterbi 解码。
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        # 设置非法转移的潜能为 -inf
        for i, prev_label in all_labels.items():
            for j, label in all_labels.items():
                if i != j and label[0] == 'I' and not prev_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

    def get_start_transitions(self):
        """
        定义序列起始位置的合法转移潜能。
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        start_transitions = torch.zeros(len(all_labels))

        for i, label in all_labels.items():
            if label[0] == "I":  # I 标签不能作为起始标签
                start_transitions[i] = float("-inf")
        return start_transitions
