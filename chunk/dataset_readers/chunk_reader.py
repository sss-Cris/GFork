import logging
import json
from typing import Dict, List, Iterable, Tuple, Any, Optional

import numpy as np
from overrides import overrides
from transformers import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field, TextField, SequenceLabelField, MetadataField, AdjacencyField
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from chunk.dataset_readers.dataset_helper import (
    convert_dep_tags_to_wordpiece_dep_tags,
    convert_dep_adj_to_wordpiece_dep_adj,
    convert_bound_indices_to_wordpiece_indices,
    convert_tag_indices_to_wordpiece_indices
)

# 设置日志
logger = logging.getLogger(__name__)

@DatasetReader.register(name="chunk_reader")
class ChunkReader(DatasetReader):
    """
    用于处理基于块的数据集的DatasetReader，支持BERT分词和语言学特征的处理。
    """

    def __init__(self,
                token_indexers: Dict[str, TokenIndexer] = None,
                domain_identifier: str = None,
                max_instances: Optional[int] = None,
                validation: bool = False,
                verbal_indicator: bool = True,
                transformer_model_name: str = None) -> None:
        """
        初始化 ChunkReader。

        参数：
            token_indexers: 用于标记字段的索引器。
            domain_identifier: 用于过滤特定领域数据的字符串。
            max_instances: 最多加载的实例数量，默认为 None 表示不限制。
            validation: 指定是否为验证模式。
            verbal_indicator: 是否启用语言指示器。
            transformer_model_name: 用于分词的预训练 BERT 模型名称。
        """
        super().__init__(max_instances=max_instances)


        # 初始化标记索引器，如果未提供则使用默认值
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._pos_tag_indexers = {"pos_tags": SingleIdTokenIndexer(namespace="pos_labels")}
        self._dep_tag_indexers = {"dep_tags": SingleIdTokenIndexer(namespace="dependency_labels")}

        # 存储其他参数
        self._domain_identifier = domain_identifier
        self._validation = validation
        self._verbal_indicator = verbal_indicator

        # 初始化BERT分词器
        self.bert_tokenizer = BertTokenizer.from_pretrained(transformer_model_name)
        self.lowercase_input = "uncased" in transformer_model_name

    # def _wordpiece_tokenize_input(self, tokens: List[str]) -> Tuple[List[str], List[int], List[int]]:
    #     """
    #     使用BERT分词器将输入标记分词为WordPiece标记。

    #     参数：
    #         tokens: 输入标记列表。

    #     返回：
    #         一个元组，包含WordPiece标记、起始偏移和结束偏移。
    #     """
    #     word_piece_tokens: List[str] = []
    #     start_offsets: List[int] = []
    #     end_offsets: List[int] = []

    #     cumulative = 0
    #     for token in tokens:
    #         # 如果需要，转换为小写
    #         if self.lowercase_input:
    #             token = token.lower()

    #         # 使用WordPiece分词器分词
    #         word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)

    #         # 记录起始和结束偏移
    #         start_offsets.append(cumulative + 1)
    #         cumulative += len(word_pieces)
    #         end_offsets.append(cumulative)

    #         # 扩展WordPiece标记
    #         word_piece_tokens.extend(word_pieces)
    #     MAX_LEN = 512

    #     # 最终 WordPiece token 加上 [CLS] 和 [SEP] 总长度不能超过 512
    #     if len(word_piece_tokens) + 2 > MAX_LEN:
    #         # 截断到最多 510 个 WordPiece token
    #         word_piece_tokens = word_piece_tokens[:MAX_LEN - 2]

    #     # 为BERT添加特殊标记
    #     wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]
    #     return wordpieces, end_offsets, start_offsets
    def _wordpiece_tokenize_input(self, tokens: List[str]) -> Tuple[List[str], List[int], List[int]]:
        """
        使用BERT分词器将输入标记分词为WordPiece标记，并确保总长度不超过512。

        参数：
            tokens: 输入的原始词语列表。

        返回：
            wordpieces: 添加了 [CLS] 和 [SEP] 的 WordPiece 标记序列
            end_offsets: 每个原始 token 在 WordPiece 序列中的结束位置（不含 [CLS]）
            start_offsets: 每个原始 token 的起始位置（不含 [CLS]）
        """
        max_wordpieces = 510  # 为 [CLS] 和 [SEP] 留出空间
        word_piece_tokens: List[str] = []
        start_offsets: List[int] = []
        end_offsets: List[int] = []

        cumulative = 0
        for token in tokens:
            if self.lowercase_input:
                token = token.lower()

            word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)

            # 如果加上当前 token 会超出限制，就终止
            if cumulative + len(word_pieces) > max_wordpieces:
                break

            start_offsets.append(cumulative + 1)  # +1 是因为 [CLS] 占位
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)

            word_piece_tokens.extend(word_pieces)

        # 添加特殊 token
        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]
        return wordpieces, end_offsets, start_offsets


    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        从给定的数据集文件中读取实例。

        参数：
            file_path: 数据集文件的路径。

        生成：
            包含处理字段的Instance对象。
        """
        # print("正确进入！")
        # 如果需要，解析缓存的文件路径
        file_path = cached_path(file_path)
        logger.info("从以下位置读取数据集文件中的实例：%s", file_path)

        # 如果指定了领域标识符，则进行过滤
        if self._domain_identifier:
            logger.info("仅包括包含以下领域的文件路径：%s", self._domain_identifier)

        # 加载JSON数据
        with open(file_path, 'r', encoding='utf-8') as file_in:
            json_sentences = json.load(file_in)
        
        # print(f"Total examples in dataset: {len(json_sentences)}")  # 输出总样本数
            
        # 处理每个句子
        for index, item in enumerate(json_sentences):
            # print(f"Processing example {index}: {item}")
            # logger.info(f"Processing sentence {index}: {item}")
            tokens = item['words']
            bound_tags = item.get('bound_tags', None)
            token_tags = item.get('token_tags', None)
            pos_tags = item['spacy_pos']
            dep_edges = item['dep_graph_edges']
            dep_nodes = item['dep_graph_nodes']

            # 将数据转换为Instance
            instance = self.text_to_instance(
                tokens, bound_tags, token_tags, pos_tags, dep_edges, dep_nodes, index
            )

            yield self.text_to_instance(tokens, bound_tags, token_tags, pos_tags, dep_edges, dep_nodes, index)

    def text_to_instance(self, 
                         tokens: List[str], 
                         bound_tags: List[int], 
                         token_tags: List[str], 
                         pos_tags: List[str], 
                         dep_edges: List[List], 
                         dep_nodes: List[str], 
                         sent_id: int = None) -> Instance:
        """
        将原始数据转换为AllenNLP的Instance。

        参数：
            tokens: 句子的原始标记。
            bound_tags: 标记的边界标签。
            token_tags: 标记的标签。
            pos_tags: 词性标签。
            dep_edges: 依存关系边。
            dep_nodes: 依存关系节点。
            sent_id: 句子ID。

        返回：
            包含所有字段的Instance。
        """
        
        # print("tokens:", tokens)
        # print("bound_tags:", bound_tags)
        # print("token_tags:", token_tags)
        # print("pos_tags:", pos_tags)
        # print("dep_edges:", dep_edges)
        # print("dep_nodes:", dep_nodes)
        
        
        # 存储附加信息的元数据
        metadata_dict = {}

        # 将标记分词为WordPiece标记
        wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input(tokens)
        metadata_dict["offsets"] = offsets

        # 使用WordPiece标记创建文本字段
        text_field = TextField(
            [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
            token_indexers=self._token_indexers
        )

        # 将语言学特征转换为WordPiece格式
        pos_wordpc_tags = convert_dep_tags_to_wordpiece_dep_tags(pos_tags, offsets)
        pos_field = TextField(
            [Token(t) for t in pos_wordpc_tags], 
            token_indexers=self._pos_tag_indexers
        )

        dep_graph_nodes = convert_dep_tags_to_wordpiece_dep_tags(dep_nodes, offsets)
        dep_edges_tuple = [(item[0][0], item[0][1]) for item in dep_edges]
        dep_edges_tuple = convert_dep_adj_to_wordpiece_dep_adj(dep_edges_tuple, start_offsets, offsets)

        dep_field = TextField(
            [Token(t) for t in dep_graph_nodes],
            token_indexers=self._dep_tag_indexers
        )
        dep_adj_field = AdjacencyField(dep_edges_tuple, dep_field, padding_value=0)

        # 处理边界和标记标签
        fields = {"tokens": text_field, "pos_tags": pos_field, "dep_nodes": dep_field, "dep_edges": dep_adj_field}
        if bound_tags is not None:
            new_bound_tags = convert_bound_indices_to_wordpiece_indices(bound_tags, offsets)
            new_token_tags = convert_tag_indices_to_wordpiece_indices(token_tags, offsets)

            fields.update({
                'bound_tags': SequenceLabelField(new_bound_tags, text_field),
                'token_tags': SequenceLabelField(new_token_tags, text_field)
            })
            metadata_dict.update({"bound_tags": bound_tags, "token_tags": token_tags})

        # 添加元数据
        metadata_dict.update({"words": tokens, "validation": self._validation})
        if sent_id is not None:
            metadata_dict['sent_id'] = sent_id

        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)

# import logging
# import json
# from typing import Dict, List, Iterable, Tuple, Any, Optional

# import numpy as np
# from overrides import overrides
# from transformers import BertTokenizer

# from allennlp.common.file_utils import cached_path
# from allennlp.data.dataset_readers.dataset_reader import DatasetReader
# from allennlp.data.fields import (
#     Field, TextField, SequenceLabelField, MetadataField, AdjacencyField
# )
# from allennlp.data.instance import Instance
# from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
# from allennlp.data.tokenizers import Token
# from chunk.dataset_readers.dataset_helper import (
#     convert_dep_tags_to_wordpiece_dep_tags,
#     convert_dep_adj_to_wordpiece_dep_adj,
#     convert_bound_indices_to_wordpiece_indices,
#     convert_tag_indices_to_wordpiece_indices
# )

# # 设置日志
# logger = logging.getLogger(__name__)

# @DatasetReader.register(name="chunk_reader")
# class ChunkReader(DatasetReader):
#     """
#     用于处理基于块的数据集的DatasetReader，支持BERT分词和语言学特征的处理。
#     """

#     def __init__(self,
#                 token_indexers: Dict[str, TokenIndexer] = None,
#                 domain_identifier: str = None,
#                 max_instances: Optional[int] = None,
#                 validation: bool = False,
#                 verbal_indicator: bool = True,
#                 bert_model_name: str = None) -> None:
#         """
#         初始化 ChunkReader。

#         参数：
#             token_indexers: 用于标记字段的索引器。
#             domain_identifier: 用于过滤特定领域数据的字符串。
#             max_instances: 最多加载的实例数量，默认为 None 表示不限制。
#             validation: 指定是否为验证模式。
#             verbal_indicator: 是否启用语言指示器。
#             bert_model_name: 用于分词的预训练 BERT 模型名称。
#         """
#         super().__init__(max_instances=max_instances)


#         # 初始化标记索引器，如果未提供则使用默认值
#         self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
#         self._pos_tag_indexers = {"pos_tags": SingleIdTokenIndexer(namespace="pos_labels")}
#         self._dep_tag_indexers = {"dep_tags": SingleIdTokenIndexer(namespace="dependency_labels")}

#         # 存储其他参数
#         self._domain_identifier = domain_identifier
#         self._validation = validation
#         self._verbal_indicator = verbal_indicator

#         # 初始化BERT分词器
#         self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
#         self.lowercase_input = "uncased" in bert_model_name

#     def _wordpiece_tokenize_input(self, tokens: List[str]) -> Tuple[List[str], List[int], List[int]]:
#         """
#         使用BERT分词器将输入标记分词为WordPiece标记。

#         参数：
#             tokens: 输入标记列表。

#         返回：
#             一个元组，包含WordPiece标记、起始偏移和结束偏移。
#         """
#         word_piece_tokens: List[str] = []
#         start_offsets: List[int] = []
#         end_offsets: List[int] = []

#         cumulative = 0
#         for token in tokens:
#             # 如果需要，转换为小写
#             if self.lowercase_input:
#                 token = token.lower()

#             # 使用WordPiece分词器分词
#             word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)

#             # 记录起始和结束偏移
#             start_offsets.append(cumulative + 1)
#             cumulative += len(word_pieces)
#             end_offsets.append(cumulative)

#             # 扩展WordPiece标记
#             word_piece_tokens.extend(word_pieces)

#         # 为BERT添加特殊标记
#         wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]
#         return wordpieces, end_offsets, start_offsets

#     @overrides
#     def _read(self, file_path: str) -> Iterable[Instance]:
#         """
#         从给定的数据集文件中读取实例。

#         参数：
#             file_path: 数据集文件的路径。

#         生成：
#             包含处理字段的Instance对象。
#         """
#         # print("正确进入！")
#         # 如果需要，解析缓存的文件路径
#         file_path = cached_path(file_path)
#         logger.info("从以下位置读取数据集文件中的实例：%s", file_path)

#         # 如果指定了领域标识符，则进行过滤
#         if self._domain_identifier:
#             logger.info("仅包括包含以下领域的文件路径：%s", self._domain_identifier)

#         # 加载JSON数据
#         with open(file_path, 'r', encoding='utf-8') as file_in:
#             json_sentences = json.load(file_in)
        
#         # print(f"Total examples in dataset: {len(json_sentences)}")  # 输出总样本数
            
#         # 处理每个句子
#         for index, item in enumerate(json_sentences):
#             # print(f"Processing example {index}: {item}")
#             # logger.info(f"Processing sentence {index}: {item}")
#             tokens = item['words']
#             bound_tags = item.get('bound_tags', None)
#             token_tags = item.get('token_tags', None)
#             pos_tags = item['spacy_pos']
#             dep_edges = item['dep_graph_edges']
#             dep_nodes = item['dep_graph_nodes']

#             # 将数据转换为Instance
#             instance = self.text_to_instance(
#                 tokens, bound_tags, token_tags, pos_tags, dep_edges, dep_nodes, index
#             )

#             yield self.text_to_instance(tokens, bound_tags, token_tags, pos_tags, dep_edges, dep_nodes, index)

#     def text_to_instance(self, 
#                          tokens: List[str], 
#                          bound_tags: List[int], 
#                          token_tags: List[str], 
#                          pos_tags: List[str], 
#                          dep_edges: List[List], 
#                          dep_nodes: List[str], 
#                          sent_id: int = None) -> Instance:
#         """
#         将原始数据转换为AllenNLP的Instance。

#         参数：
#             tokens: 句子的原始标记。
#             bound_tags: 标记的边界标签。
#             token_tags: 标记的标签。
#             pos_tags: 词性标签。
#             dep_edges: 依存关系边。
#             dep_nodes: 依存关系节点。
#             sent_id: 句子ID。

#         返回：
#             包含所有字段的Instance。
#         """
        
#         # print("tokens:", tokens)
#         # print("bound_tags:", bound_tags)
#         # print("token_tags:", token_tags)
#         # print("pos_tags:", pos_tags)
#         # print("dep_edges:", dep_edges)
#         # print("dep_nodes:", dep_nodes)
        
        
#         # 存储附加信息的元数据
#         metadata_dict = {}

#         # 将标记分词为WordPiece标记
#         wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input(tokens)
#         metadata_dict["offsets"] = offsets

#         # 使用WordPiece标记创建文本字段
#         text_field = TextField(
#             [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
#             token_indexers=self._token_indexers
#         )

#         # 将语言学特征转换为WordPiece格式
#         pos_wordpc_tags = convert_dep_tags_to_wordpiece_dep_tags(pos_tags, offsets)
#         pos_field = TextField(
#             [Token(t) for t in pos_wordpc_tags], 
#             token_indexers=self._pos_tag_indexers
#         )

#         dep_graph_nodes = convert_dep_tags_to_wordpiece_dep_tags(dep_nodes, offsets)
#         dep_edges_tuple = [(item[0][0], item[0][1]) for item in dep_edges]
#         dep_edges_tuple = convert_dep_adj_to_wordpiece_dep_adj(dep_edges_tuple, start_offsets, offsets)

#         dep_field = TextField(
#             [Token(t) for t in dep_graph_nodes],
#             token_indexers=self._dep_tag_indexers
#         )
#         dep_adj_field = AdjacencyField(dep_edges_tuple, dep_field, padding_value=0)

#         # 处理边界和标记标签
#         fields = {"tokens": text_field, "pos_tags": pos_field, "dep_nodes": dep_field, "dep_edges": dep_adj_field}
#         if bound_tags is not None:
#             new_bound_tags = convert_bound_indices_to_wordpiece_indices(bound_tags, offsets)
#             new_token_tags = convert_tag_indices_to_wordpiece_indices(token_tags, offsets)

#             fields.update({
#                 'bound_tags': SequenceLabelField(new_bound_tags, text_field),
#                 'token_tags': SequenceLabelField(new_token_tags, text_field)
#             })
#             metadata_dict.update({"bound_tags": bound_tags, "token_tags": token_tags})

#         # 添加元数据
#         metadata_dict.update({"words": tokens, "validation": self._validation})
#         if sent_id is not None:
#             metadata_dict['sent_id'] = sent_id

#         fields["metadata"] = MetadataField(metadata_dict)
#         return Instance(fields)
