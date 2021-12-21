#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2021/12/21 10:54
# @File     : PostProcessing.py
# @Project  : BBox2Html

from abc import ABCMeta, abstractmethod
from mmcv.utils import Registry
import numpy as np


POSTPROCESS = Registry('postprocess')


def adj_to_cell(adj, bboxes, mod):
    """Calculating start and end row / column of each cell according to row / column adjacent relationships

    Args:
        adj(np.array): (n x n). row / column adjacent relationships of non-empty aligned cells
        bboxes(np.array): (n x 4). bboxes of non-empty aligned cells
        mod(str): 'row' or 'col'

    Returns:
        list(np.array): start and end row of each cell if mod is 'row' / start and end col of each cell if mod is 'col'
    """

    assert mod in ('row', 'col')

    # generate graph of each non-empty aligned cells
    nodenum = adj.shape[0]                               # 节点（单元格）数量
    edge_temp = np.where(adj != 0)                       # 寻找有邻接的节点(坐标化后表示谁与谁之间有连接)
    edge = list(zip(edge_temp[0], edge_temp[1]))
    table_graph = Graph()                                # 生成图, 添加节点与边
    table_graph.add_nodes_from(list(range(nodenum)))
    table_graph.add_edges_from(edge)

    # Find maximal clique in the graph
    clique_list = list(find_cliques(table_graph))        # 生成所有的最大团

    # Sorting the maximal cliques
    coord = []
    times = np.zeros(nodenum)
    for clique in clique_list:
        for node in clique:
            times[node] += 1                            # 统计每个节点属于多少个最大团（节点本身不是最大团说明和其他节点需要合并）
    for ind, clique in enumerate(clique_list):
        # 除非该最大团中的所有节点都属于多个最大团，否则将选择仅属于该最大团的节点进行排序
        nodes_nospan = [node for node in clique if times[node] == 1]
        nodes_select = nodes_nospan if len(nodes_nospan) else clique
        # 当前团的中心坐标（横坐标or纵坐标）
        coord_mean = [ind, (bboxes[nodes_select, 1] + bboxes[nodes_select, 3]).mean()] if mod == 'row' \
            else [ind, (bboxes[nodes_select, 0] + bboxes[nodes_select, 2]).mean()]
        coord.append(coord_mean)
    coord = np.array(coord, dtype='int')
    coord = coord[coord[:, 1].argsort()]  # 根据最大团中心坐标进行排序

    # 记录属于最大团的节点其属于哪个最大团, 有些节点会属于多个最大团(对应的实际情况为：一行单元格是一个最大团，但是跨行的单元格属于两个或多个最大团)
    listcell = [[] for _ in range(nodenum)]
    for ind, coo in enumerate(coord[:, 0]):
        for node in clique_list[coo]:
            listcell[node] = np.append(listcell[node], ind)

    return listcell


def rect_max_iou(box_1, box_2):
    """Calculate the maximum IoU between two boxes: the intersect area / the area of the smaller box

    Args:
        box_1 (np.array | list): [x1, y1, x2, y2]
        box_2 (np.array | list): [x1, y1, x2, y2]

    Returns:
        float: maximum IoU between the two boxes
    """

    addone = 0  # 0 in mmdet2.0 / 1 in mmdet 1.0
    box_1, box_2 = np.array(box_1), np.array(box_2)

    x_start = np.maximum(box_1[0], box_2[0])
    y_start = np.maximum(box_1[1], box_2[1])
    x_end = np.minimum(box_1[2], box_2[2])
    y_end = np.minimum(box_1[3], box_2[3])

    area1 = (box_1[2] - box_1[0] + addone) * (box_1[3] - box_1[1] + addone)
    area2 = (box_2[2] - box_2[0] + addone) * (box_2[3] - box_2[1] + addone)
    overlap = np.maximum(x_end - x_start + addone, 0) * np.maximum(y_end - y_start + addone, 0)

    return overlap / min(area1, area2)


def nms_inter_classes(bboxes, iou_thres=0.3):
    """NMS between all classes

    Args:
        bboxes(list): [bboxes in cls1(np.array), bboxes in cls2(np.array), ...]. bboxes of each classes
        iou_thres(float): nsm threshold

    Returns:
        np.array: (n x 4).bboxes of targets after NMS between all classes
        list(list): (n x 1).labels of targets after NMS between all classes
    """

    lable_id = 0
    merge_bboxes, merge_labels = [], []
    for bboxes_cls in bboxes:
        if lable_id:
            merge_bboxes = np.concatenate((merge_bboxes, bboxes_cls), axis=0)
        else:
            merge_bboxes = bboxes_cls
        merge_labels += [[lable_id]] * len(bboxes_cls)
        lable_id += 1

    mark = np.ones(len(merge_bboxes), dtype=int)
    score_index = merge_bboxes[:, -1].argsort()[::-1]
    for i, cur in enumerate(score_index):
        if mark[cur] == 0:
            continue
        for ind in score_index[i + 1:]:
            if mark[ind] == 1 and rect_max_iou(merge_bboxes[cur], merge_bboxes[ind]) >= iou_thres:
                mark[ind] = 0
    new_bboxes = merge_bboxes[mark == 1, :4]
    new_labels = np.array(merge_labels)[mark == 1]
    new_labels = [list(map(int, lab)) for lab in new_labels]

    return new_bboxes, new_labels


def area_to_html(area, labels, texts_tokens):
    """ Generate structure html and text tokens from area, which is the intermediate result of post-processing.

    Args:
        area(np.array): (n x m). 表格区域行列单元格划分(合并前,如3行5列:n=3,m=5).
        labels(list[list]): (n x 1).非空cell对应的标签，如 0:theah 、 1:tbody
        texts_tokens(list[list]): texts_tokens for each non-empty cell

    Returns:
        list(str): The html that characterizes the structure of table
        list(str): text tokens for each cell (including empty cells)
    """

    area_extend = np.zeros([area.shape[0] + 1, area.shape[1] + 1])
    area_extend[:-1, :-1] = area
    html_struct_recon = []
    text_tokens_recon = []
    headend = 0
    for height in range(area.shape[0]):
        html_struct_recon.append("<tr>")
        width = 0
        numhead, numbody = 0, 0
        while width < area.shape[1]:
            # curent cell is rest part of a rowspan cell
            if height != 0 and area_extend[height, width] == area_extend[height - 1, width]:
                width += 1

            # td 标签没有合并（直接 添加<td></td>）
            elif area_extend[height, width] != area_extend[height + 1, width] and area_extend[height, width] != \
                    area_extend[height, width + 1]:
                html_struct_recon.append("<td>")
                html_struct_recon.append("</td>")
                texts_insert = texts_tokens[int(area_extend[height, width]) - 1] if int(
                    area_extend[height, width]) >= 1 else [""]
                text_tokens_recon.append({'tokens': texts_insert})

                # 计算这行中"</thead>" and "<body>" 的数量
                # int(area_extend[height, width])计算要获取标签的索引，索引小于1全是负数是空单元格
                # labels[int(area_extend[height, width]) - 1]是取单元格对应的label值, labels标记了head、body
                if int(area_extend[height, width]) < 1:
                    pass
                elif labels[int(area_extend[height, width]) - 1][0]:
                    numbody += 1
                elif not labels[int(area_extend[height, width]) - 1][0]:
                    numhead += 1
                width += 1

            # td 标签仅有列合并时 （添加: <td, colspan='colspan', >,</td>）
            elif area_extend[height, width] != area_extend[height + 1, width] and area_extend[height, width] == \
                    area_extend[height, width + 1]:
                colspan = 1
                while area_extend[height, width] == area_extend[height, width + colspan]:
                    colspan += 1
                    if (width + colspan) == (area.shape[1]):
                        break
                html_struct_recon.append("<td")
                html_struct_recon.append(" colspan=\"%s\"" % str(colspan))
                html_struct_recon.append(">")
                html_struct_recon.append("</td>")
                texts_insert = texts_tokens[int(area_extend[height, width]) - 1] if int(
                    area_extend[height, width]) >= 1 else [""]
                text_tokens_recon.append({'tokens': texts_insert})

                # caculate the number of "</thead>" and "<body>" in this row
                if int(area_extend[height, width]) < 1:
                    pass
                elif labels[int(area_extend[height, width]) - 1][0]:
                    numbody += 1
                elif not labels[int(area_extend[height, width]) - 1][0]:
                    numhead += 1
                width += colspan

            # td 标签仅有行合并时（添加: <td, rowspan='rowspan', >,</td>）
            elif area_extend[height, width] == area_extend[height + 1, width] and area_extend[height, width] != \
                    area_extend[height, width + 1]:
                rowspan = 1
                while area_extend[height, width] == area_extend[height + rowspan, width]:
                    rowspan += 1
                    if height + rowspan == area.shape[0]:
                        break
                html_struct_recon.append("<td")
                html_struct_recon.append(" rowspan=\"%s\"" % str(rowspan))
                html_struct_recon.append(">")
                html_struct_recon.append("</td>")
                texts_insert = texts_tokens[int(area_extend[height, width]) - 1] if int(
                    area_extend[height, width]) >= 1 else [""]
                text_tokens_recon.append({'tokens': texts_insert})

                # caculate the number of "</thead>" and "<body>" in this row
                if int(area_extend[height, width]) < 1:
                    pass
                elif labels[int(area_extend[height, width]) - 1][0]:
                    numbody += 1
                elif not labels[int(area_extend[height, width]) - 1][0]:
                    numhead += 1
                width += 1

            # td 同时有行、列合并时先行搜索再列搜索（添加: <td, rowspan='rowspan', colspan='colspan', >,</td>）
            elif area_extend[height, width] == area_extend[height + 1, width] and area_extend[height, width] == \
                    area_extend[height, width + 1]:
                rowspan = 1
                while area_extend[height, width] == area_extend[height + rowspan, width]:
                    rowspan += 1
                    if height + rowspan == area.shape[0]:
                        break
                html_struct_recon.append("<td")
                html_struct_recon.append(" rowspan=\"%s\"" % str(rowspan))
                colspan = 1
                while area_extend[height, width] == area_extend[height, width + colspan]:
                    colspan += 1
                    if (width + colspan) == (area.shape[1]):
                        break
                html_struct_recon.append(" colspan=\"%s\"" % str(colspan))
                html_struct_recon.append(">")
                html_struct_recon.append("</td>")
                texts_insert = texts_tokens[int(area_extend[height, width]) - 1] if int(
                    area_extend[height, width]) >= 1 else [""]
                text_tokens_recon.append({'tokens': texts_insert})

                # caculate the number of "</thead>" and "<body>" in this row
                if int(area_extend[height, width]) < 1:
                    pass
                elif labels[int(area_extend[height, width]) - 1][0]:
                    numbody += 1
                elif not labels[int(area_extend[height, width]) - 1][0]:
                    numhead += 1
                width += colspan

        html_struct_recon.append("</tr>")
        if numhead > numbody:
            headend = height + 1

    # insert '<thead>', '</thead>', '<tbody>' and '</tbody>'
    rowindex = [ind for ind, td in enumerate(html_struct_recon) if td == '</tr>']
    if headend:
        html_struct_recon.insert(rowindex[headend - 1] + 1, '</thead>')
        html_struct_recon.insert(rowindex[headend - 1] + 2, '<tbody>')
    else:
        trindex = html_struct_recon.index('</tr>')
        html_struct_recon.insert(trindex + 1, '</thead>')
        html_struct_recon.insert(trindex + 2, '<tbody>')
    html_struct_recon.insert(0, '<thead>')
    html_struct_recon.append('</tbody>')

    return html_struct_recon, text_tokens_recon


def format_html(html_struct, text_tokens):
    """ Formats HTML code from structure html and text tokens

    Args:
        html_struct (list(str)): structure html
        text_tokens (list(dict)): text tokens

    Returns:
        str: The final html of table.
    """

    html_code = html_struct.copy()
    to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]
    for i, cell in zip(to_insert[::-1], text_tokens[::-1]):
        # 非空单元格进行识别文字插入
        if cell['tokens']:
            cell = [escape(token) if len(token) == 1 else token for token in cell['tokens']]
            cell = ''.join(cell)
            html_code.insert(i + 1, cell)
    html_code = ''.join(html_code)
    html_code = '''<html><body><table>%s</table></body></html>''' % html_code

    return html_code


def bbox2adj(bboxes_non):
    """Calculating row and column adjacent relationships according to bboxes of non-empty aligned cells
    https://www.cnblogs.com/dan-baishucaizi/articles/15701739.html
    Args:
        bboxes_non(np.array): (n x 4).bboxes of non-empty aligned cells

    Returns:
        np.array: (n x n).row adjacent relationships of non-empty aligned cells
        np.array: (n x n).column adjacent relationships of non-empty aligned cells
    """
    # 分别声明行、列的邻接矩阵
    adjr = np.zeros([bboxes_non.shape[0], bboxes_non.shape[0]], dtype='int')
    adjc = np.zeros([bboxes_non.shape[0], bboxes_non.shape[0]], dtype='int')
    # 分别计算每个box的横坐标、纵坐标中心
    x_middle = bboxes_non[:, ::2].mean(axis=1)
    y_middle = bboxes_non[:, 1::2].mean(axis=1)
    for i, box in enumerate(bboxes_non):
        # 任意一个box若y_middle落入其中，则是同行；若x_middle落入其中则是同列
        indexr = np.where((bboxes_non[:, 1] < y_middle[i]) & (bboxes_non[:, 3] > y_middle[i]))[0]
        indexc = np.where((bboxes_non[:, 0] < x_middle[i]) & (bboxes_non[:, 2] > x_middle[i]))[0]
        # 邻接矩阵是对称的
        adjr[indexr, i], adjr[i, indexr] = 1, 1
        adjc[indexc, i], adjc[i, indexc] = 1, 1

        # 确定是否存在特殊的行关系
        for j, box2 in enumerate(bboxes_non):
            # (box2[1] + 4 >= box[3] or box[1] + 4 >= box2[3])参考下面图示1，保证了两个框在行维度上有交集
            if not (box2[1] + 4 >= box[3] or box[1] + 4 >= box2[3]):
                indexr2 = np.where((max(box[1], box2[1]) < y_middle[:]) & (y_middle[:] < min(box[3], box2[3])))[0]
                if len(indexr2):  # 参考图示2
                    adjr[j, i], adjr[i, j] = 1, 1

        # Determine if there are special column relationship
        for j, box2 in enumerate(bboxes_non):
            if not (box2[0] + 0 >= box[2] or box[0] + 0 >= box2[2]):
                indexc2 = np.where((max(box[0], box2[0]) < x_middle[:]) & (x_middle[:] < min(box[2], box2[2])))[0]
                if len(indexc2):
                    adjc[j, i], adjc[i, j] = 1, 1

    return adjr, adjc


class BasePostDetector(object):
    """Base method of post-processing for detectors. Contains method of
       post_processing(batch_result):do post processing
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def post_processing(self, batch_result, **kargs):
        """ Abstract method need to be implemented"""
        pass

    def __call__(self, batch_result, **kargs):
        """ Main process of post processing"""
        return self.post_processing(batch_result, **kargs)


@POSTPROCESS.register_module()
class Box2Html(BasePostDetector):
    """
        Get the format html of table获取格式化的表格html代码
    """

    def __init__(self,
                 nms_inter=True,
                 nms_threshold=0.3
                 ):
        """
        Args:
            nms_inter(bool): 是否进行非极大值抑制.
            nms_threshold(float): nms阈值
        """

        super().__init__()
        self.nms_inter = nms_inter
        self.nms_threshold = nms_threshold

    def post_processing(self, batch_result, **kwargs):
        """
        Args:
            batch_result(list(Tensor)): prediction results,
                like [(box_result, seg_result, local_pyramid_masks, global_pyramid_masks), ...]
            **kwargs: other parameters

        Returns:
            list(str): Format results, like [html of table1 (str), html of table2 (str), ...]
        """

        table_results = []
        for result in batch_result:     # 针对batch中的多个图片单独进行处理
            table_result = dict()
            bboxes_results = result[0]
            if self.nms_inter:          # 进行非极大值抑制
                bboxes, labels = nms_inter_classes(bboxes_results, self.nms_threshold)
                labels = [[lab[0]] for lab in labels]
            else:
                bboxes, labels = bboxes_results[0], [[0]] * len(bboxes_results[0])
                for cls in range(1, len(bboxes_results)):
                    bboxes = np.concatenate((bboxes, bboxes_results[cls]), axis=0)
                    labels += [[cls]] * len(bboxes_results[cls])
            bboxes = [list(map(round, b[0:4])) for b in bboxes]   # 坐标进行截断，使其为整数
            bboxes_np = np.array(bboxes)

            # 分别计算行、列的邻接矩阵
            adjr, adjc = bbox2adj(bboxes_np)

            # 通过行列邻接矩阵、图论分别生成每个节点属于哪一列、哪一行(每个最大团标识一行或者一列, 跨行跨列则对应多个最大团)
            colspan = adj_to_cell(adjc, bboxes_np, 'col')
            rowspan = adj_to_cell(adjr, bboxes_np, 'row')
            cells_non = [[row.min(), col.min(), row.max(), col.max()] for col, row in zip(colspan, rowspan)]
            # 行列划分后，cells_non记录了每个节点的左上角右下角坐标（坐标的值是行列编码）
            cells_non = np.array([list(map(int, cell)) for cell in cells_non])

            # 搜索空单元格并使用arearec记录
            arearec = np.zeros([cells_non[:, 2].max() + 1, cells_non[:, 3].max() + 1])  # 根据行数、列数初始化
            # cellid对应着box编码、rec标识矩形区域
            for cellid, rec in enumerate(cells_non):
                srow, scol, erow, ecol = rec[0], rec[1], rec[2], rec[3]
                arearec[srow:erow + 1, scol:ecol + 1] = cellid + 1        # 这里box编码是从1开始的
            empty_index = -1  # deal with empty cell 记录了有多少个空白单元格
            for row in range(arearec.shape[0]):
                for col in range(arearec.shape[1]):
                    if arearec[row, col] == 0:
                        arearec[row, col] = empty_index
                        empty_index -= 1
            # # arearec有token区域使用的是box编码填充、空单元格区域使用负数填充

            # Generate html of each table.
            # texts_tokens = [[""]] * len(labels)  # 如果使用文本识别结果，则最终的html可用。
            texts_tokens = [[f"{i}"] for i in range(len(labels))]         # 使用bbox编号进行单元格占位
            # html_str_rec表格结构代码，html_text_rec是每个token对应的text
            html_str_rec, html_text_rec = area_to_html(arearec, labels, texts_tokens)
            # format_html进行格式化，插入token
            table_result['html'] = format_html(html_str_rec, html_text_rec)
            table_result['bboxes'] = bboxes
            table_result['labels'] = labels
            table_results.append(table_result)

        return table_results

