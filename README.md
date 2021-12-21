# BBox2Html
Structuring bbox into HTML code 将边框结构化为html代码

这个repo主要记录了LGPMA算法的bbox转化为html的过程，适应于表格边框检测格式化。接口的入口为：

```python
from utils.PostProcessing import Box2Html
```

所有的逻辑都在**PostProcessing.py**文件中，主要的思路已经使用中文进行了注释。

