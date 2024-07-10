<div align="center">

# AudioTools

个人常用的音频 AI 处理脚本\
Personal commonly used audio AI processing scripts

</div>

> [!NOTE]
>
> 1. 部分单文件`py`文件不提供依赖目录，请根据`import`自行安装
> 2. 所有的`py`文件都可以通过传入`-h`或`--help`参数获取使用帮助
> 3. 部分脚本是我本人写的，技术有限，请多包涵
> 4. 如果你有好的工具脚本或遇到问题，欢迎提交`issue`或`pull request`

工具列表：

| 名称                                                                       | 说明                                                                                             |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [audio_slicer.py](./audio_slicer.py)                                       | 音频切分工具                                                                                     |
| [loudness_matching](./loudness_matching/)                                  | 用于对音频文件进行响度匹配的工具，有命令行版和 GUI 版                                            |
| [make_rvc_pretrain_model.py](./make_rvc_pretrain_model.py)                 | 用于删除`optimizer`等网络层，制作 RVC 底模的工具                                                 |
| [make_sovits_pretrain_model.py](./make_sovits_pretrain_model.py)           | 用于删除`optimizer`, `emb_g.weight`等网络层，制作 sovits 底模的工具                              |
| [merge_audios.py](./merge_audios.py)                                       | 用于合并短音频到长音频的工具                                                                     |
| [sdr_measure.py](./sdr_measure.py)                                         | 用于测量两端音频（一般是原始音频和处理后音频）SDR 值的工具                                       |
| [sovits_datasets_batch_renaming.bat](./sovits_datasets_batch_renaming.bat) | sovits 数据集批量重命名，它将会自动重命名文件夹下所有音频文件为数字而不改变其排序，by 领航员未鸟 |
