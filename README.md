# lstmForGPU
方便我把代码传到实验室。

### 当前问题描述
    
    model.add(GRU(xxx))

这行代码报错。

根据查阅资料。这个问题在[github](https://github.com/fchollet/keras/commit/1de4bf1b5989f76e377f0b8022b41773a354ba99)这里得到了解决。

但问题是目前我没有权限能够修改。我尝试了不修改源文件的方式，但无果。

所以只能等待学长帮我把keras版本升级到2.0.4。
