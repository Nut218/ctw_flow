# 原始观测残差条件 Flow + 轻量联合损失训练

从 2026-09-22 的历史 apply_patch 操作恢复，而非重新设计的训练脚本。基础版本为 Git cd53ddc；恢复 07:50:39、07:51:31、07:52:01 UTC 的三组训练相关补丁。原始轻量版本训练命令记录于 07:59:52 UTC。

参数：环秩 12/12/12，核秩 3/3/3，1500 步，batch 16，学习率 0.0002，终点损失 0.1，匹配损失 0.05，TW 重建损失 0.02，观测损失 0.01，观测特征维度 68。

运行：在安装 torch、numpy、scipy、tensorly 的 Python 环境中执行 `powershell -ExecutionPolicy Bypass -File .\train_original.ps1`。训练结果写入本目录 outputs，不覆盖原权重。脚本保留原始语料及 SRF 的绝对路径；迁移电脑时修改 train_original.ps1 的 --code_pt 和 --srf。

原语料是 rank_factor_corpus_192/tw_codes.pt，含 CAVE、Pavia、Chikusei 的训练/验证因子，不是后续 CAVE-only 512 语料。训练使用训练场景的干净 TW 因子作为 Flow 先验目标，不属于完全零样本；测试阶段不使用测试 HR-HSI 作为优化目标。

对应单幅 256 测试记录 PSNR 43.46526391 dB；不代表数据集平均指标。恢复脚本尚未重新训练，不能保证重训结果逐位一致。
