# Streamlit 部署包（7 变量 RSF）

这个文件夹是给 Streamlit Community Cloud / GitHub 用的完整可运行副本。

## 内容

| 文件 | 说明 |
|---|---|
| `streamlit_app.py` | 应用脚本（已改为 7 变量、300 次嵌套验证、加免责声明与模型信息面板） |
| `streamlit_model.joblib` | 1.9 MB，压缩存储；预测值与未压缩版逐位一致（已核对 max abs diff = 0.0） |
| `requirements.txt` | 依赖清单（含 scikit-survival） |

## 模型是什么

- 随机生存森林（RSF）：100 棵树、`min_samples_split=20`、`min_samples_leaf=5`、`max_depth=None`
- 7 个预测子，全部来自 QFR/RWS 离线分析：
  `rws_percent`、`rwsqfr_delta`、`QFR_stent`、`stenosis_percent_stent`、`delta_qfr`、`rws`、`ES_QFR`
- 变量筛选：LASSO（λ.min、5 折）在 300 次嵌套重抽样里各自重做一次，保留入选频率 ≥ 0.65 的变量
- 性能（来自 `nested_bootstrap.py`，不是本应用）：袋外 C-index 0.659（百分位区间 0.578–0.744），1 年 AUC 0.740，.632+ 0.702
- 元数据 `meta` 里记录了种子、λ 规则、折数、筛选方式与事件数

## 本地先跑一遍（可选）

```powershell
cd D:\paper\QFR\final\review_work\deploy_streamlit
D:\anacoda\python.exe -m streamlit run streamlit_app.py
```

默认开在 http://localhost:8501。

## 部署到 Streamlit Cloud

1. 把本文件夹的三个文件放进 GitHub 仓库（例如 `Reinct/QFR` 的 `app/` 目录）。
2. 在 share.streamlit.io 新建或编辑应用：
   - Main file path：`app/streamlit_app.py`
   - Python version：3.11 或 3.12（scikit-survival 0.27 需要 3.10+）
3. **重要**：应用里有一段兜底逻辑——如果同目录找不到 `streamlit_model.joblib`，会从
   `https://github.com/Reinct/QFR/releases/latest/download/streamlit_model.joblib` 下载。
   现在的 release 附件还是旧的九变量模型（约 105 MB），必须用本文件夹里的 1.9 MB 新版替换，
   否则线上会继续跑旧模型。两种做法任选：
   - 把 joblib 提交进仓库（1.9 MB 可以直接进 Git），这样就永远走本地文件；
   - 或者把 release 附件换成新版。
4. 部署后在页面上按一次 Predict，确认显示 2 年风险与 SHAP 瀑布图。

## 我已经验证过什么

- 加载模型、7 个输入框、预测、SHAP 瀑布图全部跑通（`streamlit.testing.AppTest`）：
  - 初始渲染无异常；
  - 点击 Predict 无异常；
  - 默认输入下输出 2 年 ISR 风险 31.1%、"Moderate risk"、队列 2 年事件率 42.7%。
- 健康检查端点返回 200（`/_stcore/health`）。

## 仓库里那份旧模型

`D:\paper\QFR\final\ml_survival_analysis\streamlit_model.joblib` 仍是旧的九变量版本——
该目录的 ACL 不允许我写入（`PermissionError: [Errno 13]`）。需要你手动替换：

```powershell
Copy-Item D:\paper\QFR\final\review_work\deploy_streamlit\streamlit_model.joblib `
          D:\paper\QFR\final\ml_survival_analysis\streamlit_model.joblib -Force
```

同时 `ml_survival_analysis/streamlit_app.py` 也建议换成这里的版本（或至少把
"100 iterations" 改成 300、把标题与免责声明同步）。
