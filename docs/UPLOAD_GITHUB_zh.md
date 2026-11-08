# 上传GitHub

本地整理不会自动创建仓库或发布内容。建议仓库名：`FPP-Net`。

## 1. 上传代码仓库

解压 `FPP-Net-github-source.zip`，进入里面的 `FPP-Net` 文件夹。上传该文件夹内部的文件，不要再嵌套一层同名目录。仓库包括MIT LICENSE、NOTICE、英文/中文README、代码、清单、18个检查点和结果。

使用Git时，在GitHub创建空仓库后执行（把YOUR_USERNAME替换为你的账号）：

```bash
git init
git add .
git commit -m "Release FPP-Net code and frozen evaluation protocols"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/FPP-Net.git
git push -u origin main
```

也可以通过网页上传解压后的源码文件。**不要将整个源码ZIP作为仓库唯一文件**，否则README和代码无法正常展示。

## 2. 上传三个数据Release附件

在仓库页面选择Releases→Draft a new release，标签建议 `v0.1.0`，上传：

1. `kaist-xm-processed-v1.zip`
2. `ceitec-xl-processed-v1.zip`
3. `kaist-audit-features-v1.zip`
4. 与它们同目录的 `SHA256SUMS.txt`

Release说明可写：

> Frozen derived evaluation features for KAIST-XM and CEITEC-XL, with original sample coordinates and record labels. Extract each ZIP at repository root. Data are derived from the original KAIST and CEITEC measurements under CC BY4.0; see NOTICE.md for attribution and modifications. Code is MIT-licensed. Original raw TDMS/MAT acquisitions are available from the source DOIs.

这些附件不要直接提交到Git历史。GitHub普通Git文件超过100MiB会被拒绝，浏览器上传仓库文件还有25MiB限制；Release适合分发较大二进制附件。
官方说明：[About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 3. 发布后补充链接

- README中的“Releases”可换成你仓库的实际Release链接。
- CITATION.cff可补 `repository-code` 字段及实际URL。
- 论文Data Availability可填你的仓库链接；若之后通过Zenodo归档软件获得DOI，再补DOI。
- 不要把原始数据作者的DOI当作你自己的数据重构或代码DOI。

## 4. 验证下载

从新仓库重新克隆，下载并解压两个主数据附件，然后运行：

```bash
python -m pip install -r requirements.txt
python run.py check
python run.py evaluate --device cpu
```

应得到KAIST记录级75.00%、CEITEC记录级100.00%。不同窗口级/记录级指标的含义见README与PROTOCOL。
