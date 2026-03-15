<div align="center">

<img src="./docs/images/banner.png" width="320px"  alt="banner"/>

<h2 id="title">PDFMathTranslate</h2>

</div>

---

> [!IMPORTANT]
> This repository is our working fork of PDFMathTranslate for self-hosted use.
>
> - Primary deployment target: Ubuntu/Debian-based Linux Docker containers.
> - Future changes should prioritize Docker build reproducibility, runtime stability, and headless/container-friendly behavior.
> - Windows EXE and macOS-specific workflows are out of scope unless they affect shared code paths.
> - When installation or runtime behavior changes, update Docker/build documentation first.


PDF scientific paper translation and bilingual comparison. Based on [BabelDOC](https://github.com/funstory-ai/BabelDOC). Additionally, this project is also the official reference implementation for calling BabelDOC to perform PDF translation.

- 📊 Preserve formulas, charts, table of contents, and annotations _([preview](#preview))_.
- 🌐 Support [multiple languages](https://pdf2zh-next.com/supported_languages.html), and diverse [translation services](https://pdf2zh-next.com/advanced/Documentation-of-Translation-Services.html).
- 🤖 Provides [commandline tool](https://pdf2zh-next.com/getting-started/USAGE_commandline.html), [interactive user interface](https://pdf2zh-next.com/getting-started/USAGE_webui.html), and [Docker](https://pdf2zh-next.com/getting-started/INSTALLATION_docker.html)

<!-- Feel free to provide feedback in [GitHub Issues](https://github.com/PDFMathTranslate-next/PDFMathTranslate-next/issues) or [Telegram Group](https://t.me/+Z9_SgnxmsmA5NzBl). -->

> [!WARNING]
>
> This project is provided "as is" under the [AGPL v3](https://github.com/PDFMathTranslate-next/PDFMathTranslate-next/blob/main/LICENSE) license, and no guarantees are provided for the quality and performance of the program. **The entire risk of the program's quality and performance is borne by you.** If the program is found to be defective, you will be responsible for all necessary service, repair, or correction costs.
>
> Due to the maintainers' limited energy, we do not provide any form of usage assistance or problem-solving. Related issues will be closed directly! (Pull requests to improve project documentation are welcome; bugs or friendly issues that follow the issue template are not affected by this)

<h2 id="preview">Preview</h2>

<div align="center">
<!-- <img src="./docs/images/preview.gif" width="80%"  alt="preview"/> -->
<img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

<h2 id="install">Installation and Usage</h2>

### Installation

1. [**Docker**](https://pdf2zh-next.com/getting-started/INSTALLATION_docker.html) <small>Recommand for Linux</small>
2. [**uv** (a Python package manager)](https://pdf2zh-next.com/getting-started/INSTALLATION_uv.html) <small>Recommand for macOS</small>

---

### Build On Debian amd64 Linux

For this fork, the primary build target is Debian/Ubuntu-like amd64 Linux.

#### Recommended: build the Docker image

```bash
git clone <your-fork-url>
cd PDFMathTranslate-Self
docker build -t pdfmathtranslate-self .
docker run --rm pdfmathtranslate-self pdf2zh --version
docker run --rm -p 7860:7860 pdfmathtranslate-self
```

Then open:

```text
http://localhost:7860/
```

#### Alternative: build and run from source on Debian amd64

Install system dependencies:

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends -y \
  python3 python3-venv python3-pip \
  build-essential \
  libgl1 libglib2.0-0 libxext6 libsm6 libxrender1
```

Create a local environment and install from this repository:

```bash
python3 -m pip install --user uv
uv venv
. .venv/bin/activate
uv pip install --no-cache -r pyproject.toml
uv pip install --no-cache .
babeldoc --warmup
pdf2zh --version
pdf2zh --gui
```

#### Current build notes

- The first build can take a while because BabelDOC assets are warmed up during installation.
- The current Docker build needs outbound network access for Python packages and BabelDOC-related asset downloads.
- If you only need a quick smoke test after building, run `pdf2zh --version` before starting the WebUI.

---

### Usage

1. [Using **WebUI**](https://pdf2zh-next.com/getting-started/USAGE_webui.html)
2. [Using **Zotero Plugin**](https://github.com/guaguastandup/zotero-pdf2zh) (Third party program)
3. [Using **Commandline**](https://pdf2zh-next.com/getting-started/USAGE_commandline.html)

For different use cases, we provide distinct methods to use our program. Check out [this page](./getting-started/getting-started.md) for more information.

<h2 id="usage">Advanced Options</h2>

For detailed explanations, please refer to our document about [Advanced Usage](https://pdf2zh-next.com/advanced/advanced.html) for a full list of each option.

<h2 id="downstream">Secondary Development (APIs)</h2>

<!-- <!-- For downstream applications, please refer to our document about [API Details](./docs/APIS.md) for futher information about: -->

- [Python API](./docs/en/advanced/API/python.md), how to use the program in other Python programs
<!-- - [HTTP API](./docs/APIS.md#api-http), how to communicate with a server with the program installed -->

<h2 id="langcode">Language Code</h2>

If you don't know what code to use to translate to the language you need, check out [this documentation](https://pdf2zh-next.com/advanced/Language-Codes.html)

<h2 id="acknowledgement">Acknowledgements</h2>

- [Immersive Translation](https://immersivetranslate.com) sponsors monthly Pro membership redemption codes for active contributors to this project, see details at: [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md)

- [SiliconFlow](https://siliconflow.cn) provides a free translation service for this project, powered by large language models (LLMs).

- 1.x version: [Byaidu/PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)


- backend: [BabelDOC](https://github.com/funstory-ai/BabelDOC)

- PDF Library: [PyMuPDF](https://github.com/pymupdf/PyMuPDF)

- PDF Parsing: [Pdfminer.six](https://github.com/pdfminer/pdfminer.six)

- PDF Preview: [Gradio PDF](https://github.com/freddyaboulton/gradio-pdf)

- Layout Parsing: [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

- PDF Standards: [PDF Explained](https://zxyle.github.io/PDF-Explained/), [PDF Cheat Sheets](https://pdfa.org/resource/pdf-cheat-sheets/)

- Multilingual Font: see [BabelDOC-Assets](https://github.com/funstory-ai/BabelDOC-Assets)

- [Asynchronize](https://github.com/multimeric/Asynchronize/tree/master?tab=readme-ov-file)

- [Rich logging with multiprocessing](https://github.com/SebastianGrans/Rich-multiprocess-logging/tree/main)

- Documentation i18n using [Weblate](https://hosted.weblate.org/projects/pdfmathtranslate-next/) 


<h2 id="conduct">Before submit your code</h2>

We welcome the active participation of contributors to make pdf2zh better. Before you are ready to submit your code, please refer to our [Code of Conduct](https://pdf2zh-next.com/community/CODE_OF_CONDUCT.html) and [Contribution Guide](https://pdf2zh-next.com/community/Contribution-Guide.html).
