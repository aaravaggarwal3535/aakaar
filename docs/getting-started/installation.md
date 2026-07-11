# Installation

<div class="install-matrix">
  <div class="row">
    <div class="label">Your OS</div>
    <div class="options" data-group="os">
      <button class="opt active" data-value="windows">Windows</button>
      <button class="opt" data-value="linux">Linux</button>
    </div>
  </div>

  <div class="row">
    <div class="label">Package</div>
    <div class="options" data-group="package">
      <button class="opt active" data-value="pip">Pip</button>
    </div>
  </div>

  <div class="row">
    <div class="label">Python</div>
    <div class="options" data-group="python">
      <button class="opt active" data-value="cp310">3.10</button>
      <button class="opt" data-value="cp311">3.11</button>
      <button class="opt" data-value="cp312">3.12</button>
      <button class="opt" data-value="cp313">3.13</button>
      <button class="opt" data-value="cp314">3.14</button>
    </div>
  </div>

  <div class="row">
    <div class="label">Compute Platform</div>
    <div class="options" data-group="compute">
      <button class="opt active" data-value="cuda">CUDA</button>
      <button class="opt" data-value="cpu">CPU</button>
    </div>
  </div>

  <div class="row command-row">
    <div class="label">Run this Command</div>
    <div class="command-box">
      <code id="install-command">pip install aakaar</code>
      <button id="copy-btn" title="Copy">📋</button>
    </div>
  </div>
</div>

<style>
.install-matrix {
  border: 1px solid var(--md-default-fg-color--lightest, #e0e0e0);
  border-radius: 6px;
  overflow: hidden;
  margin: 1.5em 0;
  font-size: 0.9em;
}
.install-matrix .row {
  display: flex;
  align-items: center;
  border-bottom: 1px solid var(--md-default-fg-color--lightest, #e0e0e0);
}
.install-matrix .row:last-child { border-bottom: none; }
.install-matrix .label {
  flex: 0 0 160px;
  padding: 10px 16px;
  font-weight: 600;
  color: var(--md-default-fg-color--light, #555);
}
.install-matrix .options {
  display: flex;
  flex: 1;
}
.install-matrix .opt {
  flex: 1;
  padding: 10px 12px;
  border: none;
  background: var(--md-default-bg-color--light, #f5f5f5);
  cursor: pointer;
  font-size: 0.95em;
  color: var(--md-default-fg-color, #333);
  border-right: 1px solid var(--md-default-bg-color, #fff);
  transition: background 0.15s, color 0.15s;
}
.install-matrix .opt:last-child { border-right: none; }
.install-matrix .opt:hover { background: #6b5b56; }
.install-matrix .opt.active {
  background: #2caaee;
  color: #fff;
  font-weight: 600;
}
.install-matrix .opt.disabled {
  cursor: not-allowed;
  opacity: 0.4;
  pointer-events: none;
}
.install-matrix .command-row {
  background: var(--md-code-bg-color, #f8f8f8);
}
.install-matrix .command-box {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
}
.install-matrix .command-box code {
  font-family: var(--md-code-font, monospace);
  font-size: 0.85em;
  word-break: break-all;
}
.install-matrix #copy-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 1em;
  padding: 4px 8px;
  flex-shrink: 0;
}
</style>

<script>
(function() {
  const matrix = document.currentScript.previousElementSibling.classList.contains('install-matrix')
    ? document.currentScript.previousElementSibling
    : document.querySelector('.install-matrix');

  const state = { os: 'windows', package: 'pip', python: 'cp310', compute: 'cuda' };

  const GITHUB_REPO = "aaravaggarwal3535/aakaar-wheels";
  const RELEASE_TAG = "v0.1.11";
  const PKG_VERSION = "0.1.11";

  const LINUX_WHEELS = {
    cp310: `aakaar-${PKG_VERSION}-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`,
    cp311: `aakaar-${PKG_VERSION}-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`,
    cp312: `aakaar-${PKG_VERSION}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`,
    cp313: `aakaar-${PKG_VERSION}-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`,
    cp314: `aakaar-${PKG_VERSION}-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`,
  };

  function updateCommand() {
    const codeEl = matrix.querySelector('#install-command');
    if (state.os === 'windows') {
      codeEl.textContent = 'pip install aakaar';
    } else {
      const wheel = LINUX_WHEELS[state.python];
      codeEl.textContent = `pip install https://github.com/${GITHUB_REPO}/releases/download/${RELEASE_TAG}/${wheel}`;
    }
  }

  matrix.querySelectorAll('.options').forEach(group => {
    const groupName = group.dataset.group;
    group.querySelectorAll('.opt').forEach(btn => {
      btn.addEventListener('click', () => {
        if (btn.classList.contains('disabled')) return;
        group.querySelectorAll('.opt').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state[groupName] = btn.dataset.value;
        updateCommand();
      });
    });
  });

  matrix.querySelector('#copy-btn').addEventListener('click', () => {
    const text = matrix.querySelector('#install-command').textContent;
    navigator.clipboard.writeText(text);
  });

  updateCommand();
})();
</script>

## Prerequisites

* **Python Version:** 3.10, 3.11, 3.12, 3.13, or 3.14
* **Operating System:** Windows 10/11 (64-bit) or Ubuntu 22.04+ LTS (or other manylinux_2_28-compatible Linux distro)

### GPU Acceleration (Optional)

* **NVIDIA GPU** with a compatible driver installed
* **NVIDIA CUDA Toolkit** — bundled with the Windows wheel; Linux wheels also bundle the CUDA runtime, no separate install needed

If no CUDA installation is detected, aakaar automatically falls back to CPU-only execution.

> **Note:** Linux wheels are hosted outside PyPI due to file size limits, and the download link is pinned to a specific release version. If `v0.1.11` is out of date, check the [releases page](https://github.com/aaravaggarwal3535/aakaar-wheels/releases) for the latest tag.