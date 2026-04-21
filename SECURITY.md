# Security policy

## Reporting a vulnerability

Please **do not** open public GitHub issues for security reports. Instead, email **security@stackone.com** with:

- A description of the issue and the version of stackvox affected.
- Steps to reproduce or a proof-of-concept, if available.
- Any known mitigations.

We aim to acknowledge reports within 3 business days and to provide a remediation timeline within 10 business days of the initial acknowledgement.

## Scope

In scope:

- The stackvox library and CLI (`stackvox`, `stackvox-say`).
- The daemon's unix-socket protocol and file-permission handling.
- Supply-chain issues in the immediate `stackvox` package (e.g. typo-squat risk, malicious publish).

Out of scope (report upstream instead):

- Vulnerabilities in Kokoro-82M weights or voice files — report to [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M).
- Vulnerabilities in `kokoro-onnx`, `onnxruntime`, `phonemizer-fork`, `sounddevice`, `soundfile`, or `numpy` — report to the respective project.

## Supported versions

stackvox is pre-1.0. Only the latest published release receives security fixes. Once 1.0 ships, we will maintain the most recent minor release of the current major version.
