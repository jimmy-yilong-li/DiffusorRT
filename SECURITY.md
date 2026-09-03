# Security Policy

## Supported Versions

DiffusorRT is pre-1.0. Security fixes are provided for the latest tagged release and
the current `main` branch. Older pre-1.0 releases may require upgrading rather
than receiving a backport.

## Reporting A Vulnerability

Do not open a public issue for an unpatched vulnerability. Use GitHub's private
security-advisory reporting for this repository:

<https://github.com/jimmy-yilong-li/DiffusorRT-Source/security/advisories/new>

Include the affected version or commit, platform, package format, reproduction
steps, impact, and any known mitigation. Avoid attaching model weights,
credentials, private prompts, or other sensitive data.

The maintainers will acknowledge a complete report when it is reviewed, assess
scope and severity, and coordinate disclosure after a fix or mitigation is
available. This project does not promise a fixed response-time SLA.

## Security Boundary

Execution packages, manifests, tokenizer artifacts, graphs, snapshots, and
cache payloads are untrusted inputs. Supported paths validate identity, size,
schema, and package-relative references before use. A bypass, traversal,
unbounded read, stale-cache reuse, or unsafe deserialization issue is
security-relevant even if the default path is unaffected.

Model output quality, hallucination, and prompt injection are important product
risks but are not, by themselves, vulnerabilities in the runtime. Reports that
show code execution, data exposure, integrity loss, resource exhaustion, or a
contract bypass should use the private channel above.
