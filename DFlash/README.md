<div align="center">

<sub><strong>DIFFUSORRT RESEARCH &nbsp; / &nbsp; MOBILE INFERENCE</strong></sub>
<h1>DFlash on iPhone</h1>
<p><strong>A diffusion model drafts. A 4B language model verifies. Both fit on an iPhone.</strong></p>

[On-device results](#on-device-results) · [Cross-Accelerator Inference](#cross-accelerator-inference) · [DiffusorRT](../README.md)

</div>

The hard part of speculative decoding on a phone is making the second model earn its place: drafting is extra work, and the user only benefits if the verified reply arrives sooner. DiffusorRT integrates [DFlash](https://github.com/z-lab/dflash) into its iPhone chat app. Add a **303 MB drafter** to the same **4-bit Qwen3.5-4B** target and switch between autoregressive chat and DFlash without downloading a second copy of the target. Both models run on the phone.

## On-device results

| iPhone 15 Pro · time from Send to completed reply | Autoregressive | DFlash | Speedup |
|:---|---:|---:|---:|
| Make a deadline request sound polite | 4.55 s | **2.73 s** | **1.67×** |
| Work out a train’s average speed | 4.64 s | **2.26 s** | **2.05×** |

Both modes produced the same tokens for each prompt. These were two warm comparisons in the **Metal** app, one run per mode and prompt with a 64-token limit. The gain was in **finishing the reply**, not in showing its first word.

## Cross-Accelerator Inference

**Next milestone: draft on the Apple Neural Engine, verify on the Metal GPU.** The Neural Engine proposes a block of words; the target model on the GPU checks that block in one pass. The idea is to free the GPU from drafting so it can spend its time verifying more text. The 4-bit Neural Engine drafter is compiled and running on Apple Silicon. Measuring the two accelerators working together on iPhone is next.

*iPhone app 1.1 is in developer preview.* [Original DFlash project](https://github.com/z-lab/dflash) · [Mobile target](https://huggingface.co/jimmylearnML/DiffusorRT-iPhone-Qwen3.5-4B-Q4) · [Mobile drafter](https://huggingface.co/jimmylearnML/DiffusorRT-iPhone-Qwen3.5-4B-DFlash-Q4)
