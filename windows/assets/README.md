# Windows interface preview

`interface-preview.png` shows the actual PC chat frontend shipped in SDK
0.2.6, rendered on Mac with a documentation-only event fixture.

- The conversation, recent items and masked-sequence events are sample data.
- No model, CUDA runtime or device measurement was run.
- Speed, reply latency and GPU memory are not reported. The writing progress
  belongs to the sample event, not a measured generation.
- The blue preview label is a documentation overlay. The interface, layout
  and controls use the existing PC HTML/CSS/JavaScript.
- This is not a screenshot taken on Windows. A real Windows capture can
  replace it when supplied by the platform maintainer.

Frontend identity: SDK source `cad923c2`; `app.js` SHA-256
`ab3bfcc02ccfd0e1bd5cc28ac0b6110d964c999ca21c0821712b10cb94bac182`.
The image is not evidence of model quality, CUDA performance or Windows
release acceptance.
