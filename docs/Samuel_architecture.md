# Atari Pong DQN architecture

![DQN architecture](Samuel_architecture.png)

Regenerate PNG/SVG with `uv run python docs/draw_architecture.py`.

This diagram describes the implementation in `train.py` and `play.py`.
`B` is the minibatch size and `A` is `env.action_space.n` (six in the tested Pong setup).

```mermaid
flowchart TD
    pong["ALE/Pong-v5: RGB frames"] --> preprocess["AtariPreprocessing: frame skip 4, grayscale, resize 84 x 84"]
    preprocess --> stack["FrameStackObservation: uint8, 4 x 84 x 84"]
    stack --> policy["Online CNN: epsilon-greedy action"]
    policy --> pong
    stack --> replay["Replay: state, action, reward, next state, terminated"]
    replay --> batch["Sample B transitions on DEVICE"]
    batch --> online["Online CNN: gather Q of sampled action, shape B"]
    batch --> target["Target CNN: max over next-state actions, shape B; no gradients"]
    target --> bellman["Target = reward + gamma * next Q * (1 - terminated)"]
    online --> loss["Smooth L1 loss; gradient norm clipping at 10; Adam"]
    bellman --> loss
    loss --> policy
    policy -->|"Hard copy every 1000 environment decisions by default"| target
    pong --> logger["Finished episodes: raw reward, length, cumulative steps, epsilon, latest loss to CSV"]
    policy --> save["Periodic and final online state_dict saved atomically"]
    save --> play["play.py: matching CNN and preprocessing, greedy evaluation, optional MP4 capture"]
```

## Shared CNN for online and target networks

```mermaid
flowchart LR
    input["B x 4 x 84 x 84, pixels 0 to 255"] --> normalize["float32 / 255"]
    normalize --> convOne["Conv 4 to 32, kernel 8, stride 4; ReLU; B x 32 x 20 x 20"]
    convOne --> convTwo["Conv 32 to 64, kernel 4, stride 2; ReLU; B x 64 x 9 x 9"]
    convTwo --> convThree["Conv 64 to 64, kernel 3, stride 1; ReLU; B x 64 x 7 x 7"]
    convThree --> flatten["Flatten: B x 3136"]
    flatten --> hidden["Linear 3136 to 512; ReLU"]
    hidden --> output["Linear 512 to A; Q-values B x A"]
```

Divine owns preprocessing and the CNN; Gabriel owns replay, exploration and
scheduling; Samuel owns gradient updates, synchronization, logging, checkpoint
saving, evaluation and recording. The training loop resets on termination or
truncation; only true termination masks the Bellman bootstrap. Replay images
must not be normalized twice. Saved weights support inference, not exact
training resumption.

## Integration with main (13 November 2026)

`ReplayBuffer.sample()` now returns uint8 images, and `train_step(online_net,
target_net, optimizer, batch, gamma)` consumes the sampled tuple directly.
The CNN now uses Kaiming initialization. The main loop uses Adam with
`eps=1.5e-4`, defaults to training every decision, and saves weights periodically
as well as at completion. `--model_path` is honored by atomic checkpoint saving.
Pong rewards are stored as supplied by the environment; no sign conversion is
applied by the current loop.
