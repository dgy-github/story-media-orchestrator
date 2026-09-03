<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  let config = { story_model: "configured-by-story-runtime", image_model: "wan2.2-t2i-flash", image_size: "720*1280", video_model: "minimax_h3_fl2va", comfyui_url: "http://127.0.0.1:8000", sidecar_url: "http://127.0.0.1:8765" };
  let keys = { dashscope: "", sidecar: "", capability: "" };
  let status = "就绪"; let run: any = null;
  async function save() { await invoke("save_settings", { settings: config, credentials: keys }); status = "已保存到本机"; }
  async function generateToken() { keys.sidecar = await invoke("generate_sidecar_token"); status = "已生成 sidecar token"; }
  async function start() { status = "已提交调度"; run = await invoke("start_media_run", { storyInput: "single-scene" }); poll(); }
  async function poll() { if (!run) return; run = await invoke("get_media_run", { runId: run.run_id }); if (run.status !== "succeeded" && run.status !== "failed") setTimeout(poll, 700); }
</script>
<main>
  <header><div><h1>Story Media Orchestrator</h1><p>Rust/Tauri 统一调度 · 故事 → 图片 → 视频</p></div><span class="badge">{status}</span></header>
  <section class="grid">
    <div class="card"><h2>模型与服务</h2>
      <label>故事模型<input bind:value={config.story_model}></label><label>图片模型<input bind:value={config.image_model}></label><label>图片尺寸<input bind:value={config.image_size}></label><label>视频模型<input bind:value={config.video_model}></label><label>ComfyUI URL<input bind:value={config.comfyui_url}></label><label>Sidecar URL<input bind:value={config.sidecar_url}></label>
    </div>
    <div class="card"><h2>API key / 凭据</h2><label>DashScope API key<input type="password" bind:value={keys.dashscope}></label><label>Sidecar token<input type="password" bind:value={keys.sidecar}></label><label>Capability token<input type="password" bind:value={keys.capability}></label><div class="actions"><button on:click={generateToken}>自动生成 sidecar token</button><button class="primary" on:click={save}>保存配置</button></div><small>如果你之前在 BugleCat / .nanocodex 配过 vl_api_key，DashScope 可以留空，Rust 保存时会自动复用。</small></div>
  </section>
  <section class="card run"><div class="runhead"><h2>运行调度</h2><button class="primary" on:click={start}>开始单场景运行</button></div>
    {#if run}<div class="stages">{#each run.stages as stage}<div class:active={stage.state === "running"} class:done={stage.state === "succeeded"} class="stage"><strong>{stage.name}</strong><span>{stage.state}</span>{#if stage.artifact}<code>{stage.artifact}</code>{/if}{#if stage.retryable}<button on:click={() => status = `已请求重试：${stage.name}`}>重试</button>{/if}</div>{/each}</div>{:else}<p class="muted">尚未运行。点击开始后，Rust 调度器会按阶段记录状态和 artifact。</p>{/if}
  </section>
</main>
