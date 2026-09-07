<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  export let disabled = false;
  let url = "";
  let loaded = false;
  let saving = false;
  let message = "";
  async function load() {
    if (loaded) return;
    try { url = await invoke<string>("video_service_settings", {url: null}); loaded = true; }
    catch (e) { message = String(e); }
  }
  async function save() {
    saving = true;
    try { url = await invoke<string>("video_service_settings", {url}); message = "地址已保存，下次生成使用。"; }
    catch (e) { message = String(e); }
    finally { saving = false; }
  }
</script>

<details on:toggle={(event) => { if (event.currentTarget.open) void load(); }}>
  <summary>ComfyUI 服务配置</summary>
  <label>服务地址<input bind:value={url} disabled={disabled || saving} placeholder="http://127.0.0.1:8188" /></label>
  <button disabled={disabled || saving || !loaded} on:click={save}>{saving ? "保存中…" : "保存服务地址"}</button>
  <p>连接原 MiniMax H3 工作流所在的 ComfyUI。环境变量中的地址优先于此设置；保存不会提交生成任务。</p>
  {#if message}<p role="status">{message}</p>{/if}
</details>
