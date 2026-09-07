<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  export let disabled = false;
  export let premise = "";
  export let savedJob: any = null;
  export let draft: any = null;
  const dispatch = createEventDispatcher();
  let templates: any[] = [];
  let providerStatus: any = null;
  let providerError = "";
  let credentialProvider = "deepseek";
  let credentialSecret = "";
  let savingCredential = false;
  let credentialMessage = "";
  async function saveCredential() {
    if (savingCredential || busy || !credentialSecret) return;
    savingCredential = true; credentialMessage = ""; providerError = "";
    try {
      await invoke("save_story_credential", {provider: credentialProvider, secret: credentialSecret});
      credentialMessage = "已保存到原项目的 Windows 凭据存储。若服务已经启动，新凭据在重启应用后生效。";
      await checkProviders();
    } catch (e) { providerError = String(e); }
    finally { credentialSecret = ""; savingCredential = false; }
  }
  async function checkProviders() {
    providerError = "";
    try { providerStatus = await invoke("story_provider_status"); }
    catch (e) { providerError = String(e); }
  }
  let template = "";
  let audience = "25-45";
  let constraintProfile = "short-vertical-v1";
  let episodes = 6;
  function applyConstraintProfile() {
    episodes = constraintProfile === "long-serial-v1" ? 40 : 6;
    minutes = 2;
    tokens = Math.max(300000, 120000 + episodes * 30000);
  }
  let minutes = 2;
  let limits = "";
  let tokens = 300000;
  let cost = 1200;
  let deadline = 900;
  let busy = false;
  let error = "";
  let storyPackage: any = null;
  function selectTemplate() { audience = templates.find(t => t.pack_id === template)?.default_audience || "25-45"; }
  let appliedSettings: any = undefined;
  function restoreSettings(restored: any) {
    template = templates.find(t => t.pack_id === "family-grounded-v1")?.pack_id || templates[0]?.pack_id || "";
    selectTemplate();
    constraintProfile = "short-vertical-v1"; episodes = 6; minutes = 2;
    limits = ""; tokens = 300000; cost = 1200; deadline = 900;
    if (restored) {
      template = restored.genre_pack_id || template;
      constraintProfile = restored.constraint_profile_id || constraintProfile;
      audience = restored.audience || audience;
      episodes = restored.format?.episodes ?? episodes;
      minutes = restored.format?.minutes_per_episode ?? minutes;
      limits = (restored.content_limits || []).join("\n");
      tokens = restored.budget?.max_tokens ?? tokens;
      cost = restored.budget?.max_cny_fen ?? cost;
      deadline = restored.budget?.deadline_seconds ?? deadline;
    }
    appliedSettings = restored;
    storyPackage = null;
  }
  $: if (templates.length && !busy && appliedSettings !== (draft || savedJob)) restoreSettings(draft || savedJob);
  async function load() {
    try {
      templates = await invoke("list_story_templates");
      restoreSettings(draft || savedJob);
      error = "";
    } catch (e) { error = "模板加载失败：" + String(e); }
  }
  onMount(load);
  onDestroy(() => {
    if (template) dispatch("draft", {genre_pack_id: template, constraint_profile_id: constraintProfile, audience,
      format: {episodes, minutes_per_episode: minutes},
      content_limits: limits.split("\n"),
      budget: {max_tokens: tokens, max_cny_fen: cost, deadline_seconds: deadline}});
  });
  function buildJob() {
      const selected = templates.find(t => t.pack_id === template);
      return {
        schema: "story-job/v1", job_id: "job_desktop_" + crypto.randomUUID(),
        content_form: "scripted_short_drama", input: premise,
        genre_mode: "fixed", allowed_genres: [selected.genre], genre_pack_id: template,
        constraint_profile_id: constraintProfile, audience,
        format: {episodes, minutes_per_episode: minutes},
        content_limits: limits.split("\n").map(s => s.trim()).filter(Boolean),
        budget: {max_tokens: tokens, max_cny_fen: cost, deadline_seconds: deadline}
      };
  }
  let checking = false;
  let checked = "";
  let checkedFingerprint = "";
  $: settingsFingerprint = JSON.stringify([template, constraintProfile, audience, episodes, minutes, limits, tokens, cost, deadline, premise]);
  $: if (checked && checkedFingerprint !== settingsFingerprint) checked = "";
  async function validate() {
    if (checking || busy || disabled) return;
    const fingerprint = settingsFingerprint;
    checking = true; error = ""; checked = "";
    try {
      const job = buildJob();
      await invoke("validate_template_story", {job});
      checkedFingerprint = fingerprint;
      checked = `设置检查通过：${job.format.episodes} 集，每集 ${job.format.minutes_per_episode} 分钟；未调用模型。`;
    } catch (e) { error = "设置检查未通过：" + String(e); }
    finally { checking = false; }
  }
  let retryJob: any = null;
  let retryFingerprint = "";
  const pendingStoryKey = "story-media-pending-stories-v1";
  let pendingStories: any[] = [];
  function readPendingStories() {
    const saved = JSON.parse(localStorage.getItem(pendingStoryKey) || "[]");
    if (!Array.isArray(saved) || saved.some(job => job?.schema !== "story-job/v1" || typeof job.job_id !== "string" || !job.job_id || typeof job.input !== "string")) throw new Error("故事任务记录格式错误");
    return saved;
  }
  function refreshPendingStories() {
    try { pendingStories = readPendingStories(); }
    catch { error = "故事任务记录无法读取，请检查本地存储。"; }
  }
  onMount(refreshPendingStories);
  async function generate(resumeJob: any = null) {
    if (busy || checking || disabled) return;
    busy = true; error = ""; dispatch("busy", true);
    try {
      const job = resumeJob || (retryJob && retryFingerprint === settingsFingerprint ? retryJob : buildJob());
      pendingStories = [job, ...readPendingStories().filter(item => item.job_id !== job.job_id)];
      localStorage.setItem(pendingStoryKey, JSON.stringify(pendingStories));
      retryJob = job; retryFingerprint = resumeJob ? "" : settingsFingerprint;
      const result: any = await invoke("generate_template_story", {job});
      const scenes = result.scenes || [];
      const text = scenes.map((s: any) => (Array.isArray(s.lines) ? s.lines : []).map((line: any) => typeof line?.text === "string" ? line.text.trim() : "").filter(Boolean).join("\n") || s.action || s.description || s.summary).filter(Boolean).join("\n");
      if (!text) throw new Error("故事包缺少可导入的场景文本，未替换原故事。");
      storyPackage = result; retryJob = null;
      pendingStories = readPendingStories().filter(item => item.job_id !== job.job_id);
      localStorage.setItem(pendingStoryKey, JSON.stringify(pendingStories));
      dispatch("generated", {text, package: result, job});
    } catch (e) { error = "故事生成失败，原内容已保留：" + String(e); }
    finally { busy = false; dispatch("busy", false); }
  }
</script>
<div class="story-template">
  <label>故事模板<select bind:value={template} on:change={selectTemplate} disabled={disabled || busy || !templates.length}><option value="" disabled>请选择模板</option>{#each templates as item}<option value={item.pack_id}>{item.display_name}</option>{/each}</select></label>
  <label>篇幅约束<select bind:value={constraintProfile} on:change={applyConstraintProfile} disabled={disabled || busy}><option value="short-vertical-v1">短篇 6–12 集</option><option value="long-serial-v1">长篇 40–80 集</option></select></label>
  <details><summary>高级参数</summary><div class="fields"><label>受众<input bind:value={audience} disabled={disabled || busy}></label><label>集数<input type="number" min={constraintProfile === "long-serial-v1" ? 40 : 6} max={constraintProfile === "long-serial-v1" ? 80 : 12} bind:value={episodes} disabled={disabled || busy}></label><label>每集分钟<input type="number" min={constraintProfile === "long-serial-v1" ? 2 : 1} max={constraintProfile === "long-serial-v1" ? 5 : 3} bind:value={minutes} disabled={disabled || busy}></label><label>Token 上限<input type="number" min="1" bind:value={tokens} disabled={disabled || busy}></label><label>费用上限（分）<input type="number" min="1" bind:value={cost} disabled={disabled || busy}></label><label>时限（秒）<input type="number" min="1" bind:value={deadline} disabled={disabled || busy}></label></div><label>内容限制<textarea bind:value={limits} disabled={disabled || busy} placeholder="每行一条"></textarea></label></details>
  <button disabled={disabled || busy || checking || !template || !premise.trim()} on:click={validate}>{checking ? "检查中…" : "检查设置"}</button>
  {#if checked}<p role="status">{checked}</p>{/if}
  <button disabled={disabled || busy || checking || !template || !premise.trim()} on:click={() => generate()}>{busy ? "正在生成故事…" : "按模板生成故事"}</button>
  <p>使用原故事服务生成，成功后将场景导入下方正文；不会自动生图。</p>
  <details on:toggle={refreshPendingStories}><summary>未取回的故事任务（{pendingStories.length}）</summary>
    <p>沿用原任务编号和提交参数取回结果。若原服务已停止执行，本操作不会重新生成故事。</p>
    {#each pendingStories as job}<div><strong>{job.genre_pack_id} · {job.format?.episodes} 集</strong><p>{job.input}</p><button disabled={disabled || busy || checking} on:click={() => generate(job)}>取回原任务结果</button></div>{:else}<p>暂无待取回任务。</p>{/each}
  </details>
  {#if disabled && !busy}<p>已创建项目的故事已锁定；修改模板生成新故事，请先载入新项目。</p>{/if}
  {#if storyPackage}<details><summary>完整故事包</summary><pre>{JSON.stringify(storyPackage, null, 2)}</pre></details>{/if}
  {#if error}<p role="alert">{error}</p>{#if !templates.length}<button on:click={load}>重试加载模板</button>{/if}{/if}
</div>
<style>pre{max-height:240px;overflow:auto;white-space:pre-wrap;font-size:11px}.story-template{padding-bottom:18px;margin-bottom:18px;border-bottom:1px solid var(--border)}summary{cursor:pointer;margin:14px 0;font-size:12px}.fields{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:12px}button{margin-top:12px}p{font-size:12px;color:#567797;margin:10px 0}p[role=alert]{color:#a63b4e;overflow-wrap:anywhere}textarea{min-height:70px}@media(max-width:720px){.fields{grid-template-columns:1fr 1fr}}</style>

<details><summary>故事服务配置</summary><label>设置凭据的通道<select bind:value={credentialProvider} disabled={busy || savingCredential}><option value="deepseek">DeepSeek · 故事生成</option><option value="aliyun_bailian">阿里云 · 故事审核</option></select></label><label>API 密钥<input type="password" autocomplete="new-password" bind:value={credentialSecret} disabled={busy || savingCredential} placeholder="输入要保存的密钥" /></label><button disabled={busy || savingCredential || !credentialSecret} on:click={saveCredential}>{savingCredential ? "保存中…" : "保存此通道密钥"}</button>{#if credentialMessage}<p role="status">{credentialMessage}</p>{/if}<button on:click={checkProviders}>检查原项目模型配置</button>{#if providerStatus}{#each providerStatus.routes as route}<p>{route.provider === "deepseek" ? "故事生成" : "故事审核"} · {route.model} · {route.configured ? "凭据已配置" : "缺少原项目凭据"}</p>{/each}<p>{providerStatus.busy ? "故事任务正在运行" : providerStatus.host_connected ? "本地故事服务已连接" : "生成时会自动启动本地故事服务"}</p>{/if}{#if providerError}<p role="alert">{providerError}</p>{/if}</details>
