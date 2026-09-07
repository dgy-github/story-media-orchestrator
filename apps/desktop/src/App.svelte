<script lang="ts">
  import VideoServiceSettings from "./VideoServiceSettings.svelte";
  import { onMount } from "svelte";
  import StoryTemplates from "./StoryTemplates.svelte";
  import PreviewProject from "./PreviewProject.svelte";
  import { invoke, convertFileSrc } from "@tauri-apps/api/core";
  type Tab = "home" | "story" | "image" | "video";
  let tab: Tab = "home";
  let storyVisited = false;
  $: if (tab === "story") storyVisited = true;
  let status = "就绪";
  let run: any = null;
  let storyText = "一个动漫女孩在黄昏的街道上转身离开，镜头保持中景。";
  let storyInput = '{\n  "title": "我的短剧",\n  "scenes": [{"summary": "人物转身离开", "source_spans": ["scene-1"]}]\n}';
  let storyScenes: any[] = [];
  let sceneIndex = 0;
  function selectScene() {
    imagePromptRevision = "";
    const scene = storyScenes[sceneIndex];
    imagePrompt = scene?.action || scene?.description || (Array.isArray(scene?.lines) ? scene.lines : []).map((line: any) => typeof line?.text === "string" ? line.text.trim() : "").filter(Boolean).join(" ") || scene?.summary || "";
    videoPrompt = imagePrompt;
    imageNegative = scene?.negative || ""; imageFraming = scene?.framing || ""; imageContinuity = scene?.continuity || "";
    imageReady = false; imageArtifact = null; candidateResults = [];
    videoUrl = ""; playbackError = ""; quality = null; stageOutput = "";
  }
  let imagePrompt = "动漫女孩，黄昏街道，中景，角色服装和场景保持一致，首帧与尾帧构图连续";
  let candidateCount = 1;
  let imageNegative = "";
  let imagePromptRevision = "";
  type PromptBatch = {batch: string; scene: string; time: string; revisions: {prompt: string; negative_prompt: string; source_spans: string[]}[]};
  let promptBatches: PromptBatch[] = [];
  let promptHistoryWarning = "";
  const promptHistoryKey = "story-media-prompt-history-v1";
  onMount(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(promptHistoryKey) || "[]");
      if (!Array.isArray(saved)) throw new Error("invalid history");
      promptBatches = saved.filter(item => typeof item?.batch === "string" && typeof item?.scene === "string" && typeof item?.time === "string" && Array.isArray(item?.revisions) && item.revisions.every((r: any) => typeof r?.prompt === "string" && typeof r?.negative_prompt === "string" && Array.isArray(r?.source_spans) && r.source_spans.every((span: any) => typeof span === "string"))).slice(0, 20);
    } catch { promptHistoryWarning = "历史提示词无法读取，本次生成仍可使用。"; }
  });
  function rememberPromptBatch(result: any, request: {input: string; scene: string}) {
    if (!result.prompt_history?.length) return;
    const batch = JSON.parse(request.input).batch_id;
    const item: PromptBatch = {batch, scene: request.scene, time: new Date().toLocaleString(), revisions: result.prompt_history.map((r: any) => ({prompt:r.prompt, negative_prompt:r.negative_prompt || "", source_spans:r.source_spans || []}))};
    promptBatches = [item, ...promptBatches.filter(item => item.batch !== batch)].slice(0, 20);
    try { localStorage.setItem(promptHistoryKey, JSON.stringify(promptBatches)); }
    catch { promptHistoryWarning = "本批提示词记录未能保存，当前仍可查看。"; }
  }

  function reusePromptRevision(prompt: string) {
    imagePromptRevision = prompt;
    const editor = document.querySelector<HTMLTextAreaElement>("#image-prompt-revision");
    const panel = editor?.closest("details");
    if (panel) panel.open = true;
    editor?.focus();
  }
  let imageFraming = "";
  let imageContinuity = "";
  let candidateResults: any[] = [];
  function selectCandidate(candidate: any) {
    if (mediaBusy || candidate.status !== "succeeded") return;
    imageArtifact = {...imageArtifact, first_frame_ref: candidate.artifact_ref, last_frame_ref: candidate.artifact_ref, selected_request_id: candidate.request_id};
    imageReady = true;
    videoUrl = ""; playbackError = ""; quality = null; stageOutput = "";
  }
  let imageSize = "720*1280";
  let videoPrompt = "动漫女孩在黄昏街道转身离开，动作完整，保持角色身份、服装、镜头和光线连续";
  let videoSteps = 20;
  let videoTurbo = false;
  let videoSeed = 0;
  let videoProvider = "comfyui";
  let wanModel = "wanx2.1-t2v-turbo";
  let wanSize = "1280*720";
  let wanDuration = 5;
  let wanNegative = "";
  let videoWidth = 864;
  let videoHeight = 480;
  let videoFrames = 124;
  let videoFps = 24;
  let videoUrl = "";
  let lastVideoInput = "";
  let lastVideoScene = "";
  let videoNeedsResume = false;
  let pendingVideos: {input: string; scene: string}[] = [];
  const pendingVideoKey = "story-media-pending-videos-v1";
  function readPendingVideos() {
    const saved = JSON.parse(localStorage.getItem(pendingVideoKey) || "[]");
    if (!Array.isArray(saved) || saved.some(item => typeof item?.input !== "string" || typeof item?.scene !== "string")) throw new Error("视频恢复记录格式错误");
    return saved;
  }
  onMount(() => {
    try {
      pendingVideos = readPendingVideos();
      if (pendingVideos.length) {
        lastVideoInput = pendingVideos[0].input; lastVideoScene = pendingVideos[0].scene;
        videoNeedsResume = true;
      }
    } catch { status = "视频恢复记录无法读取，请检查本地存储。"; }
  });
  function persistPendingVideo(complete = false) {
    const remaining = readPendingVideos().filter(item => item.input !== lastVideoInput);
    pendingVideos = complete ? remaining : [{input:lastVideoInput,scene:lastVideoScene}, ...remaining];
    localStorage.setItem(pendingVideoKey, JSON.stringify(pendingVideos));
  }
  function restoreVideo(item: {input: string; scene: string}) {
    lastVideoInput = item.input; lastVideoScene = item.scene;
    void generateVideo(true);
  }
  let playbackError = "";
  let stageOutput = "";
  let storyReady = false;
  let imageReady = false;
  let imageArtifact: any = null;
  let quality: any = null;
  function qualityLabel() { if (!quality) return "未检测"; if (quality.decision === "passed") return "通过"; if (quality.decision === "warning") return "警告"; if (quality.decision === "failed") return "失败"; return quality.reason || "不可用"; }
  let storyBusy = false;
  let mediaBusy = false;
  let templateDraft: any = null;
  function acceptStory(event: any) {
    const result = event.detail.package;
    storyInput = JSON.stringify(result, null, 2);
    stageOutput = storyInput; storyReady = true; quality = null;
    imageReady = false; imageArtifact = null; candidateResults = [];
    storyScenes = result.scenes || []; sceneIndex = 0; selectScene();
    stageOutput = storyInput;
    status = "故事已生成，可继续生成图片";
  }
  type PendingImage = {input: string; scene: string};
  let pendingImages: PendingImage[] = [];
  const pendingImageKey = "story-media-pending-images-v1";
  function readPendingImages(): PendingImage[] {
    const saved = JSON.parse(localStorage.getItem(pendingImageKey) || "[]");
    if (!Array.isArray(saved) || saved.some(item => {
      if (typeof item?.input !== "string" || typeof item?.scene !== "string") return true;
      try { const value = JSON.parse(item.input); return typeof value.batch_id !== "string" || !value.batch_id || !value.scene || !Array.isArray(value.source_spans); }
      catch { return true; }
    })) throw new Error("图片恢复记录格式错误");
    return saved;
  }
  onMount(() => {
    try { pendingImages = readPendingImages(); }
    catch { status = "图片恢复记录无法读取，请检查本地存储。"; }
  });
  async function generateImage(resume?: PendingImage) {
    if (mediaBusy || storyBusy) return;
    if (!resume && !storyReady) { status = "请先生成故事"; tab = "story"; return; }
    const story = JSON.parse(storyInput);
    const request = resume || {input: JSON.stringify({ batch_id: crypto.randomUUID(), scene: {...story.scenes?.[sceneIndex], action: imagePrompt, description: imagePrompt, ...(imageNegative.trim() ? {negative:imageNegative.trim()} : {}), ...(imageFraming.trim() ? {framing:imageFraming.trim()} : {}), ...(imageContinuity.trim() ? {continuity:imageContinuity.trim()} : {})}, source_spans: story.scenes?.[sceneIndex]?.source_spans?.length ? story.scenes[sceneIndex].source_spans : [`story-package/${story.scenes?.[sceneIndex]?.node_id || `scene-${sceneIndex + 1}`}`], image_size: imageSize, candidate_count: candidateCount, prompt_revision: imagePromptRevision.trim() || undefined }), scene: `场景 ${sceneIndex + 1}`};
    try {
      pendingImages = [request, ...readPendingImages().filter(item => item.input !== request.input)];
      localStorage.setItem(pendingImageKey, JSON.stringify(pendingImages));
    } catch { status = "无法保存图片恢复记录，尚未提交任务。"; return; }
    mediaBusy = true; candidateResults = []; imageReady = false; imageArtifact = null; quality = null; stageOutput = "";
    videoUrl = ""; playbackError = "";
    status = resume ? "正在恢复候选图…" : "正在生成图片…";
    try {
      const result: any = await invoke("run_media_stage", {stage: "image", input: request.input});
      quality = result.quality_evaluation; imageArtifact = result; stageOutput = JSON.stringify(result, null, 2);
      candidateResults = result.candidate_results || []; imageReady = !!result.first_frame_ref;
      rememberPromptBatch(result, request);
      if (imageReady) {
        const source = JSON.parse(request.input).scene;
        videoPrompt = source.action || source.description || source.summary || "";
      }
      status = result.status === "failed" ? "候选图均生成失败，可恢复原批次" : result.status === "partial" ? "部分候选图失败，可选用已完成图片或恢复原批次" : "图片已生成";
      if (result.status === "succeeded") {
        pendingImages = readPendingImages().filter(item => item.input !== request.input);
        localStorage.setItem(pendingImageKey, JSON.stringify(pendingImages));
      }
    } catch (e) { status = `图片失败: ${e}`; }
    finally { mediaBusy = false; }
  }
  async function generateVideo(resume = false) {
    if (mediaBusy || storyBusy) return;
    if (!resume && videoProvider === "dashscope" && ["wanx2.1-t2v-turbo", "wan2.1-t2v-turbo"].includes(wanModel) && wanDuration !== 5) {
      status = "当前 Wan 模型仅支持 5 秒，请调整时长后生成。";
      return;
    }
    if (resume && !lastVideoInput) return;
    if (!resume && !imageReady) { status = "请先生成图片"; tab = "image"; return; }
    if (!resume) {
      if (!Number.isSafeInteger(videoSeed) || videoSeed < 0) { status = "种子必须是非负整数"; return; }
      lastVideoInput = JSON.stringify({video_provider: videoProvider, ...(videoProvider === "dashscope" ? {batch_id: crypto.randomUUID(), video_model: wanModel, video_size: wanSize, duration_seconds: wanDuration, negative_prompt: wanNegative} : {}), first_frame_ref: imageArtifact.first_frame_ref, last_frame_ref: imageArtifact.last_frame_ref,
        source_spans: imageArtifact.source_spans, scene: {summary: videoPrompt}, prompt: videoPrompt, turbo: videoTurbo, seed: videoSeed, width: videoWidth, height: videoHeight, length: videoFrames, fps: videoFps});
      lastVideoScene = `场景 ${sceneIndex + 1}`;
    }
    try { persistPendingVideo(); }
    catch { status = "无法保存视频恢复记录，尚未提交任务。"; return; }
    mediaBusy = true; videoUrl = ""; playbackError = ""; quality = null;
    status = resume ? "正在恢复上次视频任务…" : "正在生成视频…";
    videoNeedsResume = true;
    try {
      const result: any = await invoke("run_media_stage", {stage: "video", input: lastVideoInput});
      quality = result.quality_evaluation;
      videoUrl = result.output_file ? convertFileSrc(result.output_file) : "";
      playbackError = result.playback_error || "";
      stageOutput = JSON.stringify(result, null, 2);
      videoNeedsResume = result.status !== "succeeded";
      if (!videoNeedsResume) {
        try { persistPendingVideo(true); }
        catch { playbackError = "视频已完成，但恢复记录未能清除；再次恢复会读取缓存。"; }
      }
      status = result.status === "succeeded" ? `${lastVideoScene}视频已生成` : `视频任务状态：${result.status}`;
    } catch (e) { status = `视频未完成: ${e}`; }
    finally { mediaBusy = false; }
  }
  async function startPipeline(stage = "全流程") { try { JSON.parse(storyInput); } catch { status = "故事 JSON 格式错误"; tab = "story"; return; } status = `${stage}阶段已提交`; tab = "home"; run = await invoke("start_media_run", { storyInput }); poll(); }
  async function poll() { if (!run) return; run = await invoke("get_media_run", { runId: run.run_id }); if (!["succeeded", "failed", "planned"].includes(run.status)) setTimeout(poll, 700); }
  function stageState(index: number) { return run?.stages?.[index]?.state ?? "待开始"; }
</script>
<div class="shell"><aside class="sidebar"><div class="brand"><span class="brandmark">M</span><div><strong>Story Media</strong><small>ORCHESTRATOR</small></div></div><nav><button class:active={tab === "home"} on:click={() => tab = "home"}>⌂ 创作工作台</button><details class="nav-group"><summary>创作工具</summary><div><button class:active={tab === "story"} on:click={() => tab = "story"}>▣ 生成故事</button><button class:active={tab === "image"} on:click={() => tab = "image"}>◈ 生成图片</button><button class:active={tab === "video"} on:click={() => tab = "video"}>▶ 生成视频</button></div></details></nav><div class="sidebar-note"><strong>Story Media</strong><small>让故事成为画面</small><small>本地项目 · 随时恢复</small></div></aside><main>
  <header class="app-header"><div><p class="eyebrow">工作空间 / {tab === "home" ? "创作台" : tab === "story" ? "故事" : tab === "image" ? "图片" : "视频"}</p><h1>{tab === "home" ? "流水线作业台" : tab === "story" ? "故事编辑" : tab === "image" ? "画面生成" : "视频生成"}</h1></div><span class="workspace-indicator"><i></i>{tab === "home" ? "本地工作空间" : status}</span></header>


  <div hidden={tab !== "home"}><PreviewProject /></div>
  {#if storyVisited}<div hidden={tab !== "story"}>
    <section class="card workspace"><h2>生成故事</h2><p class="muted">输入故事梗概，生成可供后续生图和生视频使用的结构化故事包。</p><label>故事梗概<textarea disabled={storyBusy} bind:value={storyText}></textarea></label><StoryTemplates disabled={mediaBusy} premise={storyText} draft={templateDraft} on:draft={(e) => templateDraft = e.detail} on:busy={(e) => storyBusy = e.detail} on:generated={acceptStory} /><div class="actions"><button on:click={() => tab = "home"}>返回流水线</button></div><label>story-package/v1 输出<textarea class="tall" readonly value={storyInput} spellcheck="false"></textarea></label></section>
  </div>{/if}
  {#if tab === "image"}
    <section class="card workspace"><h2>生成图片</h2>{#if storyScenes.length}<label>故事场景<select aria-label="故事场景" bind:value={sceneIndex} on:change={selectScene} disabled={mediaBusy || storyBusy}>{#each storyScenes as scene, index}<option value={index}>场景 {index + 1} · {scene.location || scene.summary || scene.node_id || "未命名场景"}</option>{/each}</select></label>{/if}<p class="muted">根据故事场景生成候选图片，选用后作为后续视频的关联素材。当前视频通道尚未接入首尾帧控制。</p><label>图片提示词<textarea disabled={mediaBusy} bind:value={imagePrompt}></textarea></label><label>图片尺寸<input disabled={mediaBusy} bind:value={imageSize}></label><details><summary>高级生图参数</summary><label>完整提示词修订<textarea id="image-prompt-revision" bind:value={imagePromptRevision} disabled={mediaBusy} placeholder="可选：填写后替换原工作流组合的正面提示词"></textarea></label><p>留空沿用原场景组合；修订保留故事来源与负面约束，并记录父版本。再次生成会创建新批次。</p><label>负面提示词<textarea bind:value={imageNegative} disabled={mediaBusy} placeholder="留空沿用原工作流的负面约束"></textarea></label><label>镜头构图<input bind:value={imageFraming} disabled={mediaBusy} placeholder="例如：远景、人物位于画面右侧"></label><label>连续性约束<input bind:value={imageContinuity} disabled={mediaBusy} placeholder="例如：保持蓝色外套与棕色邮包"></label><label>候选数量<select disabled={mediaBusy} bind:value={candidateCount}>{#each [1,2,3,4,5,6,7,8] as count}<option value={count}>{count} 张</option>{/each}</select></label><p>每张候选单独生成，按实际请求计费。</p></details><div class="actions"><button class="primary" disabled={!storyReady || mediaBusy || storyBusy} on:click={() => generateImage()}>生成候选图片</button><button on:click={() => tab = "home"}>返回流水线</button></div>{#if !storyReady}<p class="gate">🔒 请先完成“生成故事”</p>{/if}{#if pendingImages.length}<details open><summary>待恢复的候选图批次（{pendingImages.length}）</summary><p>恢复沿用提交时的参数，已完成图片直接复用；点击生成会新建批次并计费。</p>{#each pendingImages as item}<div class="actions"><span>{item.scene} · 批次 {JSON.parse(item.input).batch_id.slice(0, 8)}</span><button disabled={mediaBusy || storyBusy} on:click={() => generateImage(item)}>恢复这批候选图</button></div>{/each}</details>{/if}{#if promptHistoryWarning}<p role="status">{promptHistoryWarning}</p>{/if}{#if promptBatches.length}<details class="saved-prompt-history"><summary>最近提示词批次（{promptBatches.length}）</summary>{#each promptBatches as batch}<details><summary>{batch.scene} · {batch.time} · {batch.batch.slice(0, 8)}</summary>{#each batch.revisions as revision, index}<article><h3>{index === 0 ? "原始组合" : `修订 ${index}`}</h3><p style="white-space:pre-wrap;overflow-wrap:anywhere">{revision.prompt}</p><p>负面约束：{revision.negative_prompt || "无"}</p><small>原来源：{revision.source_spans.join(" · ")}</small><div class="actions"><button disabled={mediaBusy} on:click={() => reusePromptRevision(revision.prompt)}>用作当前场景修订</button></div></article>{/each}</details>{/each}<p>保留最近 20 批文字记录；复用正面提示词时，负面约束与故事来源仍取当前场景。</p></details>{/if}{#if imageArtifact?.prompt_history?.length}<details class="prompt-history"><summary>本批提示词版本（{imageArtifact.prompt_history.length}）</summary>{#each imageArtifact.prompt_history as revision, index}<article><h3>{index === 0 ? "原始组合" : `修订 ${index}`}</h3><p style="white-space:pre-wrap;overflow-wrap:anywhere">{revision.prompt}</p>{#if revision.negative_prompt}<p>负面约束：{revision.negative_prompt}</p>{/if}<small>来源：{revision.source_spans.join(" · ")}</small><div class="actions"><button disabled={mediaBusy} on:click={() => reusePromptRevision(revision.prompt)}>填入修订框</button></div></article>{/each}<p>填入仅修改草稿，点击“生成候选图片”才会创建新批次。</p></details>{/if}{#if candidateResults.length}<div class="candidate-gallery">{#each candidateResults as candidate, index}<button class:selected={imageArtifact?.selected_request_id === candidate.request_id} disabled={mediaBusy || candidate.status !== "succeeded"} on:click={() => selectCandidate(candidate)}>{#if candidate.preview_url}<img src={candidate.preview_url} alt={`候选图片 ${index + 1}`} />{/if}<span>候选 {index + 1} · {candidate.status === "failed" ? "生成失败" : imageArtifact?.selected_request_id === candidate.request_id ? "已选用" : "点击选用"}</span>{#if candidate.error}<small>{candidate.error_detail || candidate.error}</small>{/if}</button>{/each}</div><p>选用图片仅决定后续素材，质量审核仍待完成。</p>{:else}<div class="artifact empty">生成后在这里查看并选择候选图片。</div>{/if}</section>
  {:else if tab === "video"}
    <section class="card workspace"><h2>生成视频</h2><VideoServiceSettings disabled={mediaBusy} />{#if storyScenes.length}<p>当前素材：场景 {sceneIndex + 1} · {storyScenes[sceneIndex]?.location || storyScenes[sceneIndex]?.node_id || "故事场景"}</p>{/if}<label>视频通道<select bind:value={videoProvider} disabled={mediaBusy}><option value="comfyui">本地 ComfyUI · MiniMax H3</option><option value="dashscope">阿里云 · Wan</option></select></label><p class="muted">当前两条通道均使用文字生成视频，所选图片用于素材记录，尚未接入画面条件。</p><label>视频动作提示词<textarea disabled={mediaBusy} bind:value={videoPrompt}></textarea></label><details><summary>高级视频参数</summary>{#if videoProvider === "dashscope"}<label>视频模型<input bind:value={wanModel} disabled={mediaBusy} /></label><label>视频尺寸<input bind:value={wanSize} disabled={mediaBusy} /></label><label>时长（秒）<input type="number" min={["wanx2.1-t2v-turbo", "wan2.1-t2v-turbo"].includes(wanModel) ? 5 : 1} max={["wanx2.1-t2v-turbo", "wan2.1-t2v-turbo"].includes(wanModel) ? 5 : 10} step="1" bind:value={wanDuration} disabled={mediaBusy} /></label><label>负面提示词<textarea bind:value={wanNegative} disabled={mediaBusy}></textarea></label><p>{["wanx2.1-t2v-turbo", "wan2.1-t2v-turbo"].includes(wanModel) ? "当前模型仅支持 5 秒。" : ""}具体支持的尺寸和时长取决于所选模型；每次生成会创建新批次并按阿里云账户计费；恢复按钮复用原任务。</p>{:else}<label>宽度<input type="number" min="64" step="8" bind:value={videoWidth} disabled={mediaBusy} /></label><label>高度<input type="number" min="64" step="8" bind:value={videoHeight} disabled={mediaBusy} /></label><label>总帧数<input type="number" min="1" step="1" bind:value={videoFrames} disabled={mediaBusy} /></label><label>帧率<input type="number" min="1" step="1" bind:value={videoFps} disabled={mediaBusy} /></label><p>时长由总帧数 ÷ 帧率决定；更大的画面和更多帧会增加显存需求。</p><label>生成种子<input type="number" min="0" max="9007199254740991" step="1" bind:value={videoSeed} disabled={mediaBusy} /></label><p>相同参数复用已有任务；更换种子生成新版本。恢复始终沿用原种子。</p><div class="row"><p>采样步数：{videoTurbo ? 8 : 20}（由原工作流决定）</p><label class="check"><input type="checkbox" disabled={mediaBusy} bind:checked={videoTurbo}> Turbo 模式</label></div>{/if}</details><div class="actions"><button class="primary" disabled={!imageReady || mediaBusy || storyBusy} on:click={() => generateVideo()}>生成视频（约 {videoProvider === "dashscope" ? wanDuration : videoFps > 0 ? (videoFrames / videoFps).toFixed(1) : "—"} 秒）</button>{#if videoNeedsResume}<button disabled={mediaBusy || storyBusy} on:click={() => generateVideo(true)}>恢复上次视频任务（{lastVideoScene}）</button>{/if}<button on:click={() => tab = "home"}>返回流水线</button></div>{#if !imageReady}<p class="gate">🔒 请先完成“生成故事” → “生成图”</p>{/if}{#if pendingVideos.length}<details><summary>待恢复视频（{pendingVideos.length}）</summary>{#each pendingVideos as item}<p><button disabled={mediaBusy || storyBusy} on:click={() => restoreVideo(item)}>恢复 · {item.scene}</button></p>{/each}</details>{/if}<div class="video-result">{#if videoUrl}<video controls preload="metadata" src={videoUrl} on:error={() => playbackError = "视频无法播放，请检查格式或本机解码支持。"}><track kind="captions" /></video>{:else}<p>生成完成后在这里播放视频。</p>{/if}{#if playbackError}<p role="alert">{playbackError}</p>{/if}</div></section>
  {/if}
{#if tab !== "home"}<details class="diagnostics"><summary>运行诊断 <span>产物、质量检查与日志</span></summary><div class="diagnostics-grid"><section><h3>Artifacts</h3>{#if run}{#each run.stages as stage}{#if stage.artifact}<div class="artifact-row"><span>{stage.name}</span><code>{stage.artifact}</code></div>{/if}{/each}{:else if stageOutput}<div class="artifact-row"><span>当前阶段输出</span><code>已生成</code></div>{:else}<p class="muted">暂无 artifact</p>{/if}</section><section><h3>质量门禁</h3><div class="gate-row"><span>质量模型决策</span><b class:pass={quality?.decision === "passed"} class:warn={quality?.decision === "warning"} class:fail={quality?.decision === "failed"} class="ok">{qualityLabel()}</b></div><div class="gate-row"><span>检测原因</span><b class="ok">{quality?.reason ?? "等待 artifact"}</b></div>{#if quality?.failures}<div class="gate-row"><span>失败指标</span><b class="fail">{quality.failures.length}</b></div>{/if}</section><section><h3>运行日志</h3><p class="log">{status}</p>{#if run}<p class="log">run_id: {run.run_id}</p><p class="log">状态: {run.status}</p>{/if}</section></div></details>
{/if}
</main></div>

<style>
.video-result video{width:100%;max-height:480px;background:#101820;border-radius:12px;margin-top:20px}.candidate-gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:20px}.candidate-gallery button{display:flex;flex-direction:column;gap:10px;padding:10px}.candidate-gallery button.selected{border:2px solid #3979c6;background:#eaf3ff}.candidate-gallery img{width:100%;height:220px;object-fit:contain;border-radius:8px}.candidate-gallery small{overflow-wrap:anywhere}
</style>
