<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  type Tab = "home" | "story" | "image" | "video";
  let tab: Tab = "home";
  let status = "就绪";
  let run: any = null;
  let storyText = "一个动漫女孩在黄昏的街道上转身离开，镜头保持中景。";
  let storyInput = '{\n  "title": "我的短剧",\n  "scenes": [{"summary": "人物转身离开", "source_spans": ["scene-1"]}]\n}';
  let imagePrompt = "动漫女孩，黄昏街道，中景，角色服装和场景保持一致，首帧与尾帧构图连续";
  let imageSize = "720*1280";
  let videoPrompt = "动漫女孩在黄昏街道转身离开，动作完整，保持角色身份、服装、镜头和光线连续";
  let videoSteps = 20;
  let videoTurbo = false;
  let stageOutput = "";
  let storyReady = false;
  let imageReady = false;
  async function generateStory() { try { const result = await invoke("run_media_stage", { stage: "story", input: JSON.stringify({ title: storyText, scenes: [{ summary: storyText, source_spans: ["scene-1"] }] }) }); storyInput = JSON.stringify(result, null, 2); stageOutput = storyInput; storyReady = true; status = "故事已生成"; tab = "home"; } catch (e) { status = `故事失败: ${e}`; } }
  async function generateImage() { if (!storyReady) { status = "请先生成故事"; tab = "story"; return; } try { const story = JSON.parse(storyInput); const result = await invoke("run_media_stage", { stage: "image", input: JSON.stringify({ scene: story.scenes?.[0], source_spans: story.scenes?.[0]?.source_spans ?? ["scene-1"] }) }); stageOutput = JSON.stringify(result, null, 2); imageReady = true; status = "图片已生成"; tab = "home"; } catch (e) { status = `图片失败: ${e}`; } }
  async function generateVideo() { if (!imageReady) { status = "请先生成首帧/尾帧图片"; tab = "image"; return; } try { const result = await invoke("run_media_stage", { stage: "video", input: JSON.stringify({ first_frame_ref: "artifact://首帧", last_frame_ref: "artifact://尾帧", source_spans: ["scene-1"], scene: { summary: videoPrompt }, prompt: videoPrompt }) }); stageOutput = JSON.stringify(result, null, 2); status = "视频已生成"; tab = "home"; } catch (e) { status = `视频失败: ${e}`; } }
  async function startPipeline(stage = "全流程") { try { JSON.parse(storyInput); } catch { status = "故事 JSON 格式错误"; tab = "story"; return; } status = `${stage}阶段已提交`; tab = "home"; run = await invoke("start_media_run", { storyInput }); poll(); }
  async function poll() { if (!run) return; run = await invoke("get_media_run", { runId: run.run_id }); if (!["succeeded", "failed"].includes(run.status)) setTimeout(poll, 700); }
  function stageState(index: number) { return run?.stages?.[index]?.state ?? "待开始"; }
</script>
<div class="shell"><aside class="sidebar"><div class="brand"><span class="brandmark">M</span><div><strong>Story Media</strong><small>ORCHESTRATOR</small></div></div><nav><button class:active={tab === "home"} on:click={() => tab = "home"}>⌂ 首页流水线</button><button class:active={tab === "story"} on:click={() => tab = "story"}>▣ 生成故事</button><button class:active={tab === "image"} on:click={() => tab = "image"}>◈ 生成图片</button><button class:active={tab === "video"} on:click={() => tab = "video"}>▶ 生成视频</button></nav><div class="sidebar-note"><strong>执行引擎</strong><small>Rust trusted runtime</small><small>模型无关 · 可控返工</small></div></aside><main>
  <header><div><h1>Story Media Orchestrator</h1><p>统一创作工作台 · 故事 → 图片 → 视频</p></div><span class="badge">{status}</span></header>
  <aside class="right-panel"><section><h3>Artifacts</h3>{#if run}{#each run.stages as stage}{#if stage.artifact}<div class="artifact-row"><span>{stage.name}</span><code>{stage.artifact}</code></div>{/if}{/each}{:else if stageOutput}<div class="artifact-row"><span>当前阶段输出</span><code>已生成</code></div>{:else}<p class="muted">暂无 artifact</p>{/if}</section><section><h3>质量门禁</h3><div class="gate-row"><span>故事对齐</span><b class="ok">待检查</b></div><div class="gate-row"><span>身份连续性</span><b class="ok">待检查</b></div><div class="gate-row"><span>动作完整性</span><b class="ok">待检查</b></div><div class="gate-row"><span>伪影检查</span><b class="ok">待检查</b></div></section><section><h3>运行日志</h3><p class="log">{status}</p>{#if run}<p class="log">run_id: {run.run_id}</p><p class="log">状态: {run.status}</p>{/if}</section></aside>
  <nav><button class:active={tab === "home"} on:click={() => tab = "home"}>首页流水线</button><button class:active={tab === "story"} on:click={() => tab = "story"}>生成故事</button><button class:active={tab === "image"} on:click={() => tab = "image"}>生成图片</button><button class:active={tab === "video"} on:click={() => tab = "video"}>生成视频</button></nav>
  {#if tab === "home"}
    <section class="card hero"><div><h2>自动化流水线</h2><p class="muted">从一个故事输入开始，按顺序生成故事包、首帧/尾帧图片和 5 秒视频。</p></div><button class="primary large" on:click={startPipeline}>开始全流程</button></section>
    <section class="flow"><div class:done={stageState(0) === "succeeded"} class:active={stageState(0) === "running"} class="flowstage" on:click={() => tab = "story"}><span>01</span><h3>生成故事</h3><strong>{stageState(0)}</strong><small>输入故事梗概，输出 story-package</small></div><i>→</i><div class:done={stageState(1) === "succeeded"} class:active={stageState(1) === "running"} class="flowstage" on:click={() => tab = "image"}><span>02</span><h3>生成图</h3><strong>{stageState(1)}</strong><small>首帧 / 尾帧与角色连续性</small></div><i>→</i><div class:done={stageState(2) === "succeeded"} class:active={stageState(2) === "running"} class="flowstage" on:click={() => tab = "video"}><span>03</span><h3>生成视频</h3><strong>{stageState(2)}</strong><small>MiniMax H3 · 5 秒动作单元</small></div></section>
    {#if run}<section class="card output"><h2>本次运行</h2><pre>{JSON.stringify(run, null, 2)}</pre></section>{:else if stageOutput}<section class="card output"><h2>阶段输出</h2><pre>{stageOutput}</pre></section>{:else}<section class="card mutedbox"><p>尚未开始运行。你可以先分别编辑三个阶段，也可以直接点击“开始全流程”。</p></section>{/if}
  {:else if tab === "story"}
    <section class="card workspace"><h2>生成故事</h2><p class="muted">输入故事梗概，生成可供后续生图和生视频使用的结构化故事包。</p><label>故事梗概<textarea bind:value={storyText}></textarea></label><div class="actions"><button class="primary" on:click={generateStory}>生成故事包</button><button on:click={() => tab = "home"}>返回流水线</button></div><label>story-package/v1 输出<textarea class="tall" bind:value={storyInput} spellcheck="false"></textarea></label></section>
  {:else if tab === "image"}
    <section class="card workspace"><h2>生成图片</h2><p class="muted">根据故事场景生成首帧/尾帧，后续视频阶段会引用 artifact。</p><label>图片提示词<textarea bind:value={imagePrompt}></textarea></label><label>图片尺寸<input bind:value={imageSize}></label><div class="actions"><button class="primary" disabled={!storyReady} on:click={generateImage}>生成首帧 / 尾帧</button><button on:click={() => tab = "home"}>返回流水线</button></div>{#if !storyReady}<p class="gate">🔒 请先完成“生成故事”</p>{/if}<div class="artifact empty">生成后将在这里显示图片 artifact、质量门禁和连续性检查。</div></section>
  {:else}
    <section class="card workspace"><h2>生成视频</h2><p class="muted">使用首帧/尾帧和动作约束生成单条 5 秒视频。</p><label>视频动作提示词<textarea bind:value={videoPrompt}></textarea></label><div class="row"><label>Steps<input type="number" min="1" max="100" bind:value={videoSteps}></label><label class="check"><input type="checkbox" bind:checked={videoTurbo}> Turbo 模式</label></div><div class="actions"><button class="primary" disabled={!imageReady} on:click={generateVideo}>生成 5 秒视频</button><button on:click={() => tab = "home"}>返回流水线</button></div>{#if !imageReady}<p class="gate">🔒 请先完成“生成故事” → “生成图”</p>{/if}<div class="artifact empty">生成后将在这里显示视频 artifact、任务轮询、质量检测和重试按钮。</div></section>
  {/if}
</main></div>
