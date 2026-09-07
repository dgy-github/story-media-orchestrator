<script lang="ts">
  import VideoServiceSettings from "./VideoServiceSettings.svelte";
  import StoryTemplates from "./StoryTemplates.svelte";
  import { tick, onMount, onDestroy } from "svelte";
  import { projectCaptions } from "./captions";
  import { invoke, convertFileSrc } from "@tauri-apps/api/core";
  let project = "";
  let projectMenu = false;
  let choosing = false;
  let generatingStory = false;
  let recentProjects: {path: string; ready: boolean}[] = [];
  const projectName = (path: string) => path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || path;
  const pathKey = (path: string) => path.replace(/^\\\\\?\\/, "").replace(/\\/g, "/").replace(/\/$/, "").toLowerCase();
  function rememberProject(ready: boolean) {
    recentProjects = [{path: project, ready}, ...recentProjects.filter(p => pathKey(p.path) !== pathKey(project))].slice(0, 12);
    try { localStorage.setItem("story-media-projects", JSON.stringify(recentProjects)); localStorage.setItem("story-media-current-project", project); } catch { /* Session selection still works without storage. */ }
  }
  async function selectRecent(item: {path: string; ready: boolean}) {
    if (busy || choosing) return;
    project = item.path; manifest = null; videoUrl = ""; error = ""; playback = "";
    rememberProject(item.ready);
    projectMenu = false;
    if (item.ready) await action("load");
    else status = "已选择目录，可创建项目或打开已有项目。";
  }
  onMount(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("story-media-projects") || "[]");
      recentProjects = Array.isArray(saved) ? saved.filter(p => p && typeof p.path === "string" && typeof p.ready === "boolean").slice(0,12) : [];
      const last = localStorage.getItem("story-media-current-project");
      const item = recentProjects.find(p => p.path === last);
      if (item) void selectRecent(item);
    } catch { recentProjects = []; }
  });
  async function chooseProject() {
    choosing = true;
    try {
      const directory = await invoke<string | null>("choose_project_directory");
      if (directory) {
        project = directory; manifest = null; videoUrl = ""; error = ""; playback = "";
        rememberProject(false);
        projectMenu = false;
        status = "目录已选择，可以打开已有项目或创建项目。";
      }
    } catch (e) { error = String(e); status = "目录选择失败，请重试。"; }
    finally { choosing = false; }
  }
  const exampleStory = "清晨，青石城的钟楼停摆，穿蓝色短斗篷、背棕色邮包的年轻信使阿澄来到城门前，手中攥着一封必须在日出前送达钟楼的信。阿澄穿过空荡的集市，发现通往钟楼的石桥被洪水冲断，只能沿河绕向旧水车。旧水车旁，一位白发修钟匠被倒下的木梁困住，阿澄停下脚步，放下邮包，用木杆撬起横梁救出老人。老人看见信封上的齿轮徽记，取出随身铜钥匙，指向通往钟楼的地下检修门。第一缕阳光照上屋顶，阿澄和老人冲上钟楼，信封里掉出一枚失落的齿轮，老人把它装回机芯，阿澄拉动沉重的启动杆。钟声越过青石城，店铺纷纷开门，阿澄站在钟楼窗边望着苏醒的街道，老人将铜钥匙交到他手中，二人相视而笑。";
  let story = exampleStory;
  let storyPackage: any = null;
  let storyJob: any = null;
  let storyDraft: any = null;
  function useCompleteExample() {
    if (busy) return;
    storyDraft = null; storyJob = null; storyPackage = null; story = exampleStory; manifest = null; project = ""; videoUrl = ""; error = ""; playback = "";
    status = "完整示例已载入，请为新版本选择另一个项目文件夹。";
  }
  let selectedShot = "shot-01";
  let manifest: any = null;
  let busy = false;
  let status = "选择项目文件夹后，即可开始制作。";
  let error = "";
  let videoUrl = "";
  let captionsUrl = "";
  $: if (!videoUrl && captionsUrl) { URL.revokeObjectURL(captionsUrl); captionsUrl = ""; }
  onDestroy(() => { if (captionsUrl) URL.revokeObjectURL(captionsUrl); });
  let revision = 0;
  let playback = "";
  let imageSource = "dashscope";
  let outputMode = "preview";
  let ttsSource = "silent";
  let candidateCount = 2;
  let shotMotion = "none";
  let shotTransition = "cut";
  let autoRun = false;
  async function approveProject() {
    const previous = selectedShot;
    selectedShot = "all";
    await action("review");
    selectedShot = previous;
  }
  let projectVideoProvider = "comfyui";
  const defaultVideoOptions = {model: "wanx2.1-t2v-turbo", size: "1280*720", duration_seconds: 5, resolution: "720P", width: 864, height: 480, length: 124, fps: 24, seed: 0, turbo: false};
  let videoOptions = {...defaultVideoOptions};
  let imageModel = "";
  let imageSize = "";
  let selectedStage = -1;
  let previewOpen = true;
  let logsOpen = false;
  let logs: string[] = [];
  const stages = ["故事", "分镜", "生图", "合成", "成片"];
  $: shots = manifest?.shots || [];
  $: currentEffectShot = shots.find((s: any) => s.id === selectedShot);
  $: if (currentEffectShot) {
    shotMotion = currentEffectShot.motion || "none";
    shotTransition = currentEffectShot.transition || "cut";
  }
  $: imageCount = shots.filter((s: any) => !!s.assets?.image).length;
  $: fallbackShots = manifest?.mode === "render" ? shots.filter((s: any) => s.mode === "preview" && s.error?.startsWith("video fallback:")) : [];
  $: states = [manifest ? "已完成" : "待输入", shots.length ? "已完成" : "等待故事", shots.some((s: any) => s.status === "failed") ? "失败" : shots.length && imageCount === shots.length ? "已完成" : busy && !!manifest ? "处理中" : "待开始", fallbackShots.length ? "视频未完成" : manifest?.output ? "已完成" : manifest?.error && imageCount === shots.length && shots.length ? "失败" : busy && imageCount === shots.length && shots.length ? "处理中" : "待开始", videoUrl ? "可播放" : "待输出"];
  async function start() {
    if (outputMode === "render" && projectVideoProvider === "dashscope" && ["wanx2.1-t2v-turbo", "wan2.1-t2v-turbo"].includes(videoOptions.model) && videoOptions.duration_seconds !== 5) {
      selectedStage = 3;
      status = "当前 Wan 模型仅支持 5 秒，请调整时长后开始。";
      return;
    }
    if (!project) {
      projectMenu = true;
      status = "请先选择项目文件夹，再开始作业。";
      await tick();
      document.querySelector(".project-picker")?.scrollIntoView({behavior: "smooth", block: "center"});
      (document.querySelector(".directory-choice") as HTMLButtonElement)?.focus({preventScroll: true});
      return;
    }
    if (!story.trim()) {
      selectedStage = 0;
      status = "请先输入故事内容。";
      await tick();
      document.querySelector<HTMLTextAreaElement>(".node-detail textarea")?.focus();
      return;
    }
    if (!manifest) await action("create");
    if (manifest && !error) await action("run");
  }


  async function action(command: string, editOptions: any = null) {
    if (busy) return;
    busy = true;
    error = "";
    status = `${command} 正在处理…`;
    logs = [...logs, `${new Date().toLocaleTimeString()} · ${status}`];
    videoUrl = "";
    playback = "";
    let reading = false;
    let active = true;
    const timer = ["run", "resume", "retry"].includes(command) ? setInterval(async () => {
      if (reading) return;
      reading = true;
      try {
        const snapshot: any = await invoke("project_workflow", {command: "load", project});
        if (active) manifest = snapshot.manifest;
      } catch { /* The final command response reports actionable errors. */ }
      finally { reading = false; }
    }, 1200) : undefined;
    try {
      const result: any = await invoke("project_workflow", {
        command, project, storyJob: command === "create" ? storyJob : null, storyPackage: command === "create" ? storyPackage : null, story: command === "create" ? story : null,
        shot: ["retry", "review", "candidates", "select", "effects"].includes(command) ? selectedShot : null,
        editOptions,
        mode: command === "run" ? outputMode : null,
        ttsSource: command === "run" ? ttsSource : null,
        autoRun,
        videoProvider: command === "run" ? projectVideoProvider : null,
        videoOptions: command === "run" && outputMode === "render" ? videoOptions : null,
        imageSource: ["run", "resume", "retry"].includes(command) ? imageSource : null,
        imageModel: imageSource === "dashscope" ? imageModel || null : null,
        imageSize: imageSource === "dashscope" ? imageSize || null : null,
      });
      manifest = result.manifest;
      project = result.project || project;
      rememberProject(true);
      projectMenu = false;
      error = manifest.error || result.error || "";
      story = manifest.story;
      storyPackage = manifest.story_package || null;
      storyJob = manifest.story_job || null;
      storyDraft = null;
      if (command !== "create") {
        outputMode = manifest.mode || "preview";
        ttsSource = manifest.tts_source || "silent";
        projectVideoProvider = manifest.video_provider || "comfyui";
        videoOptions = {...defaultVideoOptions, ...(manifest.video_options || {})};
        imageSource = manifest.image_source || "local";
        imageModel = manifest.image_model || "";
        imageSize = manifest.image_size || "";
      }
      if (!manifest.shots.some((s: any) => s.id === selectedShot)) {
        selectedShot = manifest.shots[0]?.id || "";
      }
      if (result.output && !error) {
        revision += 1;
        videoUrl = `${convertFileSrc(result.output)}?v=${Date.now()}-${revision}`;
        if (captionsUrl) URL.revokeObjectURL(captionsUrl);
        captionsUrl = URL.createObjectURL(new Blob([projectCaptions(manifest.shots)], {type: "text/vtt"}));
      }
      const fallbackCount = manifest.mode === "render" ? manifest.shots.filter((s: any) => s.mode === "preview" && s.error?.startsWith("video fallback:")).length : 0;
      status = error ? "处理失败；修复原因后点击恢复，或重试失败镜头。" : fallbackCount ? `${fallbackCount} 个镜头的视频生成失败，当前成片使用图片回退；点击恢复可继续生成视频。` : `项目状态：${manifest.status}`;
    } catch (e) {
      error = String(e);
      status = "操作失败";
    } finally {
      active = false;
      clearInterval(timer);
      busy = false;
      logs = [...logs, `${new Date().toLocaleTimeString()} · ${status}${error ? " · " + error : ""}`];
    }
  }
</script>

<div class="project-toolbar">
  <details class="project-picker" bind:open={projectMenu}>
    <summary><span class="folder-symbol">▣</span><span><small>当前项目</small><strong>{project ? projectName(project) : "选择或创建项目"}</strong>{#if project}<span class="current-path" title={project}>{project}</span>{/if}</span><span class="picker-chevron">⌄</span></summary>
    <div class="project-menu">
      <p>最近项目</p>
      <div class="recent-projects">{#each recentProjects as item}<button class="recent-project" class:current={pathKey(item.path) === pathKey(project)} disabled={busy || choosing} on:click={() => selectRecent(item)}><strong>{projectName(item.path)}</strong><span>{item.path}</span><small>{pathKey(item.path) === pathKey(project) ? "当前选择 · " : ""}{item.ready ? "已创建项目" : "待打开 / 创建"}</small></button>{:else}<p>暂无最近项目，请先选择文件夹。</p>{/each}</div>
      <button class="directory-choice" title={project || "选择项目文件夹"} disabled={busy || choosing} on:click={chooseProject}><span class="directory-name">{choosing ? "正在选择…" : project ? `▣ ${projectName(project)}` : "▣ 选择项目文件夹"}</span><span class="directory-change">{project ? "更换…" : "浏览…"}</span></button>
      <p class="selected-directory">{project || "选择已有项目文件夹，或为新项目选择空文件夹。"}</p>
      <div class="project-actions"><button disabled={busy || !project} on:click={() => action("load")}>打开项目</button><button class="primary" disabled={busy || !project || !story.trim()} on:click={() => action("create")}>＋ 创建项目</button></div>
    </div>
  </details>
  <span class="project-state" role="status" class:working={busy}>{busy ? "处理中…" : manifest ? "项目已载入" : project ? "目录已选择 · 待打开或创建" : "尚未选择项目"}</span>
</div>

<div class="pipeline-heading"><div><p class="eyebrow">把灵感，一格格变成动画</p><h2>故事制作流水线</h2></div><button on:click={() => previewOpen = !previewOpen} aria-expanded={previewOpen}>{previewOpen ? "收起预览 ›" : "展开预览 ‹"}</button></div>
<div class="project-actions">
  <label>配音<select bind:value={ttsSource} disabled={busy}><option value="silent">静音</option><option value="windows">Windows 中文 / 英文配音</option></select></label>
  <label><input type="checkbox" bind:checked={autoRun} disabled={busy}>自动执行，跳过人工审核</label>
  {#if manifest?.status === "awaiting_storyboard_review" || manifest?.status === "awaiting_asset_review"}
    <span role="status">{manifest.status === "awaiting_storyboard_review" ? "请检查分镜后批准" : "请检查项目 frames、audio、video 目录中的资产后批准"}</span>
    <button disabled={busy} on:click={approveProject}>批准当前阶段</button>
    <button disabled={busy} on:click={() => action("resume")}>继续执行</button>
  {/if}
</div>
  {#if manifest && shots.length}
  <details class="node-detail"><summary>镜头候选、首尾帧与运镜</summary>
    <label>镜头<select bind:value={selectedShot} disabled={busy}>{#each shots as s}<option value={s.id}>{s.id}</option>{/each}</select></label>
    <label>候选数量<input type="number" min="1" max="8" bind:value={candidateCount} disabled={busy}></label>
    <button disabled={busy || manifest.image_source !== "dashscope"} on:click={() => action("candidates", {candidate_count:candidateCount})}>生成候选图（按数量计费）</button>
    {#each (shots.find((s: any) => s.id === selectedShot)?.candidates || []) as candidate}
      <figure><img src={convertFileSrc(candidate.path)} alt="镜头候选图" style="width:180px;max-width:100%;height:120px;object-fit:contain" />
        <button disabled={busy || candidate.quality?.decision === "failed"} on:click={() => action("select", {candidate_id:candidate.id,frame_role:"image"})}>选作首帧</button>
        <button disabled={busy || candidate.quality?.decision === "failed"} on:click={() => action("select", {candidate_id:candidate.id,frame_role:"last_frame"})}>选作尾帧</button>
        <figcaption>质量：{candidate.quality?.decision || "unavailable"} · {candidate.quality?.reason || "需人工审核"}</figcaption>
      </figure>
    {/each}
    <label>图片运镜<select bind:value={shotMotion}><option value="none">静止</option><option value="push_in">缓慢推进</option><option value="pan">水平平移</option></select></label>
    <label>转场<select bind:value={shotTransition}><option value="cut">直接切换</option><option value="fade">淡入淡出</option></select></label>
    <button disabled={busy} on:click={() => action("effects", {motion:shotMotion,transition:shotTransition})}>保存当前镜头效果</button>
    <button disabled={busy} on:click={() => { outputMode="render"; projectVideoProvider="dashscope"; videoOptions={...videoOptions,model:"wan2.2-kf2v-flash",duration_seconds:5}; selectedStage=3; }}>使用 Wan 首尾帧视频（每镜 5 秒）</button>
    <p>每个镜头需先选择首帧与尾帧；选择会使原视频和资产批准失效。未配置质量模型时需人工审图。</p>
  </details>
  {/if}
<div class="pipeline-layout" class:without-preview={!previewOpen}>
  <section class="pipeline-canvas">
    <div class="pipeline-nodes" aria-label="制作阶段">
      {#each stages as stage, i}
        <button class="pipeline-node" class:selected={selectedStage === i} class:complete={states[i] === "已完成" || states[i] === "可播放"} class:failed={states[i] === "失败"} disabled={generatingStory} aria-expanded={selectedStage === i} on:click={() => selectedStage = selectedStage === i ? -1 : i}>
          <span class="node-number">0{i + 1}</span><strong>{stage}</strong><span class="node-state">{states[i]}</span>
          <span class="node-progress">{i === 2 ? `${imageCount} / ${shots.length} 张` : i === 1 ? `${shots.length} 个镜头` : states[i] === "已完成" || states[i] === "可播放" ? "100%" : "—"}</span>
          {#if states[i] === "失败"}<small>点击查看错误并恢复</small>{/if}
        </button>
      {/each}
    </div>
    {#if selectedStage < 0}<div class="canvas-hint"><div class="storyboard-art" aria-hidden="true"><i></i><i></i><i></i><b>✦</b></div><h3>下一幕，从你的故事开始</h3><p>点击阶段节点查看配置与产物，准备好后开始作业。</p></div>
    {:else}<section class="node-detail"><div class="detail-heading"><h3>0{selectedStage + 1} / {stages[selectedStage]}</h3><button disabled={generatingStory} on:click={() => selectedStage = -1}>收起</button></div>
      {#if selectedStage === 0}<StoryTemplates draft={storyDraft} on:draft={(e) => storyDraft = e.detail} savedJob={storyJob} disabled={busy || !!manifest} premise={story} on:busy={(e) => { generatingStory = e.detail; busy = e.detail; }} on:generated={(e) => { story = e.detail.text; storyPackage = e.detail.package; storyJob = e.detail.job; status = "模板故事已生成，可创建项目继续制作。"; }} /><label>故事内容<textarea on:input={() => { storyPackage = null; storyJob = null; }} bind:value={story} disabled={busy || !!manifest}></textarea></label><p>模板生成保留原场景结构；手动编辑正文后按句拆分镜头；建议每句描述一个明确动作，覆盖起因、阻碍、转折和结局。</p>{#if storyPackage}<details><summary>原始故事包（角色、对白与场景）</summary><pre>{JSON.stringify(storyPackage, null, 2)}</pre></details>{/if}<button disabled={busy} on:click={useCompleteExample}>载入完整剧情示例（新项目）</button><p>保留原项目文件；新版本请使用另一个文件夹。</p>
      {:else if selectedStage === 1}{#if shots.length}<div class="shot-list">{#each shots as shot}<div><b>{shot.id}</b><span>{shot.text}</span><small>{shot.duration} 秒</small><details><summary>画面与来源</summary><p>视觉动作：{shot.scene_context?.visual_action || shot.text}</p>{#each [["构图", "framing"], ["氛围", "mood"], ["负面提示词", "negative"], ["连续性", "continuity"]] as field}{#if shot.scene_context?.[field[1]]}<p>{field[0]}：{shot.scene_context[field[1]]}</p>{/if}{/each}<p>场景：{shot.scene_id}</p>{#if shot.scene_context?.source_spans?.length}<p>原文引用：{shot.scene_context.source_spans.join("、")}</p>{/if}</details></div>{/each}</div>{:else}<p>创建项目后显示分镜列表。</p>{/if}
      {:else if selectedStage === 2}<label>画面来源<select bind:value={imageSource} disabled={busy}><option value="dashscope">阿里云 DashScope 生图</option><option value="local">本地故事卡</option></select></label>{#if imageSource === "dashscope"}<details class="model-options"><summary>模型与尺寸</summary><div class="model-fields"><label>生图模型<input bind:value={imageModel} disabled={busy} placeholder="沿用已有阿里云配置"></label><label>图片尺寸<input bind:value={imageSize} disabled={busy} placeholder="例如 720*1280"></label></div></details><p>按未缓存镜头生图，使用阿里云账户额度。</p>{/if}
      {#if shots.length}<details><summary>镜头管理与重试</summary><select bind:value={selectedShot} disabled={busy}>{#each shots as shot}<option value={shot.id}>{shot.id} · {shot.status}</option>{/each}</select><div class="project-actions"><button disabled={busy || !selectedShot} on:click={() => action("retry")}>重试镜头</button><button disabled={busy || !selectedShot} on:click={() => action("review")}>审核通过</button></div></details>{/if}
      {:else if selectedStage === 3}<VideoServiceSettings disabled={busy} /><label>合成方式<select bind:value={outputMode} disabled={busy}><option value="preview">图片预览</option><option value="render">模型视频</option></select></label>{#if outputMode === "render"}<label>项目视频通道<select bind:value={projectVideoProvider} disabled={busy}><option value="comfyui">ComfyUI · MiniMax H3</option><option value="dashscope">阿里云 · Wan</option></select></label><details><summary>视频高级参数</summary>
      {#if projectVideoProvider === "dashscope"}
      <label>视频模型<input bind:value={videoOptions.model} disabled={busy} /></label>
      <label>生成尺寸<input bind:value={videoOptions.size} disabled={busy} placeholder="1280*720" /></label>
      <label>每镜时长（秒）<input type="number" min={["wanx2.1-t2v-turbo", "wan2.1-t2v-turbo"].includes(videoOptions.model) ? 5 : 1} max={["wanx2.1-t2v-turbo", "wan2.1-t2v-turbo"].includes(videoOptions.model) ? 5 : 10} step="1" bind:value={videoOptions.duration_seconds} disabled={busy} /></label>
      <p>{["wanx2.1-t2v-turbo", "wan2.1-t2v-turbo"].includes(videoOptions.model) ? "当前模型仅支持 5 秒。" : "时长支持范围由所选模型决定。"}</p>
      {:else}
      <label>宽度<input type="number" min="64" max="4096" step="16" bind:value={videoOptions.width} disabled={busy} /></label>
      <label>高度<input type="number" min="64" max="4096" step="16" bind:value={videoOptions.height} disabled={busy} /></label>
      <label>帧数<input type="number" min="1" max="4096" step="1" bind:value={videoOptions.length} disabled={busy} /></label>
      <label>帧率<input type="number" min="1" step="1" bind:value={videoOptions.fps} disabled={busy} /></label>
      <label>随机种子<input type="number" min="0" step="1" bind:value={videoOptions.seed} disabled={busy} /></label>
      <label><input type="checkbox" bind:checked={videoOptions.turbo} disabled={busy} /> Turbo</label>
      {/if}<p>开始作业时保存；恢复沿用已保存参数。修改后开始作业会重新生成视频，复用图片与音频。</p></details><p>逐镜生成视频后合成成片；阿里云按实际生成计费。恢复沿用项目模式；视频失败时保留图片预览，可再次恢复。</p>{/if}<p>镜头素材就绪后自动合成为 MP4，模型视频保留原音频，无音频镜头使用静音音轨。</p>{#if fallbackShots.length}<div role="alert"><p>{fallbackShots.length} 个镜头使用图片回退，模型视频尚未全部完成。</p>{#each fallbackShots as shot}<p><strong>{shot.id}</strong>：{shot.error}</p>{/each}<p>修复服务后使用底部“恢复”，已完成的图片和音频会复用。</p></div>{/if}<p>总时长：{shots.reduce((sum: number, shot: any) => sum + shot.duration, 0).toFixed(1)} 秒</p>
      {:else}<p>{videoUrl ? "成片已就绪，可在右侧播放。" : "流水线完成后在这里查看成片。"}</p><button on:click={() => previewOpen = true}>显示预览</button>{/if}
      {#if error}<div class="error-box" role="alert">{error}</div>{/if}
    </section>{/if}
  </section>
  {#if previewOpen}<aside class="pipeline-preview"><div class="detail-heading"><h3>成片预览</h3><span>MP4 · 预览</span></div>
    <div class="preview-screen" class:empty={!videoUrl}>
      {#if videoUrl}
        {#key videoUrl}
          <video controls preload="metadata" src={videoUrl}
            on:loadedmetadata={() => playback = "视频已加载，可以播放"}
            on:playing={() => playback = "正在播放"}
            on:ended={() => playback = "播放完成"}
            on:error={() => playback = "视频加载失败，请重新打开项目或生成"}>
            <track kind="captions" label="故事字幕" srclang="zh" src={captionsUrl} default>
          </video>
        {/key}
      {:else}
        <div class="preview-placeholder">
          <div class="preview-icon" aria-hidden="true"><svg width="28" height="28" viewBox="0 0 24 24" fill="none"><rect x="3" y="4" width="18" height="16" rx="3" stroke="currentColor" stroke-width="1.3"/><path d="m10 8 6 4-6 4V8Z" fill="currentColor"/></svg></div>
          <h3>{busy ? "正在准备你的预览" : error ? "预览暂不可用" : "故事的下一幕，在这里"}</h3>
          <p>{busy ? "处理完成后，视频会显示在这里" : error ? "修复错误后，可继续恢复当前项目" : "创建或打开项目，生成后即可播放"}</p>
        </div>
      {/if}
    </div>
    <div class="preview-caption"><span role="status">{playback || (busy ? "正在生成素材与合成视频…" : "预览工作区")}</span><span>画面 · 时间轴 · 成片</span></div>
  </aside>{/if}
</div>
<footer class="job-controls"><div class="job-status"><span class:working={busy}>●</span><span role="status">{status}</span></div><div class="project-actions"><button class="primary" disabled={busy || choosing} on:click={start}>{busy ? "作业进行中…" : !project ? "选择项目" : !story.trim() ? "填写故事" : "▶ 开始作业"}</button><button disabled title="当前后台暂不支持安全暂停">暂停</button><button disabled={busy || !manifest} on:click={() => action("resume")}>恢复</button><button aria-expanded={logsOpen} on:click={() => logsOpen = !logsOpen}>日志 {logsOpen ? "⌃" : "⌄"}</button></div></footer>
{#if logsOpen}<div class="job-logs"><h3>运行日志</h3><pre>{logs.join("\n") || "暂无运行记录"}</pre>{#if error}<p role="alert">{error}</p>{/if}</div>{/if}

<style>
.directory-name{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-align:left}.directory-change{flex-shrink:0}.project-picker>summary>span:nth-child(2){min-width:0;flex:1}
.current-path{display:block;font-size:11px;color:#617895;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:300px;margin-top:5px}.recent-projects{max-height:260px;overflow:auto;margin-bottom:14px}.recent-project{display:block;width:100%;text-align:left;margin-bottom:7px;padding:12px;background:#f8fbff}.recent-project.current{border-color:#3979c6;background:#eaf3ff}.recent-project strong{font-size:13px;display:block}.recent-project span{display:block;font-size:11px;overflow-wrap:anywhere;color:#617895;margin:5px 0}.recent-project small{font-size:10px}.recent-projects>p{font-size:12px;color:#617895}

.project-picker{position:relative;min-width:0;width:min(100%,440px)}.project-picker>summary{display:flex;align-items:center;gap:12px;cursor:pointer;list-style:none;padding:4px}.project-picker small{display:block;font-size:10px;color:#7f8d9d;margin-bottom:4px}.project-picker strong{display:block;font-size:14px;font-weight:500;overflow-wrap:anywhere}.folder-symbol{display:grid;place-items:center;width:36px;height:36px;background:#e8f2ff;color:#3979c6;border-radius:8px}.picker-chevron{margin-left:auto;color:#94a1af}.project-picker[open] .picker-chevron{transform:rotate(180deg)}.project-menu{position:absolute;top:calc(100% + 14px);left:-4px;width:min(440px,calc(100vw - 70px));z-index:20;padding:18px;background:#ffffff;border:1px solid #cadaeb;border-radius:10px;box-shadow:0 16px 40px #8aa3c333}.project-menu>p:first-child{font-size:12px;margin:0 0 14px}.directory-choice{display:flex;justify-content:space-between;width:100%;gap:12px}.selected-directory{font-size:11px;line-height:1.7;color:#8293a5;overflow-wrap:anywhere;margin:12px 0 18px}.project-menu .project-actions button{flex:1}

  .project-toolbar{display:flex;align-items:end;gap:12px;padding:12px 16px;background:#ffffff;border:1px solid var(--border);border-radius:10px;margin-bottom:18px}
  .project-actions{display:flex;gap:8px;flex-shrink:0}.project-actions button{font-size:12px}.project-state{margin-left:auto;align-self:center;color:#718296;font-size:11px;white-space:nowrap}.project-state.working{color:var(--accent)}

.pipeline-heading,.detail-heading{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:20px}.pipeline-heading h2{font-size:19px;margin:0}.pipeline-heading .eyebrow{font-size:9px;letter-spacing:2px;color:#758595}.pipeline-layout{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:18px;min-height:410px}.pipeline-layout.without-preview{grid-template-columns:1fr}.pipeline-canvas{min-width:0;border:1px solid var(--border);border-radius:10px;padding:24px;background-color:#f6faff;background-image:radial-gradient(#cadbed77 1px,transparent 1px);background-size:18px 18px}.pipeline-nodes{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:16px;margin:26px 0}.pipeline-node{position:relative;display:flex;flex-direction:column;align-items:start;text-align:left;gap:13px;min-width:0;padding:15px 12px;background:#ffffff;border:1px solid #d4e2f0;border-radius:8px}.pipeline-node:not(:last-child):after{content:"→";position:absolute;right:-15px;top:48%;color:#8cabc8}.pipeline-node.selected{border-color:#3979c6;box-shadow:0 0 0 1px #3979c6}.pipeline-node.complete .node-state{color:#3979c6}.pipeline-node.failed{border-color:#d47b73}.pipeline-node.failed .node-state{color:#e9948e}.node-number{color:#607b9c;font-size:10px}.pipeline-node strong{font-size:16px}.node-state{font-size:11px;color:#547295}.node-progress,.pipeline-node small{font-size:10px;color:#617a98}.canvas-hint{text-align:center;padding:42px 12px;color:#64809f}.canvas-hint h3{font-size:14px;font-weight:400}.canvas-hint p{font-size:12px}.node-detail{background:#ffffff;border:1px solid #d8e4f1;border-radius:8px;padding:20px}.node-detail p{font-size:12px;color:#607996;line-height:1.8}.detail-heading h3{font-size:14px;margin:0}.detail-heading span{font-size:10px;color:#607996}.node-detail textarea{height:140px}.node-detail summary{cursor:pointer;font-size:12px;margin:14px 0}.model-fields{display:grid;grid-template-columns:1fr 1fr;gap:12px}.shot-list{max-height:240px;overflow:auto}.shot-list>div{display:flex;flex-wrap:wrap;gap:14px;padding:12px 0;border-bottom:1px solid #d6e3ef;font-size:12px}.shot-list span{flex:1;min-width:160px}.shot-list details{flex-basis:100%;overflow-wrap:anywhere}.shot-list details p{margin:8px 0}.shot-list small{white-space:nowrap;color:#607996}.pipeline-preview{padding:18px;background:#ffffff;border:1px solid var(--border);border-radius:10px;min-width:0}.preview-screen{height:300px;display:grid;place-items:center;background:#e9f3fc;border-radius:8px;overflow:hidden}.preview-screen video{width:100%;height:100%;object-fit:contain;min-height:0}.preview-placeholder{text-align:center;color:#567797;padding:12px}.preview-icon{margin:12px}.preview-placeholder h3{font-size:13px;font-weight:400}.preview-placeholder p{font-size:11px}.preview-caption{font-size:10px;color:#567797;margin-top:14px}.preview-caption>span:last-child{display:none}.job-controls{position:sticky;bottom:0;background:#ffffff;border:1px solid #cedfec;border-radius:10px;display:flex;justify-content:space-between;align-items:center;gap:16px;padding:16px;margin-top:18px;z-index:5}.job-status{display:flex;gap:9px;font-size:11px;min-width:0;color:#526f91}.job-status>span:first-child{color:#3979c6}.job-logs{padding:18px;border:1px solid var(--border);background:#f8fbff;border-radius:8px;margin-top:10px}.job-logs h3{font-size:12px}.job-logs pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:200px;overflow:auto;font-size:11px}.error-box{color:#e9948e;overflow-wrap:anywhere;margin-top:14px;font-size:12px}
@media(max-width:1200px){.pipeline-layout{grid-template-columns:minmax(0,1fr)}.pipeline-preview .preview-screen{height:220px}.pipeline-canvas{padding:18px}.job-controls{flex-wrap:wrap}}@media(max-width:720px){.pipeline-nodes{gap:9px}.pipeline-node{padding:10px 6px;gap:9px}.pipeline-node strong{font-size:13px}.pipeline-node:not(:last-child):after{right:-10px;font-size:10px}.pipeline-canvas{padding:10px}.job-controls .project-actions{flex-wrap:wrap}.model-fields{grid-template-columns:1fr}}

.pipeline-heading .eyebrow{color:#687f9e;letter-spacing:1px;font-size:12px}.pipeline-heading h2{font-size:20px;color:#35597e}.pipeline-canvas{background-color:#f8fbff;background-image:linear-gradient(#edf3fa 1px,transparent 1px),linear-gradient(90deg,#edf3fa 1px,transparent 1px);background-size:24px 24px;border-color:#d6e4f3;border-radius:18px}.pipeline-nodes{gap:18px}.pipeline-node{border-radius:12px 12px 18px 12px;border-top:4px solid #91bae8;box-shadow:0 5px 0 #e1ebf666;padding:18px 14px;gap:14px}.pipeline-node:nth-child(2){border-top-color:#b5a7d9}.pipeline-node:nth-child(3){border-top-color:#e8adc6}.pipeline-node:nth-child(4){border-top-color:#8ac8c5}.pipeline-node:nth-child(5){border-top-color:#e7c286}.pipeline-node.selected{background:#edf5ff;box-shadow:0 0 0 2px #a3c7ed}.node-number{font-size:12px;background:#edf4fc;border-radius:20px;padding:4px 8px}.pipeline-node strong{font-size:18px;color:#355576}.node-state{font-size:12px}.node-progress{font-size:11px}.project-toolbar{border-radius:14px;box-shadow:0 3px 16px #416e9b08}.project-menu{box-shadow:0 12px 32px #50759c25}.pipeline-preview{border-radius:18px;box-shadow:0 8px 26px #45628d08}.preview-screen{background:linear-gradient(160deg,#dcefff,#f6f9ff 60%,#fceaf1);border:6px solid #f2f6fb}.preview-placeholder h3{color:#42678e}.preview-placeholder p{font-size:12px}.preview-icon{color:#6795c5}.job-controls{box-shadow:0 6px 25px #446b9610;border-radius:14px}.canvas-hint{padding:32px 12px}.canvas-hint h3{font-size:16px;color:#587799}.canvas-hint p{font-size:12px}.storyboard-art{position:relative;display:flex;justify-content:center;gap:12px;height:80px;margin:0 auto 20px;width:220px}.storyboard-art i{display:block;width:56px;height:72px;border:4px solid white;border-radius:4px;background:linear-gradient(155deg,#b8dafa 50%,#dbeefb 51%);box-shadow:0 3px 8px #7797bd20;transform:rotate(-12deg)}.storyboard-art i:nth-child(2){background:linear-gradient(155deg,#ead6ee 50%,#f5e6f0 51%);transform:translateY(-7px)}.storyboard-art i:nth-child(3){background:linear-gradient(155deg,#c3e4e1 50%,#e1f3ee 51%);transform:rotate(12deg)}.storyboard-art b{position:absolute;right:4px;top:-16px;color:#dca1ba;font-size:25px}.detail-heading h3{color:#36597e}.node-detail{box-shadow:0 5px 16px #587fad0a}.pipeline-node.failed{background:#fff3f4;color:#a53d51}
@media(max-width:720px){.pipeline-node{padding:9px 5px}.pipeline-node strong{font-size:13px}.pipeline-nodes{gap:9px}.node-number{padding:3px 6px}.node-state{font-size:11px}}
</style>
