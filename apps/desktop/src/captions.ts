export function projectCaptions(shots: {duration: number; text: string; subtitle?: string}[]): string {
  const timestamp = (seconds: number) => {
    const ms = Math.round(seconds * 1000);
    return `${String(Math.floor(ms / 3600000)).padStart(2, "0")}:${String(Math.floor(ms / 60000) % 60).padStart(2, "0")}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")}.${String(ms % 1000).padStart(3, "0")}`;
  };
  let clock = 0;
  const cues = shots.map((shot, index) => {
    if (!Number.isFinite(shot.duration) || shot.duration <= 0) throw new Error("无效字幕时长");
    const start = clock;
    clock += shot.duration;
    const text = (shot.subtitle || shot.text || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\r?\n\s*\r?\n/g, "\n");
    return `${index + 1}\n${timestamp(start)} --> ${timestamp(clock)}\n${text}`;
  });
  return `WEBVTT\n\n${cues.join("\n\n")}\n`;
}
