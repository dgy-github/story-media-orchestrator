param(
    [string]$CapabilityUrl = $env:MICROCODEX_CAPABILITY_URL,
    [string]$CapabilityToken = $env:MICROCODEX_CAPABILITY_TOKEN,
    [int]$SidecarPort = 8765,
    [string]$ComfyUrl = "http://61.157.218.59:31340"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$storyRoot = "D:\github_dgy\microcodex-short-drama-studio"
$imageRoot = "D:\github_dgy\story-image-agent"
$videoRoot = "D:\github_dgy\story-video-agent"
$eventLog = Join-Path $storyRoot "sidecar\campaign_events.db"
if (!(Test-Path $storyRoot) -or !(Test-Path $imageRoot) -or !(Test-Path $videoRoot)) { throw "sibling agent directory missing" }

# Reuse a valid user-scoped token, otherwise generate one once.
$sidecarToken = [Environment]::GetEnvironmentVariable("MICROCODEX_SIDECAR_TOKEN", "User")
if (!$sidecarToken -or $sidecarToken.Length -lt 32) {
    $bytes = [byte[]]::new(32)
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $sidecarToken = [Convert]::ToBase64String($bytes)
    [Environment]::SetEnvironmentVariable("MICROCODEX_SIDECAR_TOKEN", $sidecarToken, "User")
}
$env:MICROCODEX_SIDECAR_TOKEN = $sidecarToken
$env:STORY_SIDECAR_URL = "http://127.0.0.1:$SidecarPort"
$env:STORY_SIDECAR_TOKEN = $sidecarToken
$env:STORY_IMAGE_AGENT_ROOT = $imageRoot
$env:STORY_VIDEO_AGENT_ROOT = $videoRoot
$env:MINIMAX_H3_COMFYUI_BASE_URL = $ComfyUrl
if ($CapabilityUrl) { $env:MICROCODEX_CAPABILITY_URL = $CapabilityUrl }
if ($CapabilityToken) { $env:MICROCODEX_CAPABILITY_TOKEN = $CapabilityToken }

$python = Join-Path $storyRoot ".venv\Scripts\python.exe"
if (!(Test-Path $python)) { $python = "python" }
$log = Join-Path $root "sidecar.log"
$errLog = Join-Path $root "sidecar.error.log"
$args = @("-m", "sidecar.story_sidecar", "--host", "127.0.0.1", "--port", "$SidecarPort", "--event-log", $eventLog)
Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $storyRoot -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $errLog
Start-Sleep -Seconds 2
$health = $null
try {
    $health = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 -Uri "$($env:STORY_SIDECAR_URL)/health" -Headers @{ Authorization = "Bearer $sidecarToken" }
} catch { }
Write-Output "orchestrator environment configured"
Write-Output "sidecar token configured (length=$($sidecarToken.Length))"
Write-Output "sidecar health=$([bool]$health)"
Write-Output "comfyui=$ComfyUrl"
Write-Output "log=$log"
if (!$health) { Write-Warning "sidecar did not pass health check; inspect sidecar.log" }
