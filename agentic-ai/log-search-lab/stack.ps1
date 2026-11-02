<#
  Stop and start the local log-search stack: Elasticsearch, Kibana, and optionally Docker Desktop and Ollama.

  .\stack.ps1 status                 what is running
  .\stack.ps1 stop                   stop Elasticsearch and Kibana (the logs are kept in the Docker volume)
  .\stack.ps1 stop -Quit             ... and also quit Docker Desktop and shut WSL down, to free the memory
  .\stack.ps1 start                  start Docker Desktop if needed, then Elasticsearch and Kibana, and wait until ready
  .\stack.ps1 start -Ollama          ... and also start the Ollama server for the local model
  .\stack.ps1 stop -Ollama           ... and also stop the Ollama server

  If PowerShell blocks the script, run it as:  powershell -ExecutionPolicy Bypass -File .\stack.ps1 status
#>
param(
    [Parameter(Mandatory = $true)][ValidateSet("start", "stop", "status")][string]$Action,
    [switch]$Quit,    # with stop: also quit Docker Desktop and WSL
    [switch]$Ollama   # also start or stop the Ollama server
)

Set-Location $PSScriptRoot  # docker compose reads docker-compose.yml from this folder

# Use docker from the PATH if it is there, otherwise from the install folder on F:
$docker = (Get-Command docker -ErrorAction SilentlyContinue).Source
if (-not $docker) { $docker = "F:\AI_training\docker-app\resources\bin\docker.exe" }
$dockerDesktop = "F:\AI_training\docker-app\Docker Desktop.exe"


function Test-Url($url) {
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 $url | Out-Null; return $true } catch { return $false }
}

# `docker info` can hang while the engine is down, so ask it in a background job with a time limit
function Test-Engine {
    $job = Start-Job { param($d) & $d info --format "{{.ServerVersion}}" 2>$null; $LASTEXITCODE } -ArgumentList $docker
    if (Wait-Job $job -Timeout 15) {
        $out = @(Receive-Job $job); Remove-Job $job
        return ($out.Count -gt 0 -and $out[-1] -eq 0)
    }
    Stop-Job $job; Remove-Job $job -Force
    return $false
}

function Wait-Until($what, $seconds, [scriptblock]$ready) {
    Write-Host -NoNewline "Waiting for $what "
    for ($i = 0; $i -lt $seconds / 5; $i++) {
        if (& $ready) { Write-Host " ready"; return $true }
        Write-Host -NoNewline "."; Start-Sleep -Seconds 5
    }
    Write-Host " not ready after $seconds seconds"
    return $false
}


function Show-Status {
    Write-Host "Docker engine : $(if (Test-Engine) { 'running' } else { 'not running' })"
    Write-Host "Elasticsearch : $(if (Test-Url 'http://localhost:9200') { 'up   http://localhost:9200' } else { 'down' })"
    Write-Host "Kibana        : $(if (Test-Url 'http://localhost:5601/api/status') { 'up   http://localhost:5601/app/discover' } else { 'down' })"
    Write-Host "Ollama        : $(if (Test-Url 'http://localhost:11434') { 'up   http://localhost:11434' } else { 'down' })"
}


function Start-Stack {
    if (-not (Test-Engine)) {
        Write-Host "Starting Docker Desktop..."
        Start-Process $dockerDesktop
        if (-not (Wait-Until "the Docker engine" 180 { Test-Engine })) { return }
    }
    & $docker compose up -d
    Wait-Until "Elasticsearch" 180 { Test-Url "http://localhost:9200/_cluster/health" } | Out-Null
    Wait-Until "Kibana" 180 { Test-Url "http://localhost:5601/api/status" } | Out-Null
    if ($Ollama -and -not (Test-Url "http://localhost:11434")) {
        Write-Host "Starting Ollama..."
        Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
        Wait-Until "Ollama" 60 { Test-Url "http://localhost:11434" } | Out-Null
    }
}


function Stop-Stack {
    if (Test-Engine) {
        & $docker compose down  # removes the containers but keeps the esdata volume, so the logs survive
    }
    if ($Quit) {
        Write-Host "Quitting Docker Desktop and shutting WSL down..."
        & $docker desktop stop 2>$null
        Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue | Stop-Process -Force
        wsl --shutdown
    }
    if ($Ollama) {
        Write-Host "Stopping Ollama..."
        Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
    }
}


switch ($Action) {
    "start"  { Start-Stack }
    "stop"   { Stop-Stack }
    "status" { }
}
Write-Host ""
Show-Status
