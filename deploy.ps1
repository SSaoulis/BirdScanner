<#
.SYNOPSIS
    Redeploy the BirdScanner docker-compose stack on the Raspberry Pi over SSH.

.DESCRIPTION
    Replicates the manual redeploy: SSH into the Pi, `cd` to the checkout,
    `git pull`, then `docker compose up --build`.

    Configuration is read from `deploy.env` (see deploy.env.example) sitting
    next to this script. Any value can be overridden by a real environment
    variable of the same name. The SSH password is passed to plink via -pw and
    is never echoed.

    Requires plink.exe (PuTTY) on PATH.

.EXAMPLE
    ./deploy.ps1
        Build + (re)start using the settings in deploy.env.

.EXAMPLE
    ./deploy.ps1 -Detach
        Same, but start the stack detached (docker compose up --build -d).
#>
[CmdletBinding()]
param(
    # Start the stack detached (-d) instead of streaming logs in the foreground.
    [switch]$Detach
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Import-DeployEnv {
    <#
        Load KEY=VALUE pairs from deploy.env into the current process's
        environment, without clobbering values already set in the real
        environment (those win). Blank lines and `#` comments are skipped;
        surrounding single/double quotes on a value are stripped.
    #>
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        throw "deploy.env not found at $Path. Copy deploy.env.example to deploy.env and fill it in."
    }

    foreach ($line in Get-Content -Path $Path) {
        $trimmed = $line.Trim()
        if ($trimmed -eq '' -or $trimmed.StartsWith('#')) { continue }

        $idx = $trimmed.IndexOf('=')
        if ($idx -lt 1) { continue }

        $key = $trimmed.Substring(0, $idx).Trim()
        $value = $trimmed.Substring($idx + 1).Trim()
        if ($value.Length -ge 2 -and
            (($value.StartsWith('"') -and $value.EndsWith('"')) -or
             ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        # Real environment variables take precedence over the file.
        if (-not [Environment]::GetEnvironmentVariable($key)) {
            [Environment]::SetEnvironmentVariable($key, $value)
        }
    }
}

function Get-RequiredEnv {
    <# Fetch an env var, throwing a helpful error when it is missing/blank. #>
    param([string]$Name)

    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "$Name is not set. Add it to deploy.env (see deploy.env.example)."
    }
    return $value
}

if (-not (Get-Command plink -ErrorAction SilentlyContinue)) {
    throw "plink.exe not found on PATH. Install PuTTY (https://www.putty.org) or add plink to PATH."
}

Import-DeployEnv -Path (Join-Path $scriptDir 'deploy.env')

$piHost = Get-RequiredEnv 'PI_SSH_HOST'
$user = Get-RequiredEnv 'PI_SSH_USER'
$password = Get-RequiredEnv 'PI_SSH_PASSWORD'
$port = [Environment]::GetEnvironmentVariable('PI_SSH_PORT')
if ([string]::IsNullOrWhiteSpace($port)) { $port = '22' }
$repoPath = Get-RequiredEnv 'PI_REPO_PATH'

$composeUp = 'docker compose up --build'
if ($Detach) { $composeUp += ' -d' }

# Fail fast at each step; quote the path in case it ever contains spaces.
$remoteCommand = "cd '$repoPath' && git pull && $composeUp"

Write-Host "Deploying to $user@${piHost}:$port ($repoPath)..." -ForegroundColor Cyan

# plink args as an array so the password is never string-interpolated into a
# command line (and special chars in it survive). -batch would abort on an
# unknown host key; we omit it so the first run can cache the Pi's key.
$plinkArgs = @(
    '-ssh'
    '-P', $port
    '-pw', $password
    "$user@$piHost"
    $remoteCommand
)

& plink @plinkArgs
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host "Deploy failed (plink exit code $exitCode)." -ForegroundColor Red
    exit $exitCode
}

Write-Host "Deploy finished." -ForegroundColor Green
