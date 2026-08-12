# Restart only the fixed DeepSeek V4 Flash Hugging Face download command.
# Cached shards and incomplete files are deliberately preserved for resume.
$ErrorActionPreference = 'Stop'
$needle = 'deepseek-ai/DeepSeek-V4-Flash'
$workers = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -eq 'hf.exe' -or $_.Name -eq 'python.exe') -and
    $_.CommandLine -like "*$needle*"
}

foreach ($worker in $workers) {
    Stop-Process -Id $worker.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2
schtasks /Run /TN DS4F-Official-Download-User | Out-Null
Write-Output ("restarted workers=" + $workers.Count)
