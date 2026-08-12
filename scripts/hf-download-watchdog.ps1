# Dedicated watchdog for this one DeepSeek V4 Flash download on Windows.
# It only restarts a provably stalled (zero-byte for five minutes) hf worker;
# completed shards and Hugging Face's partial-cache files are never removed.
$ErrorActionPreference = 'Stop'

$root = 'D:\ds4f-build\hf'
$downloadDir = Join-Path $root '.cache\huggingface\download'
$log = 'D:\ds4f-build\hf-watchdog.log'
$taskName = 'DS4F-Official-Download-User'

while ($true) {
    try {
        $count = (Get-ChildItem $root -Filter 'model-*.safetensors' -File).Count
        if ($count -ge 46) {
            Add-Content -LiteralPath $log -Value ((Get-Date -Format s) + ' complete')
            break
        }

        $partial = Get-ChildItem $downloadDir -Force -File |
            Where-Object { $_.Name -like '*.incomplete' } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        $hf = Get-Process hf -ErrorAction SilentlyContinue
        $stalled = $null -ne $hf -and $null -ne $partial -and
            $partial.Length -eq 0 -and
            ((Get-Date) - $partial.LastWriteTime).TotalMinutes -ge 5

        if ($stalled) {
            $workers = Get-CimInstance Win32_Process | Where-Object {
                ($_.Name -eq 'hf.exe' -or $_.Name -eq 'python.exe') -and
                $_.CommandLine -like '*deepseek-ai/DeepSeek-V4-Flash*'
            }
            foreach ($worker in $workers) {
                Stop-Process -Id $worker.ProcessId -Force -ErrorAction SilentlyContinue
            }
            Start-Sleep -Seconds 2
            schtasks /Run /TN $taskName | Out-Null
            Add-Content -LiteralPath $log -Value (
                (Get-Date -Format s) + " restarted stalled download at $count/46")
        }
    } catch {
        Add-Content -LiteralPath $log -Value (
            (Get-Date -Format s) + ' watchdog error: ' + $_.Exception.Message)
    }
    Start-Sleep -Seconds 120
}
