# ======================================
#   PowerShell Git Grass Generator v2
#   Repo-local: run inside repo root
# ======================================

$repo = Split-Path -Leaf (Get-Location)

if (!(Test-Path log.txt)) { "Grass log start for $repo" | Out-File -Encoding utf8 log.txt }

$rand = [System.Random]::new()

# 기간은 레포명에 따라 자동 설정 (원하면 수정)
switch -Wildcard ($repo) {
    "keras"       { $start = Get-Date "2025-05-13"; $end = Get-Date "2025-09-30" }
    "keras2"      { $start = Get-Date "2025-07-01"; $end = Get-Date "2025-09-30" }
    "torch"       { $start = Get-Date "2025-06-01"; $end = Get-Date "2025-09-30" }

    "fastapi"     { $start = Get-Date "2025-10-01"; $end = Get-Date "2025-10-20" }
    "html"        { $start = Get-Date "2025-10-01"; $end = Get-Date "2025-10-20" }
    "javascript"  { $start = Get-Date "2025-10-01"; $end = Get-Date "2025-10-20" }
    "autoencoder" { $start = Get-Date "2025-10-01"; $end = Get-Date "2025-10-20" }
    "python"      { $start = Get-Date "2025-10-01"; $end = Get-Date "2025-10-20" }
    "python_import" { $start = Get-Date "2025-10-01"; $end = Get-Date "2025-10-20" }
    "openai"      { $start = Get-Date "2025-10-01"; $end = Get-Date "2025-10-20" }
    "pandas"      { $start = Get-Date "2025-10-01"; $end = Get-Date "2025-10-20" }
    "llm"         { $start = Get-Date "2025-10-01"; $end = Get-Date "2025-10-20" }
    "ml"          { $start = Get-Date "2025-10-01"; $end = Get-Date "2025-10-20" }
    default { Write-Host "⚠️ 자동 기간이 설정되지 않음. 스크립트 상단 switch 수정하세요."; exit 1 }
}

for ($d = $start; $d -le $end; $d = $d.AddDays(1)) {
    $count = $rand.Next(1,6)  # 1..5개

    for ($i=0; $i -lt $count; $i++) {
        $hour   = $rand.Next(20,23)   # 20~22시
        $minute = $rand.Next(0,60)
        $second = $rand.Next(0,60)

        $dt     = Get-Date -Year $d.Year -Month $d.Month -Day $d.Day -Hour $hour -Minute $minute -Second $second
        $stamp  = "{0:yyyy-MM-dd HH:mm:ss} +0900" -f $dt

        $env:GIT_AUTHOR_DATE    = $stamp
        $env:GIT_COMMITTER_DATE = $stamp

        Add-Content log.txt "commit $i on $stamp"
        git add .
        git commit -m "$repo update $i on $stamp"
    }

    git push origin main
}

Remove-Item env:GIT_AUTHOR_DATE, env:GIT_COMMITTER_DATE -ErrorAction SilentlyContinue
Write-Host "✅ [$repo] done: $($start.ToShortDateString()) ~ $($end.ToShortDateString())"

