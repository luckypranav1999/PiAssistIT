
$baseUrl = "http://127.0.0.1:8080/api/chat"

$batchSize = 6
$totalRounds = 5
$intervalMinutes = 5

$results = @()

$questions = @(
    "What are common causes of application server performance issues?",
    "How can I troubleshoot high CPU utilization on a production server?",
    "What causes database connection pool exhaustion?",
    "How do I investigate slow API response times?",
    "What are the common reasons for memory leaks in applications?",
    "How can I troubleshoot intermittent HTTP 500 errors?",
    "What causes increased latency in microservices?",
    "How do I identify database query performance bottlenecks?",
    "What are common causes of application timeout errors?",
    "How can I investigate unexpected application restarts?",
    "What causes high memory consumption in web applications?",
    "How do I troubleshoot connection failures between services?",
    "What are common causes of slow database transactions?",
    "How can I investigate sudden increases in API traffic?",
    "What causes thread pool starvation in application servers?",
    "How do I troubleshoot intermittent DNS resolution failures?",
    "What are common causes of HTTP 502 and 503 errors?",
    "How can I investigate a sudden increase in server response time?",
    "What causes repeated database connection timeouts?",
    "How do I troubleshoot application startup failures?",
    "What are common causes of CPU spikes in backend services?",
    "How can I investigate failed requests in a distributed system?",
    "What causes network latency between application servers?",
    "How do I troubleshoot slow application deployments?",
    "What are common causes of authentication failures in APIs?",
    "How can I identify resource bottlenecks in a production application?",
    "What causes increased error rates after a software deployment?",
    "How do I troubleshoot a service that becomes unresponsive?",
    "What are common causes of excessive database queries?",
    "How can I investigate performance degradation over time?"
)

Write-Host "`nStarting PiAssistIT telemetry workload..." -ForegroundColor Cyan
Write-Host "Total requests: $($questions.Count)"
Write-Host "Batches: $totalRounds"
Write-Host "Requests per batch: $batchSize"
Write-Host "Pause between batches: $intervalMinutes minutes`n"

for ($round = 1; $round -le $totalRounds; $round++) {

    Write-Host "========== BATCH $round / $totalRounds ==========" -ForegroundColor Yellow

    $startIndex = ($round - 1) * $batchSize
    $endIndex = $startIndex + $batchSize - 1

    for ($i = $startIndex; $i -le $endIndex; $i++) {

        $question = $questions[$i]
        $requestStart = Get-Date

        Write-Host "`nRequest $($i + 1)/$($questions.Count): $question"

        try {
            $body = @{
                question = $question
            } | ConvertTo-Json

            $response = Invoke-RestMethod `
                -Uri $baseUrl `
                -Method POST `
                -ContentType "application/json" `
                -Body $body `
                -TimeoutSec 120

            $duration = [math]::Round(
                ((Get-Date) - $requestStart).TotalSeconds, 2
            )

            Write-Host "SUCCESS - ${duration}s" -ForegroundColor Green

            $results += [PSCustomObject]@{
                Round       = $round
                Request     = $i + 1
                Question    = $question
                Status      = "SUCCESS"
                DurationSec = $duration
                Timestamp   = $requestStart.ToUniversalTime().ToString("o")
                Error       = ""
            }
        }
        catch {
            $duration = [math]::Round(
                ((Get-Date) - $requestStart).TotalSeconds, 2
            )

            Write-Host "FAILED - $($_.Exception.Message)" -ForegroundColor Red

            $results += [PSCustomObject]@{
                Round       = $round
                Request     = $i + 1
                Question    = $question
                Status      = "FAILED"
                DurationSec = $duration
                Timestamp   = $requestStart.ToUniversalTime().ToString("o")
                Error       = $_.Exception.Message
            }
        }

        Start-Sleep -Seconds 5
    }

    Write-Host "`nBatch $round completed." -ForegroundColor Cyan

    if ($round -lt $totalRounds) {
        Write-Host "Waiting $intervalMinutes minutes before next batch..."
        Start-Sleep -Seconds ($intervalMinutes * 60)
    }
}

$csvPath = Join-Path $PSScriptRoot "telemetry_workload_results_2.csv"

$results | Export-Csv -Path $csvPath -NoTypeInformation

$successCount = @($results | Where-Object { $_.Status -eq "SUCCESS" }).Count
$failedCount = @($results | Where-Object { $_.Status -eq "FAILED" }).Count

Write-Host "`n========== WORKLOAD SUMMARY ==========" -ForegroundColor Cyan
Write-Host "Total requests : $($results.Count)"
Write-Host "Successful     : $successCount" -ForegroundColor Green
Write-Host "Failed         : $failedCount" -ForegroundColor Red
Write-Host "Results saved  : $csvPath"
Write-Host "`nWorkload completed." -ForegroundColor Cyan
