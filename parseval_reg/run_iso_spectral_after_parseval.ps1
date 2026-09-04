# 等 Parseval（12 个 run）完成后，依次跑 ISO（12）和 Spectral（12）。detached 进程，避免被会话后台杀掉。
$base = 'C:\Users\杨斯杰\Desktop\第四周\parseval_reg\parseval_reg'
$py = 'D:/anaconda/envs/parseval/python.exe'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

function Wait-Done($pattern, $target) {
    while ($true) {
        $logs = Get-ChildItem "$base\logs\reg_full" -Filter '*.log' | Where-Object { $_.Name -like $pattern }
        $done = ($logs | Where-Object { Select-String -Path $_.FullName -Pattern 'done RL run' -Quiet }).Count
        if ($done -ge $target) { break }
        Start-Sleep -Seconds 300
    }
}

function Launch-Method($flag, $suffixPrefix) {
    foreach ($s in @('0.0001','0.001','0.01','0.1')) {
        foreach ($seed in @('1','2','3')) {
            $suffix = "${suffixPrefix}_${s}_seed${seed}"
            $argList = @('main.py','--env','metaworld_sequence_set0','--algorithm','base','--repeat_idx','0','--seed',"$seed",'--learning_rate','0.0003','--num_steps','10000000',$flag,"$s",'--wandb','--wandb_online','--save_suffix',"$suffix")
            Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $base -RedirectStandardOutput "$base\logs\reg_full\${suffix}.log" -RedirectStandardError "$base\logs\reg_full\${suffix}.log.err" -WindowStyle Hidden
        }
    }
}

# 阶段 1：等 Parseval 12 个完成
Wait-Done 'parseval_0.*_seed*.log' 12

# 阶段 2：跑 ISO，等完成
Launch-Method '--iso_reg' 'iso'
Wait-Done 'iso_0.*_seed*.log' 12

# 阶段 3：跑 Spectral，等完成
Launch-Method '--spectral_reg' 'spectral'
Wait-Done 'spectral_0.*_seed*.log' 12
