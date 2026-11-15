<#
    Atalho para rodar o codigo deste trabalho.

    Existe porque, neste Windows, o comando `python` e' interceptado por um
    alias da Microsoft Store (WindowsApps\python.exe) que vem antes do Python
    real no PATH. Este script localiza o interpretador de verdade e roda a
    partir da pasta `src`, sem depender de como o PATH esta' configurado.

    Reproduzir o artigo:
        .\rodar.ps1              (ajuda)
        .\rodar.ps1 ver          (versao do Python e pacotes)
        .\rodar.ps1 teste        (valida kernels vs implementacao de referencia)
        .\rodar.ps1 rep          (replicacao: Figs. 1-5)
        .\rodar.ps1 long         (series de 2000 geracoes)
        .\rodar.ps1 ext          (experimentos estendidos E1-E8)
        .\rodar.ps1 ops          (ablacao de operadores, E5)
        .\rodar.ps1 figs         (gera todas as figuras)
        .\rodar.ps1 tudo         (teste + rep + figs, ~1 min)
        .\rodar.ps1 completo     (tudo, ~30 min)

    Testar cenarios proprios (os parametros valem para tipo1/tipo2/comparar/animar):
        .\rodar.ps1 tipo1
        .\rodar.ps1 tipo1 -p 0.1 -mmax 8 -geracoes 400
        .\rodar.ps1 tipo2 -operador corte -repeticoes 8
        .\rodar.ps1 comparar -p 0.05 -geracoes 300
        .\rodar.ps1 animar -tipo II -geracoes 120

    Observacao: no PowerShell, "-p 0.1" e "--p 0.1" sao equivalentes.
#>
param(
    [Parameter(Position = 0)]
    [string]$Comando = 'ajuda',

    # parametros repassados aos scripts Python
    [ValidateSet('I', 'II', 'i', 'ii')]
    [string]$tipo = 'I',
    [double]$p = 0.05,
    [int]$geracoes = 300,
    [int]$passos = 10000,
    [int]$N = 101,
    [int]$S = 0,
    [int]$mmin = 1,
    [int]$mmax = 15,
    [ValidateSet('misto', 'corte', 'emenda', 'redesenho', 'crossover', 'mutacao')]
    [string]$operador = 'misto',
    [int]$repeticoes = 4,
    [int]$semente = 7,
    [int]$fps = 8,
    [string]$saida = '',
    [int]$ultimas = 30,

    # infraestrutura
    [int]$Workers = 8
)

$ErrorActionPreference = 'Stop'
$raiz = Split-Path -Parent $MyInvocation.MyCommand.Path
$src  = Join-Path $raiz 'src'

function Find-Python {
    $cache = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'),
        'C:\Python312\python.exe',
        'C:\Python311\python.exe'
    )
    foreach ($c in $cache) {
        if (Test-Path $c) { return $c }
    }
    $g = Get-Command python -CommandType Application -ErrorAction SilentlyContinue |
         Where-Object { $_.Source -notlike '*WindowsApps*' } |
         Select-Object -First 1
    if ($g) { return $g.Source }
    throw "Python 3.10+ nao encontrado. Instale de python.org ou edite a lista de caminhos em rodar.ps1."
}

function Invoke-NoSrc([string[]]$Argumentos) {
    Push-Location $src
    try { & $script:py @Argumentos; $code = $LASTEXITCODE }
    finally { Pop-Location }
    if ($code -ne 0) { throw "Comando terminou com codigo $code" }
}

function Args-Teste([string]$qual) {
    $a = @('--tipo', $qual,
           '--p', "$p",
           '--geracoes', "$geracoes",
           '--passos', "$passos",
           '--N', "$N",
           '--mmin', "$mmin",
           '--mmax', "$mmax",
           '--operador', $operador,
           '--repeticoes', "$repeticoes",
           '--ultimas', "$ultimas",
           '--workers', "$Workers")
    if ($S -gt 0) { $a += @('--S', "$S") }
    if ($saida -ne '') { $a += @('--saida', $saida) }
    return $a
}

function Args-Comparar {
    $a = @('--p', "$p",
           '--geracoes', "$geracoes",
           '--passos', "$passos",
           '--N', "$N",
           '--mmin', "$mmin",
           '--mmax', "$mmax",
           '--operador', $operador,
           '--repeticoes', "$repeticoes",
           '--ultimas', "$ultimas",
           '--workers', "$Workers")
    if ($saida -ne '') { $a += @('--saida', $saida) }
    return $a
}

function Args-Animar {
    $a = @('--tipo', $tipo,
           '--p', "$p",
           '--geracoes', "$geracoes",
           '--passos', "$passos",
           '--N', "$N",
           '--mmax', "$mmax",
           '--operador', $operador,
           '--semente', "$semente",
           '--fps', "$fps")
    if ($saida -ne '') { $a += @('--saida', $saida) }
    return $a
}

function Show-Ajuda {
    Write-Host ""
    Write-Host "  Trabalho: Jogo da Minoria Evolucionario (Yang et al., 2004)" -ForegroundColor Cyan
    Write-Host "  Pasta   : $raiz" -ForegroundColor DarkGray
    Write-Host "  Python  : $(Find-Python)" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  Reproduzir o artigo:" -ForegroundColor White
    Write-Host "    .\rodar.ps1 ver        versao e pacotes instalados"
    Write-Host "    .\rodar.ps1 teste      valida os kernels (~15 s)"
    Write-Host "    .\rodar.ps1 rep        replicacao dos resultados (~1 min)"
    Write-Host "    .\rodar.ps1 long       series de 2000 geracoes (~8 min)"
    Write-Host "    .\rodar.ps1 ext        experimentos estendidos E1-E8 (~20 min)"
    Write-Host "    .\rodar.ps1 ops        ablacao de operadores (~2 min)"
    Write-Host "    .\rodar.ps1 figs       gera as figuras a partir de results\"
    Write-Host "    .\rodar.ps1 tudo       teste + rep + figs"
    Write-Host ""
    Write-Host "  Testar por conta propria:" -ForegroundColor White
    Write-Host "    .\rodar.ps1 tipo1              exemplo Tipo-I"
    Write-Host "    .\rodar.ps1 tipo2              exemplo Tipo-II"
    Write-Host "    .\rodar.ps1 comparar           Tipo-I x Tipo-II lado a lado"
    Write-Host "    .\rodar.ps1 animar             GIF da evolucao"
    Write-Host ""
    Write-Host "  Parametros (valem para tipo1/tipo2/comparar/animar):" -ForegroundColor White
    Write-Host "    -p 0.1        fracao dos mais pobres que mudam de estrategia"
    Write-Host "    -geracoes 400 numero de geracoes"
    Write-Host "    -passos 10000 passos por geracao"
    Write-Host "    -N 201        numero de agentes"
    Write-Host "    -S 4          estrategias por agente"
    Write-Host "    -mmax 10      memoria maxima"
    Write-Host "    -operador corte|emenda|redesenho|crossover|mutacao|misto"
    Write-Host "    -repeticoes 8 numero de sementes"
    Write-Host "    -saida meu_teste  nome da pasta em results\"
    Write-Host ""
    Write-Host "  Exemplo:  .\rodar.ps1 tipo1 -p 0.1 -mmax 8 -geracoes 400" -ForegroundColor DarkGray
    Write-Host "  No cmd.exe, use:  rodar.cmd <comando>" -ForegroundColor DarkGray
    Write-Host "  Guia completo   :  GUIA.md" -ForegroundColor DarkGray
    Write-Host ""
}

$script:py = Find-Python

switch ($Comando.ToLower()) {
    'ajuda' { Show-Ajuda }

    'ver' {
        Write-Host "interpretador: $script:py" -ForegroundColor DarkGray
        & $script:py -V
        & $script:py -m pip --version
        & $script:py -c "import numpy, numba, matplotlib; print('numpy', numpy.__version__); print('numba', numba.__version__); print('matplotlib', matplotlib.__version__)"
    }

    'teste'    { Invoke-NoSrc @('test_kernels.py') }
    'rep'      { Invoke-NoSrc @('experiments.py', 'rep', '--workers', $Workers) }
    'long'     { Invoke-NoSrc @('experiments.py', 'long', '--workers', $Workers) }
    'ext'      { Invoke-NoSrc @('experiments.py', 'ext', '--workers', $Workers) }
    'ops'      { Invoke-NoSrc @('rerun_ops.py') }
    'figs'     { Invoke-NoSrc @('make_figs.py') }

    'tudo' {
        Invoke-NoSrc @('test_kernels.py')
        Invoke-NoSrc @('experiments.py', 'rep', '--workers', $Workers)
        Invoke-NoSrc @('make_figs.py')
    }

    'completo' {
        Invoke-NoSrc @('test_kernels.py')
        Invoke-NoSrc @('experiments.py', 'rep', '--workers', $Workers)
        Invoke-NoSrc @('experiments.py', 'long', '--workers', $Workers)
        Invoke-NoSrc @('experiments.py', 'ext', '--workers', $Workers)
        Invoke-NoSrc @('rerun_ops.py')
        Invoke-NoSrc @('make_figs.py')
    }

    'tipo1'    { Invoke-NoSrc (@('rodar_teste.py') + (Args-Teste 'I')) }
    'tipo2'    { Invoke-NoSrc (@('rodar_teste.py') + (Args-Teste 'II')) }
    'comparar' { Invoke-NoSrc (@('comparar_tipos.py') + (Args-Comparar)) }
    'animar'   { Invoke-NoSrc (@('animar.py') + (Args-Animar)) }

    default {
        Write-Host "Comando desconhecido: $Comando" -ForegroundColor Yellow
        Show-Ajuda
        exit 1
    }
}
