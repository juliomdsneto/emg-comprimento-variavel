@echo off
rem Atalho para rodar o codigo a partir do cmd.exe.
rem Ex.:  rodar.cmd teste
rem       rodar.cmd figs
rem       rodar.cmd rep -Workers 10
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0rodar.ps1" %*
