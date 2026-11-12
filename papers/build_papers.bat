@echo off
rem Build both papers with MiKTeX (run from repo root: papers\build_papers.bat)
setlocal
set "REPO=%~dp0"
cd /d "%REPO%paper1_algo_comparison" && pdflatex -interaction=nonstopmode main.tex > build1.log 2>&1 && pdflatex -interaction=nonstopmode main.tex > build1b.log 2>&1
cd /d "%REPO%paper2_reward_design" && pdflatex -interaction=nonstopmode main.tex > build2.log 2>&1 && pdflatex -interaction=nonstopmode main.tex > build2b.log 2>&1
echo paper1: && dir /b "%REPO%paper1_algo_comparison\main.pdf" 2>nul
echo paper2: && dir /b "%REPO%paper2_reward_design\main.pdf" 2>nul
endlocal
