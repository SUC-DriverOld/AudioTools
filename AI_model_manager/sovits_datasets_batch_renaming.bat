chcp 65001
@echo off

title sovits数据集批量重命名 by 未鸟

echo=
echo ///////////////////////////////////////////////////////////////////////////////
echo sovits数据集批量重命名   by领航员未鸟
echo=
echo 请将这个批处理放入目标文件夹后再运行
echo=
echo 它将会自动重命名该文件夹下所有xwm和wav格式的文件为数字而不改变其排序
echo=
echo=如果不是在目标文件夹下
echo=
echo=请立即关闭，防止出现误修改!
echo=
echo ///////////////////////////////////////////////////////////////////////////////
echo=

pause

echo=
echo=

set a=1

setlocal EnableDelayedExpansion

for /f "delims=" %%i in ('dir /b *.*') do (

set "fn=%%~nxi" 

if "xwm"=="!fn:~-3!" (ren "%%i" "!a!.xwm" && echo "%%i->!a!.xwm" & echo "%%i->!a!.xwm">>文件对照表.txt  & set /A a+=1)

if "wav"=="!fn:~-3!" (ren "%%i" "!a!.wav" && echo "%%i->!a!.wav" & echo "%%i->!a!.wav">>文件对照表.txt & set /A a+=1) 

)

echo=
echo=
echo Done^^!

set /p u=
