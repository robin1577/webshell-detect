import subprocess
path=r"D:\webshell-detect\test\hello.php"
cmd="php"+" -dvld.active=1 -dvld.execute=0 "+path
status,output=subprocess.getstatusoutput(cmd)
print(status)
print(output)