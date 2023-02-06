from os import system

str1 = "python drivers/run_opt.py --agent 'Torch_wcpg-v1' --env 'Hopper-v3' --trial 0"
str2 = "python drivers/run_opt.py --agent 'Torch_wcpg-v1' --env 'Hopper-v3' --trial 1"
str3 = "python drivers/run_opt.py --agent 'Torch_wcpg-v1' --env 'Hopper-v3' --trial 2"

str4 = "python drivers/run_opt.py --agent 'Torch_wcpg-v1' --env 'Walker2d-v3' --trial 0"
str5 = "python drivers/run_opt.py --agent 'Torch_wcpg-v1' --env 'Walker2d-v3' --trial 1"
str6 = "python drivers/run_opt.py --agent 'Torch_wcpg-v1' --env 'Walker2d-v3' --trial 2"

system(str1)
system(str2)
system(str3)

system(str4)
system(str5)
system(str6)