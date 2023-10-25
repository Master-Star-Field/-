import time

from show_mat import*
from alex_fun import*

start_time = time.time()
Annenkov_time = show_mat("clown.mat")
print("--- %s Annenkov ---" % (time.time() - start_time))
start_time = time.time()
fun_alex("clown.mat")
print("--- %s Alexandr ---" % (time.time() - start_time))

