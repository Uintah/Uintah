import os
import pysci
import threading
import time

class mythread(threading.Thread) :
    def __init__(self, env) :
        threading.Thread.__init__(self)
        self.env = env
        
    def run(self) :
        pysci.start_scirun_threads(1, [], self.env)
        

#os.environ['SCIRUN_EXECUTE_ON_STARTUP'] = '1'
env = []
for k in os.environ.keys() :
    estr = "%s=%s" % (k, os.environ[k])
    env.append(estr)



net = pysci.start_scirun_threads(1, [], env)

time.sleep(5)

fname = '/scratch/mjc/tst.srn'
nio = pysci.NetworkIO()
nio.load_net(fname)
nio.load_network()


time.sleep(5)

# execute
net.schedule_all()
net.nmodules()


## print "about to make thread"
## t = mythread(env)
## t.start()

## while t.isAlive() :
##     time.sleep(1)
##     print "still alive"

print "do i ever get back?"
time.sleep(5)
