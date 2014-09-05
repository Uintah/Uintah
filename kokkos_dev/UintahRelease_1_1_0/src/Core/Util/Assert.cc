#include <Core/Util/Assert.h>
#include <cstdio>

void WAIT_FOR_DEBUGGER()
{ 
  bool wait=true; 
  char hostname[100]; 
  gethostname(hostname,100); 
  printf("%s:%d waiting for debugger\n",hostname,getpid()); 
  while(wait)
  {
    sleep(1);
  }; 
}
