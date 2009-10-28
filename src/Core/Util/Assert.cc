#include <Core/Util/Assert.h>
#include <cstdio>

bool wait_for_debugger=false;
void WAIT_FOR_DEBUGGER()
{ 
  if(!wait_for_debugger)
    return;

  bool wait=true; 
  char hostname[100]; 
  gethostname(hostname,100); 
  printf("%s:%d waiting for debugger\n",hostname,getpid()); 
  while(wait)
  {
    sleep(1);
  }; 
}
void TURN_ON_WAIT_FOR_DEBUGGER()
{
  wait_for_debugger=true;
}
void TURN_OFF_WAIT_FOR_DEBUGGER()
{
  wait_for_debugger=true;
}
