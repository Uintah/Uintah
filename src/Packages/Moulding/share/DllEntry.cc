/* DllEntry.cc */

#ifdef _WIN32

#include <afxwin.h>
#include <stdio.h>

BOOL APIENTRY DllMain(HANDLE hModule, 
                      DWORD  ul_reason_for_call, 
                      LPVOID lpReserved)
{
#ifdef DEBUG
  char reason[100]="\0";
  printf("\n*** %sd.dll is initializing {%s,%d} ***\n",__FILE__,__LINE__);
  printf("*** hModule = %d ***\n",hModule);
  switch (ul_reason_for_call){
    case DLL_PROCESS_ATTACH:sprintf(reason,"DLL_PROCESS_ATTACH"); break;
    case DLL_THREAD_ATTACH:sprintf(reason,"DLL_THREAD_ATTACH"); break;
    case DLL_THREAD_DETACH:sprintf(reason,"DLL_THREAD_DETACH"); break;
    case DLL_PROCESS_DETACH:sprintf(reason,"DLL_PROCESS_DETACH"); break;
  }
  printf("*** ul_reason_for_call = %s ***\n",reason);
#endif
  return TRUE;
}

#endif


