/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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


