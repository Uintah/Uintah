/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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


