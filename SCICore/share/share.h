/* share.h written by Chris Moulding 11/98 */

/*
 SMARTSHARE (applies to win32 only)
 
 if == 0 will allways SHARE as dllexport regardless of whether importing or 
         exporting. => safe

 if == 1 will SHARE as dllexport when exporting and as dllimport when
         importing. => efficient

*/

#ifndef SMARTSHARE
#define SMARTSHARE 0
#endif

#undef SHARE

#ifdef _WIN32
#if (defined(IMPORTING) && (SMARTSHARE))
#define SHARE __declspec(dllimport)
#else
#define SHARE __declspec(dllexport)
#endif /* (defined(IMPORTING) && (SMARTSHARE)) */
#else 
#define SHARE 
#endif /* _WIN32 */
