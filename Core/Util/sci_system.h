#ifndef SCIRun_Core_Util_sci_system
#define SCIRun_Core_Util_sci_system 1

#ifdef __sgi
#define sci_system system
#include <stdlib.h>
#else

#ifdef __cplusplus
extern "C" {
#endif
int sci_system (const char * string);
#ifdef __cplusplus
}
#endif

#endif // __sgi

#endif // SCIRun_Core_Util_sci_system

