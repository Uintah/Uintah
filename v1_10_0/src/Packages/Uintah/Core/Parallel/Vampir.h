

#ifdef USE_VAMPIR

extern "C" {
#include <VT.h>
extern void VTsetup();
}
#else
#define VT_end(x)
#define VT_begin(x)
#endif

#define VT_SEND_PARTICLES 100
#define VT_RECV_PARTICLES 101
#define VT_SEND_INITDATA 110
#define VT_RECV_INITDATA 111
#define VT_CHECKSUM 120
#define VT_SEND_COMPUTES 130
#define VT_RECV_DEPENDENCIES 131
#define VT_PERFORM_TASK 201
#define VT_EXECUTE 200

