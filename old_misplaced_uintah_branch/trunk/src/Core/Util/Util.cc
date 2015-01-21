

#include <SCIRun/Core/Thread/Mutex.h>
#include <SCIRun/Core/Util/DebugStream.h>
#include <Core/Util/uintahshare.h>

using SCIRun::Mutex;
UINTAHSHARE Mutex cerrLock( "cerr lock" );

UINTAHSHARE SCIRun::DebugStream dbg_barrier("MPIBarriers",false);
