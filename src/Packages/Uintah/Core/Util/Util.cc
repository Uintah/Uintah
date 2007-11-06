

#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>
#include <Packages/Uintah/Core/Util/uintahshare.h>

using SCIRun::Mutex;
UINTAHSHARE Mutex cerrLock( "cerr lock" );

UINTAHSHARE SCIRun::DebugStream dbg_barrier("MPIBarriers",false);
