

#include <SCIRun/Core/Thread/Mutex.h>
#include <SCIRun/Core/Util/DebugStream.h>
#include <Core/Util/share.h>

using SCIRun::Mutex;
SCISHARE Mutex cerrLock( "cerr lock" );

SCISHARE SCIRun::DebugStream dbg_barrier("MPIBarriers",false);
