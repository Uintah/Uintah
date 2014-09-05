

#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>
#include <Packages/Uintah/Core/Util/share.h>

using SCIRun::Mutex;
SCISHARE Mutex cerrLock( "cerr lock" );

SCISHARE SCIRun::DebugStream dbg_barrier("MPIBarriers",false);
