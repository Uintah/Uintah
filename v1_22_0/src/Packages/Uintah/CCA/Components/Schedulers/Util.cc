

#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>

using SCIRun::Mutex;
using SCIRun::DebugStream;
// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)
Mutex cerrLock( "cerr lock" );
DebugStream mixedDebug( "MixedScheduler", false );
DebugStream fullDebug( "MixedSchedulerFull", false );

