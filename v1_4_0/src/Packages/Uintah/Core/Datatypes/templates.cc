/*
 * Manual template instantiations for g++
 */

/*
 * These aren't used by Datatypes directly, but since they are used in
 * a lot of different modules, we instantiate them here to avoid bloat
 */

#include <Persistent/Persistent.h>
#include <Containers/LockingHandle.h>

#include <Packages/Uintah/Datatypes/Particles/ParticleGridReader.h>

using namespace SCIRun;
using namespace Uintah;

template void Pio<>(Piostream&, LockingHandle<ParticleGridReader>&);

