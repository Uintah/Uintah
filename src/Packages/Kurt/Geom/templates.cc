/*
 * Manual template instantiations for g++
 */

/*
 * These aren't used by Datatypes directly, but since they are used in
 * a lot of different modules, we instantiate them here to avoid bloat
 */

#include <Persistent/Persistent.h>
#include <Containers/LockingHandle.h>

#include <Datatypes/Particles/ParticleGridReader.h>

using namespace SCICore::PersistentSpace;
using namespace SCICore::Containers;
using namespace Uintah::Datatypes;

template void Pio<>(Piostream&, LockingHandle<ParticleGridReader>&);

