
/*
 *  Texture3D.h: The Vector Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VectorParticlesPort_h
#define SCI_project_VectorParticlesPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include "VectorParticles.h"

namespace PSECore {
namespace Datatypes {

using namespace Uintah::Datatypes;

typedef SimpleIPort<VectorParticlesHandle> VectorParticlesIPort;
typedef SimpleOPort<VectorParticlesHandle> VectorParticlesOPort;

} // End namespace Datatypes
} // End namespace PSECore

#endif
