
/*
 *  Texture3D.h: The Scalar Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VisParticleSetPort_h
#define SCI_project_VisParticleSetPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include "VisParticleSet.h"

namespace Kurt {
using namespace Kurt::Datatypes;

typedef SimpleIPort<VisParticleSetHandle> VisParticleSetIPort;
typedef SimpleOPort<VisParticleSetHandle> VisParticleSetOPort;
} // End namespace Kurt


#endif
