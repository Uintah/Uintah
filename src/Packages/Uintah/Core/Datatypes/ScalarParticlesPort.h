
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

#ifndef SCI_project_ScalarParticlesPort_h
#define SCI_project_ScalarParticlesPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include "ScalarParticles.h"

namespace Uintah {
using namespace Uintah::Datatypes;

typedef SimpleIPort<ScalarParticlesHandle> ScalarParticlesIPort;
typedef SimpleOPort<ScalarParticlesHandle> ScalarParticlesOPort;
} // End namespace Uintah


#endif
