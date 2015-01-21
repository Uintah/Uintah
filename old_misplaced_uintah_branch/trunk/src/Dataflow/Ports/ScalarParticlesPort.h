/*
 *  ????
 *
 *  Written by:
 *   ???
 *   Department of Computer Science
 *   University of Utah
 *   March 199?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef SCI_project_ScalarParticlesPort_h
#define SCI_project_ScalarParticlesPort_h 1

#include <Core/Datatypes/ScalarParticles.h>
#include <SCIRun/Dataflow/Network/Ports/SimplePort.h>

namespace Uintah {

using namespace SCIRun;

typedef SimpleIPort<ScalarParticlesHandle> ScalarParticlesIPort;
typedef SimpleOPort<ScalarParticlesHandle> ScalarParticlesOPort;

} // End namespace Uintah

#endif
