/*
 *  Texture3D.h: The Tensor Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_TensorParticlesPort_h
#define SCI_project_TensorParticlesPort_h 1

#include <Packages/Uintah/Core/Datatypes/TensorParticles.h>
#include <Dataflow/Ports/SimplePort.h>

namespace Uintah {

typedef SimpleIPort<TensorParticlesHandle> TensorParticlesIPort;
typedef SimpleOPort<TensorParticlesHandle> TensorParticlesOPort;

} // End namespace Uintah


#endif
