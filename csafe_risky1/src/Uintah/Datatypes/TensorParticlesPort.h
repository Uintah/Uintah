
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

#include <PSECore/Datatypes/SimplePort.h>
#include "TensorParticles.h"

namespace PSECore {
namespace Datatypes {

using namespace Uintah::Datatypes;

typedef SimpleIPort<TensorParticlesHandle> TensorParticlesIPort;
typedef SimpleOPort<TensorParticlesHandle> TensorParticlesOPort;

} // End namespace Datatypes
} // End namespace PSECore

#endif
