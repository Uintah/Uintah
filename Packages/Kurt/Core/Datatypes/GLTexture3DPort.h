
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

#ifndef SCI_project_GLTexture3DPort_h
#define SCI_project_GLTexture3DPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Packages/Kurt/Core/Datatypes/GLTexture3D.h>

namespace Kurt {

typedef SimpleIPort<GLTexture3DHandle> GLTexture3DIPort;
typedef SimpleOPort<GLTexture3DHandle> GLTexture3DOPort;

} // End namespace Kurt

#endif
