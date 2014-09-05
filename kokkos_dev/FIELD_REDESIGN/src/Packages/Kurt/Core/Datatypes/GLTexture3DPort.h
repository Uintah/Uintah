
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

#include <PSECore/Datatypes/SimplePort.h>
#include <Kurt/Datatypes/GLTexture3D.h>

namespace PSECore {
namespace Datatypes {

using namespace Kurt::Datatypes;

typedef SimpleIPort<GLTexture3DHandle> GLTexture3DIPort;
typedef SimpleOPort<GLTexture3DHandle> GLTexture3DOPort;

} // End namespace Datatypes
} // End namespace PSECore

#endif
