/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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
#include <Core/GLVolumeRenderer/GLTexture3D.h>

namespace SCIRun {


typedef SimpleIPort<GLTexture3DHandle> GLTexture3DIPort;
typedef SimpleOPort<GLTexture3DHandle> GLTexture3DOPort;

} // End namespace SCIRun

#endif
