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

#include <Core/GLVolumeRenderer/GLPlanes.h>
#include <GL/gl.h>

namespace SCIRun {


GLPlanes::GLPlanes(const GLVolumeRenderer* glvr) :
  GLTexRenState( glvr )
{
}

void GLPlanes::preDraw()
{
  //  glBlendColorEXT(1.f, 1.f, 1.f, 1.f/volren->slices);
  glDepthMask(GL_TRUE);
  glEnable(GL_ALPHA_TEST);
  glAlphaFunc(GL_GREATER, 0.0);
}

void GLPlanes::postDraw()
{
  glDepthMask(GL_FALSE);
  glDisable(GL_ALPHA_TEST);
}

} // End namespace SCIRun
