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

#include <Core/GLVolumeRenderer/GLMIP.h>
#include <GL/gl.h>

namespace SCIRun {



GLMIP::GLMIP(const GLVolumeRenderer* glvr) :
  GLTexRenState( glvr )
{
}

void GLMIP::preDraw()
{
  glEnable(GL_BLEND);
  glBlendEquation(GL_MAX_EXT);
  glBlendFunc(GL_ONE, GL_ONE);
}

void GLMIP::postDraw()
{
  glDisable(GL_BLEND);
}


} // End namespace SCIRun
