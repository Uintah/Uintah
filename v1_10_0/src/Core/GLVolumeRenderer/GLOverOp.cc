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

#include <Core/GLVolumeRenderer/GLOverOp.h>
#include <GL/gl.h>

namespace SCIRun {


GLOverOp::GLOverOp(const GLVolumeRenderer* glvr) :
  GLTexRenState( glvr )
{
}

void GLOverOp::preDraw()
{
  glEnable(GL_BLEND);
  glBlendEquation(GL_FUNC_ADD_EXT);
//    glBlendFunc( GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
  glBlendFunc( GL_ONE,GL_ONE_MINUS_SRC_ALPHA);
}

void GLOverOp::postDraw()
{
  glDisable(GL_BLEND);
}


} // End namespace SCIRun
