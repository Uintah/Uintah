#include <Core/Datatypes/GLMIP.h>
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
