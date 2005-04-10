#include "GLMIP.h"
#include <GL/gl.h>

namespace Kurt{
namespace GeomSpace  {



GLMIP::GLMIP(const GLVolumeRenderer* glvr) :
  GLTexRenState( glvr )
{
}

void GLMIP::preDraw()
{
  glEnable(GL_BLEND);
  glBlendEquationEXT(GL_MAX_EXT);
  glBlendFunc(GL_ONE, GL_ONE);
}

void GLMIP::postDraw()
{
  glDisable(GL_BLEND);
}


} // end namespace GeomSpace
} // end namespace Kurt
