#include "GLOverOp.h"
#include <GL/gl.h>

namespace SCICore {
namespace GeomSpace  {


GLOverOp::GLOverOp(const GLVolumeRenderer* glvr) :
  GLTexRenState( glvr )
{
}

void GLOverOp::preDraw()
{
  glEnable(GL_BLEND);
  glBlendEquation(GL_FUNC_ADD_EXT);
  glBlendFunc( GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
}

void GLOverOp::postDraw()
{
  glDisable(GL_BLEND);
}


} // end namespace Datatypes
} // end namespace Kurt
