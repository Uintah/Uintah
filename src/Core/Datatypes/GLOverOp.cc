#include <Core/Datatypes/GLOverOp.h>
#include <GL/gl.h>

namespace SCIRun {


GLOverOp::GLOverOp(const GLVolumeRenderer* glvr) :
  GLTexRenState( glvr )
{
}

void GLOverOp::preDraw()
{
  glEnable(GL_BLEND);
#ifdef __sgi
  glBlendEquation(GL_FUNC_ADD_EXT);
#endif
  glBlendFunc( GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
}

void GLOverOp::postDraw()
{
  glDisable(GL_BLEND);
}


} // End namespace SCIRun
