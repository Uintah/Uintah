#include <SCICore/Datatypes/GLOverOp.h>
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
#ifdef __sgi
  glBlendEquation(GL_FUNC_ADD_EXT);
#endif
  glBlendFunc( GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
}

void GLOverOp::postDraw()
{
  glDisable(GL_BLEND);
}


} // end namespace Datatypes
} // end namespace SCICore
