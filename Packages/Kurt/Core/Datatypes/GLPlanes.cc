#include "GLPlanes.h"
#include <GL/gl.h>

namespace SCICore {
namespace GeomSpace  {


GLPlanes::GLPlanes(const GLVolumeRenderer* glvr) :
  GLTexRenState( glvr )
{
}

void GLPlanes::preDraw()
{
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  //glBlendEquation(GL_FUNC_ADD_EXT);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);
  //  glBlendColorEXT(1.f, 1.f, 1.f, 1.f/volren->slices);

}

void GLPlanes::postDraw()
{
  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);

}

} // end namespace Datatypes
} // end namespace Kurt
