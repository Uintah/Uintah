//#include "GLAttenuate.h"
#include "GLVolumeRenderer.h"
#include <GL/gl.h>

namespace SCICore {
namespace GeomSpace  {

using namespace Kurt::Datatypes;


GLAttenuate::GLAttenuate(const GLVolumeRenderer* glvr) :
  GLTexRenState( glvr )
{
}

void GLAttenuate::preDraw()
{
  glEnable(GL_BLEND);
  glBlendEquation(GL_FUNC_ADD_EXT);
  glBlendFunc(GL_CONSTANT_ALPHA_EXT, GL_ONE);
  glBlendColorEXT(1.f, 1.f, 1.f, 1.f/volren->slices);
}

void GLAttenuate::postDraw()
{
  glDisable(GL_BLEND);
}

} // end namespace Datatypes
} // end namespace Kurt
