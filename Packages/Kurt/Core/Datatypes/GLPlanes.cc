#include "GLPlanes.h"
#include <GL/gl.h>

namespace Kurt {

GLPlanes::GLPlanes(const GLVolumeRenderer* glvr) :
  GLTexRenState( glvr )
{
}

void GLPlanes::preDraw()
{
  //  glBlendColorEXT(1.f, 1.f, 1.f, 1.f/volren->slices);
  glEnable(GL_ALPHA_TEST);
  glAlphaFunc(GL_GREATER, 0.0);
}

void GLPlanes::postDraw()
{
  glDisable(GL_ALPHA_TEST);
}
} // End namespace Kurt

