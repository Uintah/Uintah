//#include "GLAttenuate.h"
#include "GLVolumeRenderer.h"
#include <GL/gl.h>

namespace SCICore {
namespace GeomSpace  {

using namespace Kurt::Datatypes;

GLTexRenState* GLAttenuate::_instance = 0;

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

GLTexRenState* GLAttenuate::Instance(const GLVolumeRenderer* glvr)
{
  // Not a true Singleton class, but this does make sure that 
  // there is only one instance per volume renderer.
  if( _instance == 0 ){
    _instance = new GLAttenuate( glvr );
  } 

  return _instance;
}

} // end namespace Datatypes
} // end namespace Kurt
