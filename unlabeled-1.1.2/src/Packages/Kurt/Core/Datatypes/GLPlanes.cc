#include "GLPlanes.h"
#include <GL/gl.h>

namespace SCICore {
namespace GeomSpace  {

GLTexRenState* GLPlanes::_instance = 0;

GLPlanes::GLPlanes(const GLVolumeRenderer* glvr) :
  GLTexRenState( glvr )
{
}

void GLPlanes::preDraw()
{
  glEnable(GL_DEPTH_TEST);
}

void GLPlanes::postDraw()
{
  glDisable(GL_DEPTH_TEST);
}

GLTexRenState* GLPlanes::Instance(const GLVolumeRenderer* glvr)
{
  // Not a true Singleton class, but this does make sure that 
  // there is only one instance per volume renderer.
  if( _instance == 0 ){
    _instance = new GLPlanes( glvr );
  }
  
  return _instance;
}

} // end namespace Datatypes
} // end namespace Kurt
