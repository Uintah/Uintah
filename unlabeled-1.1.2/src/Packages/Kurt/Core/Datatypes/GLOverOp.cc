#include "GLOverOp.h"
#include <GL/gl.h>

namespace SCICore {
namespace GeomSpace  {

GLTexRenState* GLOverOp::_instance = 0;

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

GLTexRenState* GLOverOp::Instance(const GLVolumeRenderer* glvr)
{
  // Not a true Singleton class, but this does make sure that 
  // there is only one instance per volume renderer.
  if( _instance == 0 ){
    _instance = new GLOverOp( glvr );
  }
  
  return _instance;
}

} // end namespace Datatypes
} // end namespace Kurt
