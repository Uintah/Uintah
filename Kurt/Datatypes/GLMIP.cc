#include "GLMIP.h"
#include <GL/gl.h>

namespace SCICore {
namespace GeomSpace  {


GLTexRenState* GLMIP::_instance = 0;

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

GLTexRenState* GLMIP::Instance(const GLVolumeRenderer* glvr)
{
  // Not a true Singleton class, but this does make sure that 
  // there is only one instance per volume renderer.
  if( _instance == 0 ){
    _instance = new GLMIP( glvr );
  } 
  
  return _instance;
}

} // end namespace Datatypes
} // end namespace Kurt
