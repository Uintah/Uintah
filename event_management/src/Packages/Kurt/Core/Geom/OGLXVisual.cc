#include "OGLXVisual.h"
#include <iostream>
using std::cerr;

namespace Kurt {

int OGLXVisual::SB_VISUAL[] = {
  GLX_DEPTH_SIZE,     16,
  GLX_ALPHA_SIZE,      8,
  GLX_RED_SIZE,        8,                   
  GLX_GREEN_SIZE,      8,
  GLX_BLUE_SIZE,       8,
  None
};
int OGLXVisual::DB_VISUAL[] = {
  GLX_DEPTH_SIZE,      8,
  GLX_ALPHA_SIZE,      8,
  GLX_RED_SIZE,        8,                   
  GLX_GREEN_SIZE,      8,
  GLX_BLUE_SIZE,       8,
  GLX_DOUBLEBUFFER,    True,
  None
};
int OGLXVisual::PB_VISUAL[] = {
  GLX_RENDER_TYPE,     GLX_RGBA_BIT,
  GLX_DEPTH_SIZE,      8,
  GLX_RED_SIZE,        8,
  GLX_GREEN_SIZE,      8,
  GLX_BLUE_SIZE,       8,
  GLX_DRAWABLE_TYPE,   GLX_PBUFFER_BIT,
  None
};
int OGLXVisual::PM_VISUAL[] = {
  GLX_RENDER_TYPE,     GLX_RGBA_BIT,
  GLX_DEPTH_SIZE,      8,
  GLX_RED_SIZE,        8,
  GLX_GREEN_SIZE,      8,
  GLX_BLUE_SIZE,       8,
  GLX_DRAWABLE_TYPE,   GLX_PIXMAP_BIT,
  None
};
int OGLXVisual::ST_VISUAL[] = {
  GLX_DEPTH_SIZE,     16,
  GLX_ALPHA_SIZE,      8,
  GLX_RED_SIZE,        8,                   
  GLX_GREEN_SIZE,      8,
  GLX_BLUE_SIZE,       8,
  GLX_STEREO,          True,
  GLX_DOUBLEBUFFER,    True,
  None
};


OGLXVisual::OGLXVisual(int *att) : _att(att) {}

OGLXVisual::OGLXVisual(Type vt){
  switch(vt) {
  case RGBA_SB_VISUAL:
    _att = SB_VISUAL;
    break;
  case RGBA_DB_VISUAL:
    _att = DB_VISUAL;
    break;
  case RGBA_PB_VISUAL:
    _att = PB_VISUAL;
    break;
  case RGBA_PM_VISUAL:
    _att = PM_VISUAL;
    break;
  case RGBA_ST_VISUAL:
    _att = ST_VISUAL;
    break;      
  default:
    cerr << "OGLXVisual::OGLXVisual() : "
	 << "Warning unknown visual type specified.\n";
    _att = 0;
    break;
  }
  
}

} // end namespace Kurt
