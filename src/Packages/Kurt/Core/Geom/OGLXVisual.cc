/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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
