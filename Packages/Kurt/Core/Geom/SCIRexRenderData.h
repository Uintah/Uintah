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


#ifndef SCIREX_RENDER_DATA_H
#define SCIREX_RENDER_DATA_H

#include <sci_defs/ogl_defs.h>
#if defined(HAVE_GLEW)
#include <GL/glew.h>
#include <GL/glxew.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#endif


namespace SCIRun {
class Barrier;
class Mutex;
class DrawInfoOpenGL;
}

namespace Kurt {
using SCIRun::Barrier;
using SCIRun::Mutex;
using SCIRun::DrawInfoOpenGL;
class OGLXVisual;
struct SCIRexRenderData {

// SCIRexRenderData( unsigned char* wb,
// 		  unsigned char* db,
// 		  int x, int y,
// 		  OGLXVisual *v,
// 		  Barrier *b,
// 		  int waiters,
// 		  Mutex *m,
// 		  double *mv );

//   write_buffer_(wb), depth_buffer_(db), viewport_x_(x), viewport_y_(y),
//   visual_(v) barrier_(b), waiters_(waiters), mutex_(m), mvmat_(mv) {;}


  unsigned char* write_buffer_;
  unsigned char* depth_buffer_;
  int viewport_x_, viewport_y_;
  bool viewport_changed_;
  OGLXVisual *visual_;
  Barrier *barrier_;
  int waiters_;
  bool waiters_changed_;
  Mutex *mutex_;
  double *mvmat_;
  double *pmat_;
  DrawInfoOpenGL* di_;
  Material *mat_;
  double time_;
  int *comp_order_;
  int comp_count_;
  bool dump_;
  bool use_depth_;
  int curFrame_;
};
  
} // end namspace Kurt
#endif
