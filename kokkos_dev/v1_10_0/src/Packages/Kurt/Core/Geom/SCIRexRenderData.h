#ifndef SCIREX_RENDER_DATA_H
#define SCIREX_RENDER_DATA_H

#include <GL/gl.h>

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
