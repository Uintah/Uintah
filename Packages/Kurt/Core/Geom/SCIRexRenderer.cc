/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#include <Packages/Kurt/Core/Geom/SCIRexRenderer.h>
#include <Packages/Kurt/Core/Geom/OGLXVisual.h>
#include <Packages/Kurt/Core/Geom/SCIRexCompositer.h>
#include <Packages/Kurt/Core/Geom/SCIRexRenderData.h>
#include <Packages/Kurt/Core/Geom/SCIRexWindow.h>
#include <Core/GLVolumeRenderer/GLVolumeRenderer.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Datatypes/Field.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Mutex.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Geom/Color.h>
#include <Core/Containers/Array1.h>
#include <Core/Thread/Thread.h>
#include <GL/gl.h>
#include <iostream>
#include <sstream>
using std::cerr;
using std::ostringstream;

namespace Kurt {

using SCIRun::Color;
using SCIRun::Array1;
using SCIRun::FieldHandle;
using SCIRun::Thread;
using SCIRun::GLVolumeRenderer;



SCIRexRenderer::SCIRexRenderer(int id) 
  : GeomObj( id ),
    rs_(SCIRexRenderer::OVEROP),
    tex_(0),
    lighting_(0),
    slices_(0),
    mutex_("SCIRexRenderer Mutex"),
    win_mutex_("SCIRexWindow Mutex"),
    cmap_(0),
    brick_size_(128),
    slice_alpha_(1.0),
    windows_init_(false),
    cmapHasChanged_(true),
    render_data_(0)
{
  NOT_FINISHED("SCIRexRenderer::SCIRexRenderer(int id )");
}


SCIRexRenderer::SCIRexRenderer(int id, vector<char *>& displays,
			       int ncompositers, FieldHandle tex,
			       ColorMapHandle map, bool isfixed,
			       double min, double max,
			       GLTexture3DHandle texH)
  : GeomObj( id ),
    rs_(SCIRexRenderer::OVEROP),
    tex_(tex),
    lighting_(0),
    slices_(0),
    mutex_("SCIRexRenderer Mutex"),
    win_mutex_("SCIRexWindow Mutex"),
    cmap_(map),
    min_(min),
    max_(max),
    is_fixed_(isfixed),
    brick_size_(128),
    slice_alpha_(0.5),
    windows_init_(false),
    cmapHasChanged_(true),
    render_data_(0)
{
//   mutex_.lock();
//   NOT_FINISHED("SCIRexRenderer::SCIRexRenderer(int id, GLTexture3DHandle tex, ColorMapHandle map)");
//   mutex_.unlock();

  render_data_ = new SCIRexRenderData(); 
  render_data_->write_buffer_ = 0;
  render_data_->depth_buffer_ = 0;
  render_data_->viewport_x_ = 0;
  render_data_->viewport_y_ = 0;
  render_data_->visual_ =  new OGLXVisual(OGLXVisual::RGBA_SB_VISUAL);
  render_data_->barrier_ = new Barrier("SCIRexRenderer");
  render_data_->waiters_ = displays.size() + ncompositers + 1;
  render_data_->mutex_ = &win_mutex_;
  render_data_->mvmat_ = new double[16];
  render_data_->pmat_ = new double[16];

  // Here is where we need to split up the volume.
  GLTexture3D *texture = scinew GLTexture3D(tex_, min_, max_, is_fixed_,
					    brick_size_);
  if (!is_fixed_) { // if not fixed, overwrite min/max values on Gui
    texture->getminmax(min_, max_);
  }
  textures.push_back( texture );
    
  int i;
  for(i = 0; i < int(displays.size()); i++){
    ostringstream win;
    win << "Win"<<i;
    char *w = new char[win.str().length()];
    win.str().copy(w, win.str().length());
    windows.push_back( new SCIRexWindow(w,displays[i], render_data_));
    GLVolumeRenderer *vr;
    vr = new GLVolumeRenderer( 0x12345676, texture, cmap_);
    vr->DrawFullRes();
    vr->SetNSlices( slices_ );
    vr->SetSliceAlpha( slice_alpha_ );
    windows[i]->addGeom( vr );
    renderers.push_back(vr);
    
  }
  
  for(i = 0; i < ncompositers; i++){
//     ostringstream comp;
//     comp << "Compositer"<<i;
//     char *c = new char(comp.str().length()];
//     comp.str().copy(c, comp.str().length());
    compositers.push_back(new SCIRexCompositer( render_data_ ));
    for(int j = 0; j < int( windows.size()); j++) {
      compositers[i]->add( windows[i] );
    }
    //set up compositers
  }
}

SCIRexRenderer::SCIRexRenderer(const SCIRexRenderer& copy)
  : GeomObj( copy.id ),
    rs_(copy.rs_),
    tex_(copy.tex_),
    lighting_(copy.lighting_),
    slices_(copy.slices_),
    mutex_("SCIRexRenderer Mutex"),
    win_mutex_("SCIRexWindow Mutex"),
    cmap_(copy.cmap_),
    brick_size_(copy.brick_size_),
    slice_alpha_(copy.slice_alpha_),
    windows_init_(copy.windows_init_),
    cmapHasChanged_(copy.cmapHasChanged_)
{
  mutex_.lock();
NOT_FINISHED("SCIRexRenderer::SCIRexRenderer(const SCIRexRenderer& copy");
  mutex_.unlock();
} 

SCIRexRenderer::~SCIRexRenderer()
{
  vector<SCIRexWindow *>::iterator it = windows.begin();
  for(; it != windows.end(); it++)
    (*it)->kill();
 
  vector<SCIRexCompositer *>::iterator cit = compositers.begin();
  for(; cit != compositers.end(); cit++)
    (*cit)->kill();

  render_data_->barrier_->wait(render_data_->waiters_);
}

GeomObj* 
SCIRexRenderer::clone()
{
  return scinew SCIRexRenderer( *this );
}

void SCIRexRenderer::get_bounds(BBox& bb)
{
  std::vector<GLTexture3D *>::iterator it = textures.begin();
  for(; it != textures.end(); it++)
    (*it)->get_bounds(bb);
}

#ifdef SCI_OPENGL
void 
SCIRexRenderer::draw(DrawInfoOpenGL* di, Material* mat, double time)
{
    //AuditAllocator(default_allocator);
  if( !pre_draw(di, mat, lighting_) ) return;
  mutex_.lock();
  render_data_->di_ = di;
  render_data_->mat_ = mat; 
  render_data_->time_ = time;

  if( di->get_drawtype() == DrawInfoOpenGL::WireFrame ){
    drawWireFrame();
  } else {
    //AuditAllocator(default_allocator);
    setup();
    preDraw();
    draw();
    postDraw();
    cleanup();
  }
  //  render_data_->di_ = 0;
  mutex_.unlock();

}
#endif

void 
SCIRexRenderer::draw()
{
  if(!windows_init_){
    Thread *th;
    int i;
    for(i = 0; i < int(windows.size()); i++){
      ostringstream win;
      win << "Window-"<<i;
      th = new Thread(windows[i], win.str().c_str());
      th->detach();
    }
    
    for(i = 0; i < int(compositers.size()); i++){
      ostringstream comp;
      comp << "Compositer-"<<i;
      th = new Thread(compositers[i], comp.str().c_str());
      th->detach();
    }
    windows_init_ = true;
  }
  Barrier *barrier = render_data_->barrier_;
  int waiters = render_data_->waiters_;

  win_mutex_.lock(); 
  cerr<<"check for Exit in renderer thread"<<endl;
  win_mutex_.unlock();
  barrier->wait(waiters);

  win_mutex_.lock(); 
  cerr<<"update info for in renderer thread"<<endl;
  win_mutex_.unlock();
  barrier->wait(waiters);

  // while rendering make sure the compositers are updated
  update_compositer_data();

  win_mutex_.lock(); 
  cerr<<"render windows in renderer thread"<<endl;
  win_mutex_.unlock();
  barrier->wait(waiters);

  win_mutex_.lock(); 
  cerr<<"wait on compositers in renderer thread"<<endl;
  win_mutex_.unlock();
  barrier->wait(waiters);

  win_mutex_.lock(); 
  cerr<<"wait on Display in renderer thread"<<endl;
  win_mutex_.unlock();
  
  glDrawBuffer(GL_BACK);
  glEnable(GL_BLEND);
//   glBlendEquation(GL_FUNC_ADD_EXT);
  glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_COLOR);
//  glBlendFunc( GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
//  glBlendFunc( GL_ONE,GL_ONE_MINUS_SRC_ALPHA);

  glDrawPixels(render_data_->viewport_x_,
	       render_data_->viewport_y_,
	       GL_RGBA, GL_UNSIGNED_BYTE, render_data_->write_buffer_);
  glDisable(GL_BLEND);
  barrier->wait(waiters);
  

}

void
SCIRexRenderer::SetRenderStyle(renderStyle rs)
{
  vector<GLVolumeRenderer *>::iterator it  = renderers.begin();
  for(; it != renderers.end(); it++)
    switch(rs) {
    case OVEROP:
      (*it)->set_tex_ren_state(GLVolumeRenderer::TRS_GLOverOp);
      break;
    case MIP:
      (*it)->set_tex_ren_state(GLVolumeRenderer::TRS_GLMIP);
      break;
    case ATTENUATE:
      (*it)->set_tex_ren_state(GLVolumeRenderer::TRS_GLAttenuate);
      break;
    }
}
  
void
SCIRexRenderer::update_compositer_data()
{
  cerr<<"in pdate_compositer_data()\n";
  if(render_data_->viewport_changed_){
    double ncompositers = (double)compositers.size();
    vector<SCIRexCompositer *>::iterator it = compositers.begin();
    cerr<<"viewport = "<<render_data_->viewport_y_<<" by "<<
      render_data_->viewport_x_<<"\n";
    for(int i = 0; it != compositers.end(); it++, i++){
      int ymin = render_data_->viewport_y_ * ( i/ncompositers);
      int ymax = render_data_->viewport_y_ * ( (i+1)/ncompositers);
      cerr<<"ymin, ymax = "<<ymin<<", "<<ymax<<"\n";
      (*it)->SetFrame(0,ymin,render_data_->viewport_x_, ymax);
      (*it)->SetOffset(4*ymin*render_data_->viewport_x_);
    }
  }
  //SET ORDER SOMEWHERE HERE
}

void
SCIRexRenderer::setup()
{


//   glEnable(GL_TEXTURE_3D_EXT);
//   glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_MODULATE); 

//   if( cmap_.get_rep() ) {
// #ifdef GL_TEXTURE_COLOR_TABLE_SGI
//     //cerr << "Using Lookup!\n";
//     glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
// #elif defined(GL_SHARED_TEXTURE_PALETTE_EXT)
//     glEnable(GL_SHARED_TEXTURE_PALETTE_EXT);   
// #endif
//     if( cmapHasChanged_ ) {
//       BuildTransferFunctions();
//       cmapHasChanged_ = false;
//     }
//   }
//   glColor4f(1,1,1,1); // set to all white for modulation

  // check to see if the window size has changed.
  SCIRexRenderData *rd = render_data_;
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT,vp);
  if( vp[2] != rd->viewport_x_ || vp[3] != rd->viewport_y_ ){
    rd->viewport_x_ = vp[2];
    rd->viewport_y_ = vp[3];
    rd->viewport_changed_ = true;
    delete [] rd->depth_buffer_;
    delete [] rd->write_buffer_;
    rd->depth_buffer_ = 
      scinew unsigned char[ rd->viewport_x_ * rd->viewport_y_ ];
    rd->write_buffer_ =
      scinew unsigned char[ 4*rd->viewport_x_ * rd->viewport_y_ ];
  }

  // read the depth buffer.
  glPixelStorei(GL_PACK_ALIGNMENT,1);
  glReadBuffer(GL_BACK);
  glReadPixels(0,0,
	       rd->viewport_x_,
	       rd->viewport_y_,
	       GL_DEPTH_COMPONENT,
	       GL_UNSIGNED_BYTE, rd->depth_buffer_);

  // set the view matrix
  glGetDoublev(GL_MODELVIEW_MATRIX, rd->mvmat_);
  glGetDoublev(GL_PROJECTION_MATRIX, rd->pmat_);

}

void
SCIRexRenderer::preDraw()
{
  switch (rs_) {
  case OVEROP:
//     glEnable(GL_BLEND);
//     glBlendEquation(GL_FUNC_ADD_EXT);
//     glBlendFunc( GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    break;
  case MIP:
//     glEnable(GL_BLEND);
//     glBlendEquation(GL_MAX_EXT);
//     glBlendFunc(GL_ONE, GL_ONE);
    break;
  case ATTENUATE:
//     glEnable(GL_BLEND);
//     glBlendEquation(GL_FUNC_ADD_EXT);
//     glBlendFunc(GL_CONSTANT_ALPHA_EXT, GL_ONE);
//     glBlendColor(1.f, 1.f, 1.f, 1.f/slices_);
    break;
  default:
    cerr<<"Shouldn't be here!\n";
    ASSERT(0);
  }
}

void
SCIRexRenderer::postDraw()
{
  switch (rs_) {
  case OVEROP:
  case MIP:
  case ATTENUATE:
//     glDisable(GL_BLEND);
    break;
  default:
    cerr<<"Shouldn't be here!\n";
    ASSERT(0);
  }
}

void
SCIRexRenderer::cleanup()
{


  if( cmap_.get_rep() )
#ifdef GL_TEXTURE_COLOR_TABLE_SGI
    glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
#elif defined(GL_SHARED_TEXTURE_PALETTE_EXT)
  glDisable(GL_SHARED_TEXTURE_PALETTE_EXT);
#endif
  // glDisable(GL_TEXTURE_3D_EXT);
  // glEnable(GL_DEPTH_TEST);  
}  


#define SCIREXRENDERER_VERSION 1

void SCIRexRenderer::io(Piostream&)
{
    // Nothing for now...
  NOT_FINISHED("SCIRexRenderer::io");
}


bool
SCIRexRenderer::saveobj(std::ostream&, const std::string&, GeomSave*)
{
   NOT_FINISHED("SCIRexRenderer::saveobj");
    return false;
}

} // End namespace Uintah
