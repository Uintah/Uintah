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
#include <Core/GLVolumeRenderer/VolumeUtils.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Datatypes/Field.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Mutex.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Geom/Color.h>
#include <Core/Containers/Array1.h>
#include <Core/Thread/Thread.h>
#include <GL/gl.h>
#ifdef __sgi
#include <slist>
#else
#include <ext/slist>
#endif
#include <iostream>
#include <sstream>

namespace Kurt {

using std::cerr;
using std::ostringstream;
using std::sort;

using SCIRun::Color;
using SCIRun::Array1;
using SCIRun::FieldHandle;
using SCIRun::Thread;
using SCIRun::GLVolumeRenderer;
//using SCIrun::largestPowerOf2;


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

  // Here is where we need to split up the volume between the windows
  make_render_windows(tex_, min_, max, is_fixed_, brick_size_, displays);
   
  
  // now set up the compositers
  for(int i = 0; i < ncompositers; i++){
    //     ostringstream comp;
    //     comp << "Compositer"<<i;
    //     char *c = new char(comp.str().length()];
    //     comp.str().copy(c, comp.str().length());
    compositers.push_back(new SCIRexCompositer( render_data_ ));
    for(int j = 0; j < int( windows.size()); j++) {
      compositers[i]->add( windows[j] );
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
  BBox b;
  std::vector<GLTexture3D *>::iterator it = textures.begin();
  for(; it != textures.end(); it++){
    (*it)->get_bounds(b);
//      cerr<<"texture bounding box = "<<b.min()<<", "<<b.max()<<endl;
  }
  bb.extend(b);
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

//   win_mutex_.lock(); 
//   cerr<<"check for Exit in renderer thread"<<endl;
//   win_mutex_.unlock();
  barrier->wait(waiters);

//   win_mutex_.lock(); 
//   cerr<<"update info for in renderer thread"<<endl;
//   win_mutex_.unlock();
  barrier->wait(waiters);

  // while rendering make sure the compositers are updated
  update_compositer_data();

//   win_mutex_.lock(); 
//   cerr<<"render windows in renderer thread"<<endl;
//   win_mutex_.unlock();
  barrier->wait(waiters);

//   win_mutex_.lock(); 
//   cerr<<"wait on compositers in renderer thread"<<endl;
//   win_mutex_.unlock();
  barrier->wait(waiters);

//   win_mutex_.lock(); 
//   cerr<<"wait on Display in renderer thread"<<endl;
//   win_mutex_.unlock();
  
  glDrawBuffer(GL_BACK);
  glEnable(GL_BLEND);
//   glBlendEquation(GL_FUNC_ADD_EXT);
//   glBlendEquation(GL_MAX);
//   glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_COLOR);
//  glBlendFunc(GL_DST_COLOR, GL_SRC_COLOR);
//  glBlendFunc(GL_DST_COLOR, GL_ONE_MINUS_SRC_COLOR);
//  glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_SRC_COLOR);
//  glBlendFunc( GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
 glBlendFunc( GL_ONE,GL_ONE_MINUS_SRC_ALPHA);

  glDrawPixels(render_data_->viewport_x_,
	       render_data_->viewport_y_,
	       GL_RGBA, GL_UNSIGNED_BYTE, render_data_->write_buffer_);
  glDisable(GL_BLEND);
  barrier->wait(waiters);
  render_data_->viewport_changed_ = false;

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
  double mvmat[16];
  SCIRexRenderData *rd = render_data_;
  glPixelStorei(GL_PACK_ALIGNMENT,1);
  glReadBuffer(GL_BACK);
  glGetDoublev(GL_MODELVIEW_MATRIX, mvmat);
  /* remember that the glmatrix is stored as
     0  4  8 12
     1  5  9 13
     2  6 10 14
     3  7 11 15 */

 // read the color buffer.  This can be done during rendering
  glReadPixels(0,0,
	       rd->viewport_x_,
	       rd->viewport_y_,
	       GL_RGBA,
	       GL_UNSIGNED_BYTE, rd->write_buffer_);
  // order the compositing.  This also can be done during rendering.
  Point viewPt = Point(-mvmat[12], -mvmat[13], -mvmat[14]);
  mvmat[12] = mvmat[13] = mvmat[14] = 0;
  Transform mat;
  mat.set( mvmat );
//   cerr<<"1. View Point  = "<<viewPt<<endl;
//  viewPt = field_transform.unproject(viewPt);
//    cerr<<"2. View Point  = "<<viewPt<<endl;
  viewPt = mat.project( viewPt );
//   viewPt = field_transform.unproject(viewPt);
  int *comp = render_data_->comp_order_;
  int i, j;
//   cerr<<"View Point  = "<<viewPt<<endl;
  vector<double> dist;
  vector<GLTexture3D *>::iterator it = textures.begin();
  for(i = 0; it != textures.end(); it++, i++){
    GLTexture3D *t = *it;
    Point center = (t->min() + (t->max() -t->min()) * 0.5);
    center = field_transform.project(center);
//     cerr<<"Center Point-"<<i<<" = "<<center<<", ";
    double len = (center - viewPt).length2();
//     cerr<<"Distance^2 = "<<len<<endl;
    dist.push_back( len );
    comp[i] = i; // default order
  }
  double tmpd;
  int tmpi;
  for( j = 0; j < dist.size(); j++){
    for(i = j+1 ; i < dist.size(); i++) {
      if( dist[i] > dist[j]){
	tmpd = dist[i];
	tmpi = comp[i];
	dist[i] = dist[j];
	dist[j] = tmpd;
	comp[i] = comp[j];
	comp[j] = tmpi;
	
      }
    }
  }

  if(render_data_->viewport_changed_){
    double ncompositers = (double)compositers.size();
    vector<SCIRexCompositer *>::iterator it = compositers.begin();
//     cerr<<"viewport = "<<render_data_->viewport_y_<<" by "<<
//      render_data_->viewport_x_<<"\n";
    for(int i = 0; it != compositers.end(); it++, i++){
      int ymin = render_data_->viewport_y_ * ( i/ncompositers);
      int ymax = render_data_->viewport_y_ * ( (i+1)/ncompositers);
//       cerr<<"ymin, ymax = "<<ymin<<", "<<ymax<<"\n";
      (*it)->SetFrame(0,ymin,render_data_->viewport_x_, ymax);
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
    cerr<<"Shouldn't be here! in SCIRexRenderer::preDraw\n";
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
    cerr<<"Shouldn't be here! in SCIRexRenderer::postDraw\n";
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
void
SCIRexRenderer::make_render_windows(FieldHandle tex_, 
			      double& min_, double& max,
			      bool is_fixed_, int brick_size_,
			      vector<char *>& displays)
{
  GLTexture3D *texture;
  LatVolMesh *mesh = dynamic_cast<LatVolMesh *> (tex_->mesh().get_rep());
  if(mesh){
    // set the transform for later use.
    field_transform = mesh->get_transform();

    // establish the axis with the most data.  We'll split along that axis.
    int long_dim = 0; // 0 not set, 1 x, 2 y, 3 z
    int long_dim_size = 0;
    
    int x_dim = mesh->get_nx(),  // store these because we're
        y_dim = mesh->get_ny(),  // going to mess with them later
        z_dim = mesh->get_nz();  // and we want to save the orginal values
    cerr<<"nx, ny, nz = "<<x_dim<<", "<<y_dim<<", "<<z_dim<<endl;
    int min_x = mesh->get_min_x(),
      min_y = mesh->get_min_y(),
      min_z = mesh->get_min_z();

    if ( x_dim >= y_dim && x_dim >= z_dim ){ 
      long_dim = 1;
      long_dim_size = x_dim;
    } else if( y_dim >= z_dim){
      long_dim = 2;
      long_dim_size = y_dim;
    } else {
      long_dim = 3;
      long_dim_size = z_dim;
    }

    // How many windows?  Make a subtexture for each window.
    int nwindows = displays.size();
    int i,j;
    // is dim_size > brick_size_?  Hopefully yes, otherwise we
    // should not be using this code.
    cerr<<"Original brick size is "<<brick_size_<<endl;
    int bsize = brick_size_;
    while( long_dim_size  < (nwindows-1)* bsize ){
      bsize = largestPowerOf2( bsize -1 );
    }  
    cerr<<"Using brick size "<< bsize <<endl;
    //    cerr<<"long dim is "<< long_dim <<", long dim size is "<<long_dim_size<<endl;
    // make an array for ordering the composition
    render_data_->comp_order_ = new int[ nwindows ];

    for(i = 0, j = 0; i < nwindows; i++, j+= (bsize-1)){
      
      //build the rendering windows
      render_data_->comp_order_[i] = i; // just use default order for setup
      ostringstream win;
      win << "Win"<<i;
      char *w = new char[win.str().length()];
      strcpy(w, win.str().c_str());
      cerr<<"i = "<<i<<", ";
      cerr<<"in SCIRexRenderer using display "<<displays[i]<<endl;
      windows.push_back( new SCIRexWindow(w,displays[i], render_data_));


      // edit the mesh so that the texture constructor 
      // iterates over the correct range.
      if( i == (nwindows - 1) ){
	if(long_dim == 1){
	  mesh->set_min_x(min_x + j); 
	  mesh->set_nx(x_dim - j); 
	} else if(long_dim == 2){
	  mesh->set_min_y(min_y + j); 
	  mesh->set_ny(y_dim - j); 
	} else {
	  mesh->set_min_z(min_z + j); 
	  mesh->set_nz(z_dim - j); 
	}	  
      } else {
	if(long_dim == 1){
	  mesh->set_min_x(min_x + j);
	  mesh->set_nx(bsize); 
	} else if(long_dim == 2){
	  mesh->set_min_y(min_y + j); 
	  mesh->set_ny(bsize); 
	} else {
	  mesh->set_min_z(min_z + j);
	  mesh->set_nz(bsize);
	}	  
      }
      
      //build the texture
      texture = scinew GLTexture3D(tex_, min_, max_, is_fixed_,
				   bsize);

      if (!is_fixed_) { // if not fixed, overwrite min/max values on Gui
	texture->getminmax(min_, max_);
      }
      BBox bb0;
      texture->get_bounds( bb0 );
      cerr<<"texture bounding box at construction = "<<
	bb0.min()<<", "<<bb0.max()<<endl;

      texture->set_slice_bounds(Point(min_x, min_y, min_z),
				Point(min_x + x_dim, min_y + y_dim, 
				       min_z + z_dim));
      textures.push_back( texture );
      // Now build renderer and add it to the windows
      GLVolumeRenderer *vr;
      vr = new GLVolumeRenderer( 0x12345676, texture, cmap_);
      vr->DrawFullRes();
      vr->SetNSlices( slices_ );
      vr->SetSliceAlpha( slice_alpha_ );
      windows[i]->addGeom( vr );
      renderers.push_back(vr);

    }
    
    // Now just in case something unseen is using the mesh, set it back to
    // its original form. Probably, unecessary.
//     mesh->set_min_x(min_x);
//     mesh->set_min_y(min_y);
//     mesh->set_min_z(min_z);
//     mesh->set_nx(x_dim);
//     mesh->set_ny(y_dim);
//     mesh->set_nz(z_dim);
  } else {
    cerr<<"dynamic_cast<LatVolMesh *> failed !\n";
    cerr<<"initialization/rebuild failed !\n";
    ASSERT( mesh );
  }
}

} // End namespace Uintah
