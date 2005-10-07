//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : FlowRenderer2D.cc
//    Author : Milan Ikits
//    Date   : Thu Jul  8 00:04:15 2004

#include <Core/Datatypes/ImageField.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Geometry/Transform.h>
#include <Core/Volume/FlowRenderer2D.h>
#include <Core/Volume/VolShader.h>
#include <Core/Volume/FlowShaders.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Geom/Pbuffer.h>
#include <Core/Volume/TextureBrick.h>
#include <Core/Volume/Utils.h>
#include <Core/Volume/CM2Shader.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/NotFinished.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <stdlib.h>
#include <sci_values.h>
using std::cerr;
using std::endl;
using std::string;

namespace SCIRun {

#ifdef _WIN32
#define GL_FUNC_ADD 2
#define GL_MAX 2
#define GL_TEXTURE_3D 2
#define glBlendEquation(x)
#endif

const int BufferSize = 512;

//static SCIRun::DebugStream dbg("FlowRenderer2D", false);

FlowRenderer2D::FlowRenderer2D(FieldHandle field,
                               ColorMapHandle cmap,
                               int tex_mem):
  GeomObj(),
  field_(field),
  ifv_(0),
  cmap_(cmap),
  mutex_("FlowRenderer Mutex"),
  mode_(MODE_NONE),
  shading_(false),
  sw_raster_(false),
  di_(0),
  pbuffers_created_(false),
  cmap_tex_(0),
  cmap_dirty_(true),
  adv_tex_(0),
  adv_dirty_(true),
  adv_is_initialized_(false),
  adv_accums_(10),
  conv_tex_(0),
  conv_dirty_(true),
  conv_is_initialized_(false),
  conv_accums_(10),
  flow_tex_(0),
  flow_dirty_(true),
  re_accum_(true),
  noise_tex_(0),
  build_noise_(true),
  use_pbuffer_(false),
  buffer_width_(0),
  buffer_height_(0),
  adv_buffer_(0),
  blend_buffer_(0),
  blend_num_bits_(8),
  use_blend_buffer_(true),
  free_tex_mem_(tex_mem),
  current_shift_(0),
  conv_init_(new FragmentProgramARB( ConvInit )),
  adv_init_(new FragmentProgramARB( AdvInit )),
  conv_accum_(new FragmentProgramARB( ConvAccum )),
  adv_accum_(new FragmentProgramARB( AdvAccum )),
  conv_rewire_(new FragmentProgramARB( ConvRewire )),
  adv_rewire_(new FragmentProgramARB( AdvRewire )),
  conv_init_rect_(new FragmentProgramARB( ConvInitRect )),
  adv_init_rect_(new FragmentProgramARB( AdvInitRect )),
  conv_accum_rect_(new FragmentProgramARB( ConvAccumRect )),
  adv_accum_rect_(new FragmentProgramARB( AdvAccumRect )),
  conv_rewire_rect_(new FragmentProgramARB( ConvRewireRect )),
  adv_rewire_rect_(new FragmentProgramARB( AdvRewireRect )),
  is_initialized_(false)
{
  mode_ = MODE_LIC;
  
  for(int i = 0; i < 500; i++){
    pair<float,float> v( drand48(), drand48());
    shift_list_.push_back( v );
  }
  current_shift_ = 0;
}

FlowRenderer2D::FlowRenderer2D(const FlowRenderer2D& copy):
  GeomObj(copy),
  field_(copy.field_),
  ifv_(copy.ifv_),
  cmap_(copy.cmap_),
  mutex_("FlowRenderer Mutex"),
  mode_(copy.mode_),
  shading_(copy.shading_),
  sw_raster_(copy.sw_raster_),
  di_(copy.di_),
  pbuffers_created_(copy.pbuffers_created_),
  cmap_tex_(copy.cmap_tex_),
  cmap_dirty_(copy.cmap_dirty_),
  adv_tex_(copy.adv_tex_),
  adv_dirty_(copy.adv_dirty_),
  adv_is_initialized_(copy.adv_is_initialized_),
  adv_accums_(copy.adv_accums_),
  conv_tex_(copy.conv_tex_),
  conv_dirty_(copy.conv_dirty_),
  conv_is_initialized_(copy.conv_is_initialized_),
  conv_accums_(copy.conv_accums_),
  flow_tex_(copy.flow_tex_),
  flow_dirty_(copy.flow_dirty_),
  re_accum_(copy.re_accum_),
  use_pbuffer_(copy.use_pbuffer_),
  buffer_width_(copy.buffer_width_),
  buffer_height_(copy.buffer_height_),
  adv_buffer_(copy.adv_buffer_),
  blend_buffer_(copy.blend_buffer_),
  blend_num_bits_(copy.blend_num_bits_),
  use_blend_buffer_(copy.use_blend_buffer_),
  free_tex_mem_(copy.free_tex_mem_),
  current_shift_(copy.current_shift_),
  conv_init_(copy.conv_init_),
  adv_init_ (copy.adv_init_ ),
  conv_accum_(copy.conv_accum_),
  adv_accum_(copy.adv_accum_),
  conv_rewire_(copy.conv_rewire_),
  adv_rewire_(copy.adv_rewire_),
  conv_init_rect_(copy.conv_init_rect_),
  adv_init_rect_(copy.adv_init_rect_),
  conv_accum_rect_(copy.conv_accum_rect_),
  adv_accum_rect_(copy.adv_accum_rect_),
  conv_rewire_rect_(copy.conv_rewire_rect_),
  adv_rewire_rect_(copy.adv_rewire_rect_),
  is_initialized_(copy.is_initialized_)
{}

FlowRenderer2D::~FlowRenderer2D()
{}

GeomObj*
FlowRenderer2D::clone()
{
  return scinew FlowRenderer2D(*this);
}
void
FlowRenderer2D::reset()
{
  flow_dirty_ = true;
  adv_dirty_ = true;
  conv_dirty_ = true;
  re_accum_ = true;
  is_initialized_ = false;
  adv_is_initialized_ = false;
  conv_is_initialized_ = false;
  current_shift_ = 0;
}

void
FlowRenderer2D::set_field(FieldHandle field)
{
  mutex_.lock();
  field_ = field;
  flow_dirty_ = true;
  re_accum_ = true;
  mutex_.unlock();
}

void
FlowRenderer2D::set_colormap(ColorMapHandle cmap)
{
  mutex_.lock();
  cmap_ = cmap;
  cmap_dirty_ = true;
  mutex_.unlock();
}

void
FlowRenderer2D::set_mode(FlowMode mode)
{
  if(mode_ != mode) {
    mode_ = mode;
  }
}

    
#ifdef SCI_OPENGL
void
FlowRenderer2D::draw(DrawInfoOpenGL* di, Material* mat, double)
{
  if(!pre_draw(di, mat, shading_)) return;
  mutex_.lock();
  di_ = di;
  if(di->get_drawtype() == DrawInfoOpenGL::WireFrame ) {
    draw_wireframe();
  } else {
    draw();
  }
  di_ = 0;
  mutex_.unlock();
}

void
FlowRenderer2D::draw()
{

  di_->polycount++;
  glPushMatrix();

  
  GLfloat clear_color[4];
  glGetFloatv(GL_COLOR_CLEAR_VALUE, clear_color);
  int vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  //--------------------------------------------------------------------------
  // enable alpha test
  glEnable(GL_ALPHA_TEST);
  glAlphaFunc(GL_GREATER, 0.0);
  glColor4f(1.0, 1.0, 1.0, 1.0);
  glDepthMask(GL_TRUE);

  //--------------------------------------------------------------------------
  // build textures
  build_flow_tex();
  build_noise();
  build_colormap();
  //--------------------------------------------------------------------------
  bind_colormap(3);
  //--------------------------------------------------------------------------
  // set up pbuffers
//   if(!pbuffers_created_ && use_pbuffer_)
//     create_pbuffers(buffer_width_, buffer_height_);
  
//   if( adv_buffer_ ){
  float scale = 1.0;
//   pair<float,float> shft = shift_list_[current_shift_];
//     adv_init(adv_buffer_,scale, shift_list_[current_shift_]);

//     bind_noise();
  int c_shift = current_shift_;
  current_shift_++;
  if(current_shift_ >= shift_list_.size() ) current_shift_ = 0;
  build_adv( scale, shift_list_[c_shift] );
  //  load_adv();
  next_shift(&c_shift);
    
  if( re_accum_ ){
    //must be called after build_flow_tex()
    float pixelx = 1.f/(float)w_;
    float pixely = 1.f/(float)h_;
      for(int i = 0; i < adv_accums_; i++){
        adv_accum( pixelx, pixely, scale, shift_list_[c_shift] );
        next_shift(&c_shift);
      }
      adv_rewire();
      
      build_conv(scale);
      for(int i = 0; i < conv_accums_; i++){
        conv_accum( pixelx, pixely, scale);
      }
      conv_rewire();
      re_accum_ = false;
   }
  
  load_conv();
  bind_conv( 1 );
//   bind_noise();
//   bind_flow_tex();

  //-----------------------------------------------------
  // set up shader
  FragmentProgramARB* shader; // = adv_accum_;
  shader = new FragmentProgramARB( DrawNoise );
  //-----------------------------------------------------
  if( shader ){
    if(!shader->valid()) {
      shader->create();
      shader->setLocalParam(1, 1.0, 1.0, 1.0, 1.0);
      shader->setLocalParam(2, shift_list_[c_shift].first,
                            shift_list_[c_shift].second, 0.5, 0.5);
    }
    shader->bind();
  }
  
  glBegin( GL_QUADS );
  {
    for (int i = 0; i < 4; i++) {
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader) 
       glMultiTexCoord2fv(GL_TEXTURE0,tex_coords_+i*2);
       glMultiTexCoord2fv(GL_TEXTURE1,tex_coords_+i*2);
//       if(use_fog) {
// 	float *pos = pos_coords_+i*3;
// 	float vz = mvmat[2]* pos[0]
// 	  + mvmat[6]*pos[1] 
// 	  + mvmat[10]*pos[2] + mvmat[14];
// 	glMultiTexCoord3f(GL_TEXTURE2, -vz, 0.0, 0.0);
//    }
#else
       glTexCoord2fv(tex_coords_+i*2);
#endif
      glVertex3fv(pos_coords_+i*3);
    }
  }
  glEnd();
  glFlush();
  glDisable(GL_ALPHA_TEST);
  
  if(shader)
    shader->release();
  
//   release_flow_tex();
//   release_noise();
  release_conv(1);
  release_colormap(3);
  
  glPopMatrix();
  
  CHECK_OPENGL_ERROR("FlowRenderer2D::draw_volume end");
}

void 
FlowRenderer2D::adv_init( Pbuffer*& pb, float scale,
                          pair<float, float>& shift)
  {
  if(!adv_init_->valid()) {
    adv_init_->create();
    adv_init_->setLocalParam(1, scale, 1.0, 1.0, 1.0);
    adv_init_->setLocalParam(2, shift.first, shift.second, 1.0, 1.0);
  }
  pb->activate();
  
  pb->set_use_texture_matrix(false);
  pb->set_use_default_shader(false);

  glDrawBuffer(GL_FRONT);
  glViewport(0, 0, pb->width(), pb->height());
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  pb->swapBuffers();
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(-1.0, -1.0, 0.0);
  glScalef(2.0, 2.0, 2.0);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glDisable(GL_CULL_FACE);
  glDisable(GL_BLEND);

  //bind_noise in pbuffer context

  bind_noise();
  adv_init_->bind();
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader) 
  glActiveTexture(GL_TEXTURE0);
#endif
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  float tex_coords [] = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
  float pos_coords [] = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
  glBegin( GL_QUADS );
  {
    for (int i = 0; i < 4; i++) {
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader) 
      glMultiTexCoord2fv(GL_TEXTURE0,tex_coords+i*2);
      CHECK_OPENGL_ERROR("adv_init()");

#else
      glTexCoord2fv(tex_coords+i*2);
#endif
       glVertex3fv(pos_coords+i*3);
      CHECK_OPENGL_ERROR("adv_init()");
    }
  }
  glEnd();
  CHECK_OPENGL_ERROR("adv_init()");
  pb->swapBuffers();

  adv_init_->release();
  release_noise();
  pb->deactivate();
  pb->set_use_texture_matrix(true);

}


void 
FlowRenderer2D::init()
{
  ifv_ = dynamic_cast<ImageField<Vector>* >(field_.get_rep());
  ASSERT( ifv_ != 0 );
  is_initialized_ = true;
  build_flow_tex();
}
void
FlowRenderer2D::draw_wireframe()
{
  Ray view_ray = compute_view();
  Transform tform;
  field_->mesh()->transform(tform);
  double mvmat[16];
  tform.get_trans(mvmat);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

void
FlowRenderer2D::set_sw_raster(bool b)
{
  if(sw_raster_ != b) {
    mutex_.lock();
    sw_raster_ = b;
    cmap_dirty_ = true;
    mutex_.unlock();
  }
}


void
FlowRenderer2D::set_blend_num_bits(int b)
{
  mutex_.lock();
  blend_num_bits_ = b;
  mutex_.unlock();
}

bool
FlowRenderer2D::use_blend_buffer()
{
  return use_blend_buffer_;
}

#define FLOWRENDERER2D_VERSION 1

void 
FlowRenderer2D::io(Piostream&)
{
  // nothing for now...
  NOT_FINISHED("FlowRenderer2D::io");
}

void
FlowRenderer2D::get_bounds(BBox& bb)
{
  bb = field_->mesh()->get_bounding_box();
}

Ray
FlowRenderer2D::compute_view()
{
  Transform field_trans;
  field_->mesh()->transform(field_trans);
  double mvmat[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, mvmat);
  // index space view direction
  Vector v = field_trans.unproject(Vector(-mvmat[2], -mvmat[6], -mvmat[10]));
  v.safe_normalize();
  Transform mv;
  mv.set_trans(mvmat);
  Point p = field_->mesh()->get_bounding_box().center();
  return Ray(p, v);
}


void
FlowRenderer2D::build_colormap()
{
  if(cmap_dirty_ ) {
    bool size_dirty = false;
    if(256 != cmap_array_.dim1()) {
      cmap_array_.resize(256, 4);
      size_dirty = true;
    }
    // rebuild texture
    double dv = 1.0/(cmap_array_.dim1() - 1);
    for(int j=0; j<cmap_array_.dim1(); j++) {
      // interpolate from colormap
      const Color &c = cmap_->getColor(j*dv);
      double alpha = cmap_->getAlpha(j*dv);
      // pre-multiply and quantize
      cmap_array_(j,0) = c.r()*alpha;
      cmap_array_(j,1) = c.g()*alpha;
      cmap_array_(j,2) = c.b()*alpha;
      cmap_array_(j,3) = alpha;
    }

#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
    // This texture is not used if there is no shaders.
    // glColorTable is used instead.
    if (ShaderProgramARB::shaders_supported())
    {
      // Update 1D texture.
      if (cmap_tex_ == 0 || size_dirty)
      {
	glDeleteTextures(1, &cmap_tex_);
	glGenTextures(1, &cmap_tex_);
	glBindTexture(GL_TEXTURE_1D, cmap_tex_);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA16, cmap_array_.dim1(), 0,
		     GL_RGBA, GL_FLOAT, &cmap_array_(0,0));
      }
      else
      {
	glBindTexture(GL_TEXTURE_1D, cmap_tex_);
	glTexSubImage1D(GL_TEXTURE_1D, 0, 0, cmap_array_.dim1(),
			GL_RGBA, GL_FLOAT, &cmap_array_(0,0));
      }
    }
#endif
    cmap_dirty_ = false;
  }
  CHECK_OPENGL_ERROR("FlowRenderer2D::build_colormap()");
}




void
FlowRenderer2D::bind_colormap( int reg )
{
#if defined( GL_TEXTURE_COLOR_TABLE_SGI ) && defined(__sgi)
  glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
  glColorTable(GL_TEXTURE_COLOR_TABLE_SGI,
               GL_RGBA,
               256,
               GL_RGBA,
               GL_FLOAT,
               &(cmap_array_(0, 0)));
#elif defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
  if (ShaderProgramARB::shaders_supported())
  {
    glActiveTexture(GL_TEXTURE0+reg);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_TEXTURE_1D);
    glBindTexture(GL_TEXTURE_1D, cmap_tex_);
    // enable data texture unit 1
    glActiveTexture(GL_TEXTURE1_ARB);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glActiveTexture(GL_TEXTURE0_ARB);
  }
  else
#endif
#ifndef __sgi
#if defined(GL_EXT_shared_texture_palette) && !defined(__APPLE__)
  {
    glEnable(GL_SHARED_TEXTURE_PALETTE_EXT);
    glColorTable(GL_SHARED_TEXTURE_PALETTE_EXT,
		 GL_RGBA,
		 256,
		 GL_RGBA,
		 GL_FLOAT,
		 &(cmap_array_(0, 0)));
  }
#else
  {
    static bool warned = false;
    if( !warned ) {
      std::cerr << "No colormaps available." << std::endl;
      warned = true;
    }
  }
#endif
#endif
  CHECK_OPENGL_ERROR("FlowRenderer2D::bind_colormap()");

}


void
FlowRenderer2D::release_colormap( int reg )
{
#if defined(GL_TEXTURE_COLOR_TABLE_SGI) && defined(__sgi)
  glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
#elif defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
  if (ShaderProgramARB::shaders_supported())
  {
    // bind texture to unit 3
    glActiveTexture(GL_TEXTURE0+reg);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glDisable(GL_TEXTURE_1D);
    glBindTexture(GL_TEXTURE_1D, 0);
    // enable data texture unit 0
    glActiveTexture(GL_TEXTURE0_ARB);
  }
  else
#endif
#ifndef __sgi
#if defined(GL_EXT_shared_texture_palette) && !defined(__APPLE__)
  {
    glDisable(GL_SHARED_TEXTURE_PALETTE_EXT);
  }
#else
  {
    // Already warned in bind.  Do nothing.
  }
#endif
#endif
  CHECK_OPENGL_ERROR("FlowRenderer2D::release_colormap()");
}

void
FlowRenderer2D::build_flow_tex()
{
  if( !is_initialized_ || flow_dirty_) {
    ifv_ = dynamic_cast<ImageField<Vector>* >(field_.get_rep());
    // If we've made it to here, ifv_ better be valid!
    ASSERT( ifv_ != 0 );

    ImageMesh *im = dynamic_cast<ImageMesh *> (field_->mesh().get_rep());
    ASSERT( im != 0);
    unsigned w = im->get_ni();
    unsigned h = im->get_nj();
    if( field_->basis_order() == 0 ){
      --w; --h;
    }
    w_ = w;
    h_ = h;
    buffer_width_ = NextPowerOf2( w );
    buffer_height_ = NextPowerOf2( h );

    // Array3 is ordered opposite what OpenGL expects, 
    // switch height and width.
    if(flow_array_.dim1() != buffer_height_ ||
       flow_array_.dim2() != buffer_width_ ) {
      flow_array_.resize(buffer_height_, buffer_width_,2);
    }
    BBox bb = im->get_bounding_box();

    //***************************************************
      // we need to find the corners of the square in space
      // use the node indices to grab the corner points
      ImageMesh::Node::index_type ll(im, 0, 0);
      ImageMesh::Node::index_type lr(im, im->get_ni() - 1, 0);
      ImageMesh::Node::index_type ul(im, 0, im->get_nj() - 1);
      ImageMesh::Node::index_type ur(im, im->get_ni() - 1, 
                                     im->get_nj() - 1);
      Point p1, p2, p3, p4;
      im->get_center(p1, ll);
      pos_coords_[0] = p1.x();
      pos_coords_[1] = p1.y();
      pos_coords_[2] = p1.z();

      im->get_center(p2, lr);
      pos_coords_[3] = p2.x();
      pos_coords_[4] = p2.y();
      pos_coords_[5] = p2.z();

      im->get_center(p3, ur);
      pos_coords_[6] = p3.x();
      pos_coords_[7] = p3.y();
      pos_coords_[8] = p3.z();

      im->get_center(p4, ul);
      pos_coords_[9] = p4.x();
      pos_coords_[10] = p4.y();
      pos_coords_[11] = p4.z();

      // establish which plane is closest to the slice 
      // and drop that vector component. Hack for now...
      Vector diag = bb.max() - bb.min();
      int plane = ((diag.x() < diag.y()) ? 
                   ((diag.x() < diag.z()) ? 0 : 2) :
                   (diag.y() < diag.z()) ? 1 : 2 );

      int vi, vj;
      if( plane == 0 ){ vi = 1, vj = 2;}
      else if(plane == 1) { vi = 0, vj = 2; }
      else { vi = 0, vj = 1; }

      // Texture coords.
      double tmin_x, tmax_x, tmin_y, tmax_y;
      // Field needs to be scaled, use min and max for now...
      // To do: add Tcl variable for fixing the scale
#if 0
      double min_x = MAXDOUBLE, min_y = MAXDOUBLE, 
        max_x = -MAXDOUBLE, max_y = -MAXDOUBLE;
#endif
      if( field_->basis_order() == 0){
        // Set texture coords
        tmin_x = 0.0;
        tmax_x = w/(double)buffer_width_;
        tmin_y = 0.0;
        tmax_y = h/(double)buffer_height_;
        ImageMesh::Face::iterator iter, end;
        im->begin( iter ); im->end( end );

#if 0  // if we need to normalize the field use this
        // Get minmax values
        while(iter != end){
          Vector val;
          ifv_->value( val, *iter);
          min_x = Min(min_x, val[vi]);
          max_x = Max(max_x, val[vi]);
          min_y = Min(min_y, val[vj]);
          max_y = Max(max_y, val[vj]);
          ++iter;
        }
        // Reset iterator
        im->begin( iter );
#endif        

        while(iter != end){
          // Note the i,j switch here
          int i = iter.j_;
          int j = iter.i_;
          Vector val;
          ifv_->value( val, *iter);
          // Scale the values
#if 0  // again normalization
          flow_array_(i,j, 0) = (val[vi] - min_x)/(max_x - min_x);
          flow_array_(i,j, 1) = (val[vj] - min_y)/(max_y - min_y);
#endif
          Vector v(val[vi], val[vj], 0.0);
//           v.safe_normalize();
          flow_array_(i,j, 0) = v.x();
          flow_array_(i,j, 1) = v.y();
          ++iter;
        }

      } else if( field_->basis_order() == 1){ // basis better be 1 for now

        // Set texture coords.
        tmin_x = 0.5/(double)buffer_width_;
        tmax_x = ( w - 0.5)/(double)buffer_width_;
        tmin_y = 0.5/(double)buffer_height_;
        tmax_y = ( h - 0.5)/(double)buffer_height_;
        ImageMesh::Node::iterator iter, end;
        im->begin( iter ); im->end( end );
#if 0  // for normalization
        // Get minmax values
        while(iter != end){
          Vector val;
          ifv_->value( val, *iter);
          min_x = Min(min_x, val[vi]);
          max_x = Max(max_x, val[vi]);
          min_y = Min(min_y, val[vj]);
          max_y = Max(max_y, val[vj]);
          ++iter;
        }
        // Reset iterator
        im->begin( iter );
#endif

        while(iter != end){
          int i = (*iter).j_;
          int j = (*iter).i_;
          Vector val;
          ifv_->value( val, *iter);

          // Scale the values
          flow_array_(i,j, 0) = val[vi];//(val[vi] - min_x)/(max_x - min_x);
          flow_array_(i,j, 1) = val[vj];//(val[vj] - min_y)/(max_y - min_y);
#if 0  // again normalization
          flow_array_(i,j, 0) = val[vi];//(val[vi] - min_x)/(max_x - min_x);
          flow_array_(i,j, 1) = val[vj];//(val[vj] - min_y)/(max_y - min_y);
#endif
          ++iter;
        }
      } 
      tex_coords_[0] = tmin_x; tex_coords_[1] = tmin_y;
      tex_coords_[4] = tmax_x; tex_coords_[3] = tmin_y;
      tex_coords_[2] = tmax_x; tex_coords_[5] = tmax_y;
      tex_coords_[6] = tmin_x; tex_coords_[7] = tmax_y;
  }
   is_initialized_ = true;
     CHECK_OPENGL_ERROR("FlowRenderer2D::build_flow_tex()");

}

void
FlowRenderer2D::load_flow_tex()
{

#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
      // This texture is not loaded if there are no shaders.
      if (ShaderProgramARB::shaders_supported())
      {
        // Update 2D texture.
        if (flow_tex_ == 0 || flow_dirty_)
        {
          glDeleteTextures(1, &flow_tex_);
          glGenTextures(1, &flow_tex_);
          glBindTexture(GL_TEXTURE_2D, flow_tex_);
//           glPixelStorei(GL_UNPACK_ALIGNMENT, 1); 
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
          glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE16_ALPHA16, 
                       // Note the dim1, dim2 switch here.
                       flow_array_.dim2(), flow_array_.dim1(), 
                       0, GL_LUMINANCE_ALPHA, GL_FLOAT, &flow_array_(0,0,0));
          flow_dirty_ = false;
          glBindTexture(GL_TEXTURE_2D, 0);
        }
        else
        {
          glBindTexture(GL_TEXTURE_2D, flow_tex_);
          glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                          // Note the dim1, dim2 switch here.
                          flow_array_.dim2(), flow_array_.dim1(), 
                          GL_LUMINANCE_ALPHA, GL_FLOAT, &flow_array_(0,0,0));
        }
      }
#endif
     CHECK_OPENGL_ERROR("FlowRenderer2D::load_flow_tex()");

}


void
FlowRenderer2D::bind_flow_tex( int reg )
{
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
  if (ShaderProgramARB::shaders_supported())
  {
    glActiveTexture(GL_TEXTURE0+reg);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, flow_tex_);
    glActiveTexture(GL_TEXTURE0_ARB);
  }
  
#endif
  CHECK_OPENGL_ERROR("FlowRenderer2D::bind_flow_tex()");
}

void
FlowRenderer2D::release_flow_tex( int reg )
{
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
  if (ShaderProgramARB::shaders_supported())
  {
    // bind texture to unit 1
    glActiveTexture(GL_TEXTURE0+reg);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    // enable data texture unit 0
    glActiveTexture(GL_TEXTURE0_ARB);
  }
#endif
  CHECK_OPENGL_ERROR("FlowRenderer2D::release_flow_tex()");
}

float 
FlowRenderer2D::get_interpolated_value( Array3<float>& array, float x, float y )
{ 
  // Array3 will assert that x,y is in range.
  int i = int(x);
  int j = int(y);
  // compute the remainders
  float ir = x - float(i);
  float jr = y - float(j);
  // compute weights
  float w[4];
  w[0] =  (1 - ir) * (1 - jr);
  w[1] =  ir * (1 - jr);
  w[2] =  ir * jr; 
  w[3] =  (1 - ir) * jr;
  return w[0] * array(i, j, 2) + w[1] * array(i+1, j, 2) +
    w[2] * array(i+1, j+1, 2) + w[3] * array(i, j+1, 2);
}

void
FlowRenderer2D::build_adv(float scale, pair<float, float>& shift)
{
  // Software version.  
  if( !adv_is_initialized_ || re_accum_) {
    int nx = buffer_width_;
    int ny = buffer_height_;
    if( adv_array_.dim1() != ny || adv_array_.dim2() != nx ){
      adv_array_.resize(ny, nx, 4);
    }
    int x = 0, y = 1, z = 2, w = 3;
    float r0[4];
    for(int j = 0; j < adv_array_.dim2(); j++){
      double x_coord = (j)/float(adv_array_.dim2()-1);
       for( int i = 0; i < adv_array_.dim1(); i++){
         double y_coord = (i)/float(adv_array_.dim1()-1);
         r0[x] = shift.first + x_coord;
         r0[y] = shift.second + y_coord;
         r0[x] = ( r0[x] > 1.0 ) ? r0[x] - 1 : r0[x];
         r0[y] = ( r0[y] > 1.0 ) ? r0[y] - 1 : r0[y];
         r0[x] = get_interpolated_value(noise_array_,
                                        r0[y] * (noise_array_.dim1() - 2),
                                        r0[x] * (noise_array_.dim2() - 2));
//          if( re_accum_ && adv_is_initialized_ ) {
//            // do nothing, use stored location
//          } else {
           adv_array_(i, j, x) = x_coord * 0.5 + 0.25;
           adv_array_(i, j, y) = y_coord * 0.5 + 0.25;
//          }
        adv_array_(i, j, z) = r0[x] * scale;
        adv_array_(i, j, w) = scale;
      }
    }
    adv_is_initialized_ = true;
    adv_dirty_ = true;
  }
}

void
FlowRenderer2D::load_adv()
{
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
  // This texture is not loaded if there are no shaders.
  if (adv_dirty_)
  {
    // Update 2D texture.
    glDeleteTextures(1, &adv_tex_);
    glGenTextures(1, &adv_tex_);
    glBindTexture(GL_TEXTURE_2D, adv_tex_);
    //           glPixelStorei(GL_UNPACK_ALIGNMENT, 1); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16,
                 // Note the dim1, dim2 switch here.
                 adv_array_.dim2(), adv_array_.dim1(), 
                 0, GL_RGBA, GL_FLOAT, &adv_array_(0,0,0));
    adv_dirty_ = false;
  }
  else
  {
    glBindTexture(GL_TEXTURE_2D, adv_tex_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    // Note the dim1, dim2 switch here.
                    adv_array_.dim2(), adv_array_.dim1(), 
                    GL_RGBA, GL_FLOAT, &adv_array_(0,0,0));
  }
#endif
}

void
FlowRenderer2D::adv_accum( float pixel_x, float pixel_y,
                           float scale, pair<float,float>&  shift)
{

  // Software version.  Hardware version is done with adv_acc shader
  int endy = int(tex_coords_[2] * (flow_array_.dim2() - 1));
  int endx = int(tex_coords_[7] * (flow_array_.dim1() - 1));
  
  float r0[4], r1[4], noise;
  int x = 0, y = 1, z = 2, w = 3;

  for(int j = 0; j < endy; j++ ){
    for(int i = 0; i < endx; i++ ) {
      r0[x] = adv_array_( i, j, x);
      r0[y] = adv_array_( i, j, y);
      r0[z] = adv_array_( i, j, z);
      r0[w] = adv_array_( i, j, w);
      
      r1[z] = Clamp((r0[x] * 2) - 0.5, 0.0, 1.0);
      r1[w] = Clamp((r0[y] * 2) - 0.5, 0.0, 1.0);
      r1[x] = flow_array_( int(r1[w] * (flow_array_.dim1() - 1)), 
                           int(r1[z] * (flow_array_.dim2() - 1)), 0) - 0.5;
      r1[y] = flow_array_( int(r1[w] * (flow_array_.dim1() - 1)), 
                           int(r1[z] * (flow_array_.dim2() - 1)), 1) - 0.5;
      
      r1[z] = r1[z] + shift.first;
      r1[w] = r1[w] + shift.second;
      r1[z] = r1[z] > 1.0 ? r1[z] - 1.0 : r1[z];
      r1[w] = r1[w] > 1.0 ? r1[w] - 1.0 : r1[w];  
      noise = get_interpolated_value(noise_array_, 
                                     r1[w] * (noise_array_.dim1() - 2 ),
                                     r1[z] * (noise_array_.dim2() - 2));
      
    
      r1[x] = pixel_x * r1[x];
      r1[y] = pixel_y * r1[y];
      adv_array_( i, j, x ) = r1[x] * 4 + r0[x];
      adv_array_( i, j, y ) = r1[y] * 4 + r0[y];
      adv_array_( i, j, z ) = noise * scale + r0[z];
      adv_array_( i, j, w ) = r0[w] + scale;
    }
  }
      
  adv_dirty_ = true;   
  
}

void
FlowRenderer2D::adv_rewire()
{
  // software version
  float r0[4];
  int x = 0, y = 1, z = 2, w = 3;

  for(int j = 0; j < adv_array_.dim2(); j++){
    for( int i = 0; i < adv_array_.dim1(); i++){
      r0[x] = adv_array_( i, j, x);
      r0[y] = adv_array_( i, j, y);
      r0[z] = adv_array_( i, j, z);
      r0[w] = adv_array_( i, j, w);
      r0[x] = 1.0/r0[w];
//       adv_array_(i, j, x) = r0[z] * r0[x];
//       adv_array_(i, j, y) = r0[z] * r0[x];
      adv_array_(i, j, z) = r0[z] * r0[x];
      adv_array_(i, j, w) = 1.0;
    }
  }
}

void
FlowRenderer2D::build_conv(float scale )
{
  if( !conv_is_initialized_  || re_accum_) {
    int nx = buffer_width_;
    int ny = buffer_height_;
    if( conv_array_.dim1() != ny || conv_array_.dim2() != nx ){
      conv_array_.resize(ny, nx, 4);
    }
    int x = 0, y = 1, z = 2, w = 3;
    float r0[4];
    for(int j = 0; j < conv_array_.dim2(); j++){
      double x_coord = (j)/float(conv_array_.dim2()-1);
       for( int i = 0; i < conv_array_.dim1(); i++){
         double y_coord = (i)/float(conv_array_.dim1()-1);
         r0[x] = x_coord;
         r0[y] = y_coord;
//          r0[x] = ( r0[x] > 1.0 ) ? r0[x] - 1 : r0[x];
//          r0[y] = ( r0[y] > 1.0 ) ? r0[y] - 1 : r0[y];
         r0[x] = get_interpolated_value(adv_array_,
                                        r0[y] * (adv_array_.dim1() - 2),
                                        r0[x] * (adv_array_.dim2() - 2));
        conv_array_(i, j, x) = x_coord;
        conv_array_(i, j, y) = y_coord;
        conv_array_(i, j, z) = r0[x] * scale;
        conv_array_(i, j, w) = scale;
      }
    }
    conv_is_initialized_ = true;
    conv_dirty_ = true;
  }
}

void
FlowRenderer2D::load_conv()
{
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
  // This texture is not loaded if there are no shaders.
  if (conv_dirty_)
  {
    // Update 2D texture.
    glDeleteTextures(1, &conv_tex_);
    glGenTextures(1, &conv_tex_);
    glBindTexture(GL_TEXTURE_2D, conv_tex_);
    //           glPixelStorei(GL_UNPACK_ALIGNMENT, 1); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16,
                 // Note the dim1, dim2 switch here.
                 conv_array_.dim2(), conv_array_.dim1(), 
                 0, GL_RGBA, GL_FLOAT, &conv_array_(0,0,0));
    conv_dirty_ = false;
  }
  else
  {
    glBindTexture(GL_TEXTURE_2D, conv_tex_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    // Note the dim1, dim2 switch here.
                    conv_array_.dim2(), conv_array_.dim1(), 
                    GL_RGBA, GL_FLOAT, &conv_array_(0,0,0));
  }
#endif
}

void
FlowRenderer2D::conv_accum( float pixel_x, float pixel_y, float scale)
{
  int endy = int(tex_coords_[2] * (flow_array_.dim2() - 1));
  int endx = int(tex_coords_[7] * (flow_array_.dim1() - 1));
  
  float r0[4], r1[4], noise;
  int x = 0, y = 1, z = 2, w = 3;

  for(int j = 0; j < endy; j++ ){
    for(int i = 0; i < endx; i++ ) {
      r0[x] = conv_array_( i, j, x);
      r0[y] = conv_array_( i, j, y);
      r0[z] = conv_array_( i, j, z);
      r0[w] = conv_array_( i, j, w);
      
      r1[z] = Clamp(r0[x], 0.0, 1.0);
      r1[w] = Clamp(r0[y], 0.0, 1.0);
      r1[x] = flow_array_( int(r1[w] * (flow_array_.dim1() - 1)), 
                           int(r1[z] * (flow_array_.dim2() - 1)), 0) - 0.5;
      r1[y] = flow_array_( int(r1[w] * (flow_array_.dim1() - 1)), 
                           int(r1[z] * (flow_array_.dim2() - 1)), 1) - 0.5;
      
      r1[z] = r1[x] * r1[x];
      r1[w] = r1[y] * r1[y];
      r1[z] = r1[z] + r1[w];
      r1[z] = 1.0/sqrt(r1[z]);
      r1[x] = r1[z] * r1[x];
      r1[y] = r1[z] * r1[y];
      r0[x] = Clamp(r1[x] * pixel_x + r0[x], 0.0, 1.0);
      r0[y] = Clamp(r1[y] * pixel_y + r0[y], 0.0, 1.0);
      
      noise = get_interpolated_value(adv_array_, 
                                     r0[y] * (adv_array_.dim1() - 2 ),
                                     r0[x] * (adv_array_.dim2() - 2));
      
    
      r0[w] = r0[w] + scale;
      r0[z] = noise * scale + r0[z];
      conv_array_( i, j, x ) = r0[x];      
      conv_array_( i, j, y ) = r0[y];
      conv_array_( i, j, z ) = r0[z];
      conv_array_( i, j, w ) = r0[w];
    }
  }
}

void
FlowRenderer2D::conv_rewire()
{
  float r0[4];
  int x = 0, y = 1, z = 2, w = 3;

  for(int j = 0; j < conv_array_.dim2(); j++){
    for( int i = 0; i < conv_array_.dim1(); i++){
      r0[z] = conv_array_( i, j, z);
      r0[w] = conv_array_( i, j, w);
      r0[x] = 1.0/r0[w];
      conv_array_(i, j, x) = r0[z] * r0[x];
      conv_array_(i, j, y) = r0[z] * r0[x];
      conv_array_(i, j, z) = r0[z] * r0[x];
      conv_array_(i, j, w) = 1.0;
    }
  }
}

void
FlowRenderer2D::bind_adv( int reg )
{
  NOT_FINISHED("FlowRenderer2D::bind_adv()");  
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
  if (ShaderProgramARB::shaders_supported())
  {
    glActiveTexture(GL_TEXTURE0+reg);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, adv_tex_);
    glActiveTexture(GL_TEXTURE0_ARB);
  }
  
#endif
  CHECK_OPENGL_ERROR("FlowRenderer2D::bind_adv()");
}

void
FlowRenderer2D::release_adv( int reg )
{
  NOT_FINISHED("FlowRenderer2D::release_adv()");  
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
  if (ShaderProgramARB::shaders_supported())
  {
    // bind texture to unit 2
    glActiveTexture(GL_TEXTURE0+reg);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    // enable data texture unit 0
    glActiveTexture(GL_TEXTURE0_ARB);
  }
#endif
  CHECK_OPENGL_ERROR("FlowRenderer2D::release_adv()");

}
void
FlowRenderer2D::bind_conv( int reg )
{
  NOT_FINISHED("FlowRenderer2D::bind_conv()");  
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
  if (ShaderProgramARB::shaders_supported())
  {
    glActiveTexture(GL_TEXTURE0+reg);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, conv_tex_);
    glActiveTexture(GL_TEXTURE0_ARB);
  }
  
#endif
  CHECK_OPENGL_ERROR("FlowRenderer2D::bind_conv()");
}

void
FlowRenderer2D::release_conv( int reg )
{
  NOT_FINISHED("FlowRenderer2D::release_conv()");  
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
  if (ShaderProgramARB::shaders_supported())
  {
    // bind texture to unit 2
    glActiveTexture(GL_TEXTURE0+reg);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    // enable data texture unit 0
    glActiveTexture(GL_TEXTURE0_ARB);
  }
#endif
  CHECK_OPENGL_ERROR("FlowRenderer2D::release_conv()");

}


void
FlowRenderer2D::build_noise( float scale, float shftx, float shfty )
{
  if(build_noise_){
    noise_array_.resize(buffer_height_, buffer_width_, 4);
    srand48( noise_array_.dim1() );
    for(int i = 0; i < noise_array_.dim1(); i++){
      for(int j = 0; j < noise_array_.dim2(); j++){
        float val = drand48();
        noise_array_(i,j,0) = val;
        noise_array_(i,j,1) = val;
        noise_array_(i,j,2) = val;
        noise_array_(i,j,3) = 1.0;
      }
    }

    if( use_pbuffer_ ){
      noise_buffer_->activate();
      glDrawBuffer(GL_FRONT);
      glViewport(0, 0, noise_buffer_->width(), noise_buffer_->height());
      glClearColor(0.0, 0.0, 0.0, 0.0);
      glClear(GL_COLOR_BUFFER_BIT);
      noise_buffer_->swapBuffers();
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glTranslatef(-1.0, -1.0, 0.0);
      glScalef(2.0, 2.0, 2.0);
      glDisable(GL_DEPTH_TEST);
      glDisable(GL_LIGHTING);
      glDisable(GL_CULL_FACE);
      glDisable(GL_BLEND);
      glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

      // Bind Textures
      
      //-----------------------------------------------------
      // set up shader
      FragmentProgramARB* shader  = adv_init_;
      //-----------------------------------------------------
      if( shader ){
        if(!shader->valid()) {
          shader->create();
          shader->setLocalParam(1, scale, 1.0, 1.0, 1.0);
          shader->setLocalParam(2, shftx, shfty, 0.5, 0.5);
        }
        shader->bind();
      }

      float tex_coords[] = { 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 };
      float pos_coords[] = { 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 };

      glBegin( GL_QUADS );
      {
        for (int i = 0; i < 4; i++) {
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader) 
          glMultiTexCoord2fv(GL_TEXTURE1,tex_coords+i*2);
#else
          glTexCoord2fv(tex_coords+i*2);
#endif
          glVertex2fv(pos_coords+i*2);
        }
      }
      glEnd();
      noise_buffer_->release(GL_FRONT);
      if(shader)
        shader->release();
      noise_buffer_->swapBuffers();
      noise_buffer_->deactivate();
      noise_buffer_->set_use_texture_matrix(true);
    } else {
      // software 
      //-------------------------------------------------------
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
      // This texture is not used if there is no shaders.
      // glColorTable is used instead.
      if (ShaderProgramARB::shaders_supported())
      {
        // Update 2D texture.
        glDeleteTextures(1, &noise_tex_);
        glGenTextures(1, &noise_tex_);
        glBindTexture(GL_TEXTURE_2D, noise_tex_);
        //           glPixelStorei(GL_UNPACK_ALIGNMENT, 1); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16,
                     // Note the dim1, dim2 switch here.
                     noise_array_.dim2(), noise_array_.dim1(), 
                     0, GL_RGBA, GL_FLOAT, &noise_array_(0,0,0));
      }
      else
      {
        glBindTexture(GL_TEXTURE_2D, noise_tex_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        // Note the dim1, dim2 switch here.
                        noise_array_.dim2(), noise_array_.dim1(), 
                        GL_RGBA, GL_FLOAT, &noise_array_(0,0,0));
      }
  
#endif
      //-------------------------------------------------------
      build_noise_ = false;
      NOT_FINISHED("FlowRenderer2D::build_noise()");  
    }
    CHECK_OPENGL_ERROR("FlowRenderer2D::build_noise()");


  }
}

void
FlowRenderer2D::bind_noise( int reg)
{
  NOT_FINISHED("FlowRenderer2D::bind_noise()");  
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
  if (ShaderProgramARB::shaders_supported())
  {
    glActiveTexture(GL_TEXTURE0+reg);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, noise_tex_);
    glActiveTexture(GL_TEXTURE0_ARB);
  }
  
#endif
  CHECK_OPENGL_ERROR("FlowRenderer2D::bind_noise()");
}
void
FlowRenderer2D::release_noise( int reg )
{
  NOT_FINISHED("FlowRenderer2D::release_noise()");  
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
  if (ShaderProgramARB::shaders_supported())
  {
    // bind texture to unit 2
    glActiveTexture(GL_TEXTURE0+reg);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    // enable data texture unit 0
    glActiveTexture(GL_TEXTURE0_ARB);
  }
#endif
  CHECK_OPENGL_ERROR("FlowRenderer2D::release_noise()");

}

void
FlowRenderer2D::create_pbuffers(int w, int h)
{
  int psize[2];
  psize[0] = NextPowerOf2(w);
  psize[1] = NextPowerOf2(h);
  if(!adv_buffer_ && use_pbuffer_) {
     adv_buffer_ = new Pbuffer( psize[0], psize[1], GL_FLOAT, 
                                 32, true, GL_FALSE);
     noise_buffer_ = new Pbuffer( psize[0], psize[1], GL_FLOAT, 
                                  32, true, GL_FALSE);
    CHECK_OPENGL_ERROR("");    
    if(!adv_buffer_->create() || !noise_buffer_->create() ) {
      NOT_FINISHED("Something wrong with pbuffers"); 
      adv_buffer_->destroy();
      noise_buffer_->destroy();
      delete adv_buffer_;
      delete noise_buffer_;
      adv_buffer_ = 0;
      noise_buffer_ = 0;
      use_pbuffer_ = false;
      return;
    } else {
      adv_buffer_->set_use_default_shader(false);
    }
  }
  pbuffers_created_ = true;
}


void
FlowRenderer2D::next_shift(int *shft)
{
  int shift = *shft+1;
  *shft = ( shift % shift_list_.size());
}

#endif // SCI_OPENGL


} // namespace SCIRun
