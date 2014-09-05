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
//    File   : SliceRenderer.cc
//    Author : Milan Ikits
//    Date   : Wed Jul  7 23:37:16 2004


#include <sci_defs/ogl_defs.h>
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <Core/Geom/DrawInfoOpenGL.h>
#include <Core/Volume/SliceRenderer.h>
#include <Core/Volume/VolShader.h>
#include <Core/Volume/TextureBrick.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Math/Trig.h> // for M_PI

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <string>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using std::endl;
using std::string;

namespace SCIRun {

#ifndef GL_TEXTURE_3D
#define GL_TEXTURE_3D 0x806F
#endif

#ifdef _WIN32
#undef min
#undef max
#endif

SliceRenderer::SliceRenderer(TextureHandle tex,
                             ColorMapHandle cmap1, 
			     vector<ColorMap2Handle> &cmap2,
                             int tex_mem)
  : TextureRenderer(tex, cmap1, cmap2, tex_mem),
    control_point_(Point(0,0,0)),
    draw_x_(false),
    draw_y_(false),
    draw_z_(false),
    draw_view_(false),
    draw_phi0_(false),
    phi0_(0),
    draw_phi1_(false),
    phi1_(0),
    draw_cyl_(false),
    draw_level_outline_(false),
    draw_level_(20)
{
  lighting_ = 1;
  mode_ = MODE_SLICE;
  outline_colors_.resize(20);
  for(int i = 0; i < 20; i++){
    outline_colors_[i] = Color(1.0,1.0, 1.0);
  }
}

SliceRenderer::SliceRenderer(const SliceRenderer& copy)
  : TextureRenderer(copy),
    control_point_(copy.control_point_),
    draw_x_(copy.draw_x_),
    draw_y_(copy.draw_y_),
    draw_z_(copy.draw_x_),
    draw_view_(copy.draw_view_),
    draw_phi0_(copy.phi0_),
    phi0_(copy.phi0_),
    draw_phi1_(copy.draw_phi1_),
    phi1_(copy.phi1_),
    draw_cyl_(copy.draw_cyl_),
    draw_level_outline_(copy.draw_level_outline_),
    draw_level_(copy.draw_level_)
{
  lighting_ = 1;
}

SliceRenderer::~SliceRenderer()
{}

GeomObj*
SliceRenderer::clone()
{
  return scinew SliceRenderer(*this);
}

void 
SliceRenderer::set_outline_colors( vector< Color >& colors )
{
  if( outline_colors_.size() != colors.size() ){
    outline_colors_.resize( colors.size() );
    outline_colors_.clear();
  }

  for(unsigned int i = 0; i < colors.size(); i++){
    //reverse them
    outline_colors_[i] = colors[(colors.size() - 1) - i];
  }
}


#ifdef SCI_OPENGL
void
SliceRenderer::draw(DrawInfoOpenGL* di, Material* mat, double)
{
  //AuditAllocator(default_allocator);
  if(!pre_draw(di, mat, lighting_)) return;
  mutex_.lock();
  di_ = di;
  if(di->get_drawtype() == DrawInfoOpenGL::WireFrame) {
    draw_wireframe();
  } else {
    draw_slice();
  }
  di_ = 0;
  mutex_.unlock();
}

void
SliceRenderer::draw_slice()
{
  if( tex_->nlevels() > 1 ){
    multi_level_draw();
    return;
  }
  
  tex_->lock_bricks();
  
  Ray view_ray = compute_view();
  vector<TextureBrickHandle> bricks;
  tex_->get_sorted_bricks(bricks, view_ray);
  if(bricks.size() == 0) return;

  //--------------------------------------------------------------------------

  const int nc = tex_->nc();
  const int nb0 = tex_->nb(0);
  const bool use_cmap1 = cmap1_.get_rep();
  const bool use_cmap2 =
    cmap2_.size() && nc == 2 && ShaderProgramARB::shaders_supported();
  if (!use_cmap1 && !use_cmap2)
  {
    tex_->unlock_bricks();
    return;
  }
  GLboolean use_fog = glIsEnabled(GL_FOG);
  
  //--------------------------------------------------------------------------
  // load colormap texture
  if(use_cmap2) {
    // rebuild if needed
    build_colormap2();
    bind_colormap2();
  } else {
    // rebuild if needed
//     build_colormap1();
//     bind_colormap1();
    build_colormap1(cmap1_array_, cmap1_tex_, cmap1_dirty_, alpha_dirty_);
    bind_colormap1(cmap1_array_, cmap1_tex_);

  }
  
  //--------------------------------------------------------------------------
  // enable data texture unit 0
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#ifdef _WIN32
  if (glActiveTexture)
#endif
  glActiveTexture(GL_TEXTURE0_ARB);
#endif
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glEnable(GL_TEXTURE_3D);

  //--------------------------------------------------------------------------
  // enable alpha test
  glEnable(GL_ALPHA_TEST);
  glAlphaFunc(GL_GREATER, 0.0);
  glColor4f(1.0, 1.0, 1.0, 1.0);
  glDepthMask(GL_TRUE);

  glEnable(GL_BLEND);
  glBlendEquation(GL_FUNC_ADD);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

  //--------------------------------------------------------------------------
  // set up shaders
  FragmentProgramARB* shader = 0;
  int blend_mode = 0;
  shader = vol_shader_factory_->shader(use_cmap2 ? 2 : 1, nb0, tex_->nc(),
                                       false, true,
                                       use_fog, blend_mode, cmap2_.size());

  if(shader) {
    if(!shader->valid()) {
      shader->create();
    }
    shader->bind();
  }

  //--------------------------------------------------------------------------
  // render bricks

  vector<float> vertex;
  vector<float> texcoord;
  vector<int> size;
  
  Transform tform = tex_->transform();
  double mvmat[16];
  tform.get_trans(mvmat);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);
  
  for(unsigned int i=0; i<bricks.size(); i++) {
    double t;
    TextureBrickHandle b = bricks[i];
    if (!test_against_view(b->bbox())) continue; // Clip brick against view.

    vertex.clear();
    texcoord.clear();
    size.clear();
    const Point view = view_ray.origin();
    const Point &bmin = b->bbox().min();
    const Point &bmax = b->bbox().max();
    const Point &bmid = b->bbox().center();
    const Point c(control_point_);
    bool draw_z = false;
    if(draw_cyl_) {
      const double to_rad = M_PI / 180.0;
      BBox bb;
      tex_->get_bounds(bb);
      Point cyl_mid = bb.center();
      if(draw_phi0_) {
	Vector phi(1.,0,0);
	Transform rot;
	rot.pre_rotate(phi0_ * to_rad, Vector(0,0,1.));
	phi = rot.project(phi);
	Ray r(cyl_mid, phi);
        r.planeIntersectParameter(-r.direction(), control_point_, t);
        b->compute_polygon(r, t, vertex, texcoord, size);
      }
      if(draw_phi1_) {
	Vector phi(1.,0,0);
	Transform rot;
	rot.pre_rotate(phi1_ * to_rad, Vector(0,0,1.));
	phi = rot.project(phi);
	Ray r(cyl_mid, phi);
        r.planeIntersectParameter(-r.direction(), control_point_, t);
        b->compute_polygon(r, t, vertex, texcoord, size);
      }
      if(draw_z_) {
        draw_z = true;
      }
    } else {
      if(draw_view_) {
        view_ray.planeIntersectParameter(-view_ray.direction(), control_point_, t);
        b->compute_polygon(view_ray, t, vertex, texcoord, size);
      } else {
	if(draw_x_) {
	  Point o(bmin.x(), bmid.y(), bmid.z());
	  Vector v(c.x() - o.x(), 0,0);
	  if(c.x() > bmin.x() && c.x() < bmax.x() ){
	    if(view.x() > c.x()) {
	      o.x(bmax.x());
	      v.x(c.x() - o.x());
	    } 
	    Ray r(o,v);
            r.planeIntersectParameter(-r.direction(), control_point_, t);
            b->compute_polygon(r, t, vertex, texcoord, size);
	  }
	}
	if(draw_y_) {
	  Point o(bmid.x(), bmin.y(), bmid.z());
	  Vector v(0, c.y() - o.y(), 0);
	  if(c.y() > bmin.y() && c.y() < bmax.y() ){
	    if(view.y() > c.y()) {
	      o.y(bmax.y());
	      v.y(c.y() - o.y());
	    } 
	    Ray r(o,v);
            r.planeIntersectParameter(-r.direction(), control_point_, t);
            b->compute_polygon(r, t, vertex, texcoord, size);
	  }
	}
        if(draw_z_) {
          draw_z = true;
        }
      }
    }
    
    if (draw_z) {
      Point o(bmid.x(), bmid.y(), bmin.z());
      Vector v(0, 0, c.z() - o.z());
      if(c.z() > bmin.z() && c.z() < bmax.z() ) {
	if(view.z() > c.z()) {
	  o.z(bmax.z());
	  v.z(c.z() - o.z());
	} 
	Ray r(o,v);
        r.planeIntersectParameter(-r.direction(), control_point_, t);
        b->compute_polygon(r, t, vertex, texcoord, size);
      }
    }

    if (size.size()) // Only load brick if there is something to draw.
    {
      load_brick(bricks, i, use_cmap2);
      draw_polygons(vertex, texcoord, size, true, use_fog, 0);
    }
  }

  glPopMatrix();
  
  //--------------------------------------------------------------------------
  // release shaders

  if(shader && shader->valid())
    shader->release();

  //--------------------------------------------------------------------------

  glDisable(GL_BLEND);
  
  glDisable(GL_ALPHA_TEST);
  glDepthMask(GL_TRUE);

  if(use_cmap2) {
    release_colormap2();
  } else {
    release_colormap1();
  }
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#ifdef _WIN32
  if (glActiveTexture)
#endif
  glActiveTexture(GL_TEXTURE0_ARB);
#endif
  glDisable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, 0);

  tex_->unlock_bricks();
}


void
SliceRenderer::multi_level_draw()
{
  tex_->lock_bricks();
  
  Ray view_ray = compute_view();
  vector<TextureBrickHandle> bricks;
  tex_->get_sorted_bricks(bricks, view_ray);

  int levels = tex_->nlevels();

  //--------------------------------------------------------------------------

  const int nc = tex_->nc();
  const int nb0 = tex_->nb(0);
  const bool use_cmap1 = cmap1_.get_rep();
  const bool use_cmap2 =
    cmap2_.size() && nc == 2 && ShaderProgramARB::shaders_supported();
  if (!use_cmap1 && !use_cmap2)
  {
    tex_->unlock_bricks();
    return;
  }
  
  GLboolean use_fog = glIsEnabled(GL_FOG);
  
  //--------------------------------------------------------------------------
  // load colormap texture
  if(use_cmap2) {
    // rebuild if needed
    build_colormap2();
    bind_colormap2();
  } else {
    // rebuild if needed
    build_colormap1(cmap1_array_, cmap1_tex_, cmap1_dirty_, alpha_dirty_);
    bind_colormap1(cmap1_array_, cmap1_tex_);
  }
  
  //--------------------------------------------------------------------------
  // enable data texture unit 0
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#ifdef _WIN32
  if (glActiveTexture)
#endif
  glActiveTexture(GL_TEXTURE0_ARB);
#endif  
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glEnable(GL_TEXTURE_3D);

  //--------------------------------------------------------------------------
  // enable alpha test
  glEnable(GL_ALPHA_TEST);
  glAlphaFunc(GL_GREATER, 0.0);
  glColor4f(1.0, 1.0, 1.0, 1.0);
  glDepthMask(GL_TRUE);

  //--------------------------------------------------------------------------
  // set up shaders
  FragmentProgramARB* shader = 0;
  int blend_mode = 0;
  shader = vol_shader_factory_->shader(use_cmap2 ? 2 : 1, nb0, tex_->nc(),
                                       false, true,
                                       use_fog, blend_mode, cmap2_.size());

  if(shader) {
    if(!shader->valid()) {
      shader->create();
    }
    shader->bind();
  }

  //-------------------------------------------------------------------------

  // set up stenciling
  if(use_stencil_){
    glClearStencil(0);
    glStencilMask(1);
    glStencilFunc(GL_EQUAL, 0, 1);
    glStencilOp(GL_KEEP, GL_KEEP,GL_INCR);
    glEnable(GL_STENCIL_TEST);
  } 

  //--------------------------------------------------------------------------
  // render bricks

  vector<float> vertex;
  vector<float> texcoord;
  vector<int> size;
  
  Transform tform = tex_->transform();
  double mvmat[16];
  tform.get_trans(mvmat);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);
  
  vector<vector<TextureBrickHandle> > blevels;
  blevels.resize(levels);
  for(int k = levels - 1; k >= 0;  --k ){
    tex_->get_sorted_bricks(blevels[levels - (k + 1)], view_ray, k);
  }
  if( use_stencil_){
    glStencilMask(~(GLuint(0)));
    glClear(GL_STENCIL_BUFFER_BIT);
    glStencilMask(1);
  }
//    for(unsigned int j = 0; j < levels; ++j ){
  bool draw_z = false;
  //  double t;
  if(draw_cyl_) {
    const double to_rad = M_PI / 180.0;
    BBox bb;
    tex_->get_bounds(bb);
    Point cyl_mid = bb.center();
    if(draw_phi0_) {
      Vector phi(1.,0,0);
      Transform rot;
      rot.pre_rotate(phi0_ * to_rad, Vector(0,0,1.));
      phi = rot.project(phi);
      Ray r(cyl_mid, phi);

      if( use_stencil_){
	glStencilMask(~(GLuint(0)));
	glClear(GL_STENCIL_BUFFER_BIT);
	glStencilMask(1);
      }

      for(int j = 0; j < levels; ++j ){
	if(!draw_level_[j]) continue;
	vector<TextureBrickHandle>& bs  = blevels[j];
	for(unsigned int i=0; i<bs.size(); i++) {
	  double t;
	  TextureBrickHandle b = bs[i];
	  load_brick(bs, i, use_cmap2);
	  vertex.resize(0);
	  texcoord.resize(0);
	  size.resize(0);
	  draw_z = false;
	  r.planeIntersectParameter(-r.direction(), control_point_, t);
	  b->compute_polygon(r, t, vertex, texcoord, size);
	}
	draw_polygons(vertex, texcoord, size, true, use_fog, 0);
        draw_level_outline(vertex, size, use_fog, j, shader);
      }
    }
    if(draw_phi1_) {
      Vector phi(1.,0,0);
      Transform rot;
      rot.pre_rotate(phi1_ * to_rad, Vector(0,0,1.));
      phi = rot.project(phi);
      Ray r(cyl_mid, phi);

      if( use_stencil_){
	glStencilMask(~(GLuint(0)));
	glClear(GL_STENCIL_BUFFER_BIT);
	glStencilMask(1);
      }

      for(int j = 0; j < levels; ++j ){
	if(!draw_level_[j]) continue;
	vector<TextureBrickHandle>& bs  = blevels[j];
	
	for(unsigned int i=0; i<bs.size(); i++) {
	  double t;
	  TextureBrickHandle b = bs[i];
	  load_brick(bs, i, use_cmap2);
	  vertex.resize(0);
	  texcoord.resize(0);
	  size.resize(0);
	  //Point c(control_point_);
	  draw_z = false;
	  r.planeIntersectParameter(-r.direction(), control_point_, t);
	  b->compute_polygon(r, t, vertex, texcoord, size);
	}
	draw_polygons(vertex, texcoord, size, true, use_fog, 0);
        draw_level_outline(vertex, size, use_fog, j, shader);
      }
    }
    if(draw_z_) {
      draw_z = true;
    }
  } else {
    if(draw_view_) {
      if( use_stencil_){
	glStencilMask(~(GLuint(0)));
	glClear(GL_STENCIL_BUFFER_BIT);
	glStencilMask(1);
      }

      for(int j = 0; j < levels; ++j ){
	if(!draw_level_[j]) continue;
	vector<TextureBrickHandle>& bs  = blevels[j];
	    
	for(unsigned int i=0; i<bs.size(); i++) {
	  double t;
	  TextureBrickHandle b = bs[i];
	  load_brick(bs, i, use_cmap2);
	  vertex.resize(0);
	  texcoord.resize(0);
	  size.resize(0);
	  view_ray.planeIntersectParameter(-view_ray.direction(),
					   control_point_, t);
	  draw_z = false;
	  b->compute_polygon(view_ray, t, vertex, texcoord, size);
	}
	draw_polygons(vertex, texcoord, size, true, use_fog, 0);
        draw_level_outline(vertex, size, use_fog, j, shader);
      }
      
    } else {
      if(draw_x_) {
	if( use_stencil_){
	  glStencilMask(~(GLuint(0)));
	  glClear(GL_STENCIL_BUFFER_BIT);
	  glStencilMask(1);
	}

	for(int j = 0; j < levels; ++j ){
	  if(!draw_level_[j]) continue;
	  vector<TextureBrickHandle>& bs  = blevels[j];
	  
	  for(unsigned int i=0; i<bs.size(); i++) {
	    double t;
	    TextureBrickHandle b = bs[i];
	    load_brick(bs, i, use_cmap2);
	    vertex.resize(0);
	    texcoord.resize(0);
	    size.resize(0);
	    Point view = view_ray.origin();
	    const Point &bmin = b->bbox().min();
	    const Point &bmax = b->bbox().max();
	    const Point &bmid = b->bbox().center();
	    Point c(control_point_);
	    Point o( bmin.x(), bmid.y(), bmid.z());
	    Vector v(c.x() - o.x(), 0,0);
	    if(c.x() > bmin.x() && c.x() < bmax.x() ){
	      if(view.x() > c.x()) {
		o.x(bmax.x());
		v.x(c.x() - o.x());
	      } 
	      Ray r(o,v);
	      r.planeIntersectParameter(-r.direction(), control_point_, t);
	      
	      draw_z = false;
	      b->compute_polygon(r, t, vertex, texcoord, size);
	    }
	    draw_polygons(vertex, texcoord, size, true, use_fog, 0);
            draw_level_outline(vertex, size, use_fog, j, shader);
	  }
	}
      }
      if(draw_y_) {
	if( use_stencil_){
	  glStencilMask(~(GLuint(0)));
	  glClear(GL_STENCIL_BUFFER_BIT);
	  glStencilMask(1);
	}

	for(int j = 0; j < levels; ++j ){
	  if(!draw_level_[j]) continue;
	  vector<TextureBrickHandle>& bs  = blevels[j];

	  for(unsigned int i=0; i<bs.size(); i++) {
	    double t;
	    TextureBrickHandle b = bs[i];
	    load_brick(bs, i, use_cmap2);
	    vertex.resize(0);
	    texcoord.resize(0);
	    size.resize(0);
	    Point view = view_ray.origin();
	    const Point &bmin = b->bbox().min();
	    const Point &bmax = b->bbox().max();
	    const Point &bmid = b->bbox().center();
	    Point c(control_point_);
	    Point o(bmid.x(), bmin.y(), bmid.z());
	    Vector v(0, c.y() - o.y(), 0);
	    if(c.y() > bmin.y() && c.y() < bmax.y() ){
	      if(view.y() > c.y()) {
		o.y(bmax.y());
		v.y(c.y() - o.y());
	      } 
	      Ray r(o,v);
	      r.planeIntersectParameter(-r.direction(), control_point_, t);
	      draw_z = false;
	      b->compute_polygon(r, t, vertex, texcoord, size);
	    }
            draw_polygons(vertex, texcoord, size, true, use_fog, 0);
            draw_level_outline(vertex, size, use_fog, j, shader);
	  }
	}
      }
      if(draw_z_) {
	draw_z = true;
      }
    }
  }
    
  if (draw_z) {
    if( use_stencil_){
      glStencilMask(~(GLuint(0)));
      glClear(GL_STENCIL_BUFFER_BIT);
      glStencilMask(1);
    }

    for(int j = 0; j < levels; ++j ){
      if(!draw_level_[j]) continue;
      vector<TextureBrickHandle>& bs  = blevels[j];

      for(unsigned int i=0; i<bs.size(); i++) {
	double t;
	TextureBrickHandle b = bs[i];
	load_brick(bs, i, use_cmap2);
	vertex.resize(0);
	texcoord.resize(0);
	size.resize(0);
	Point view = view_ray.origin();
	const Point &bmin = b->bbox().min();
	const Point &bmax = b->bbox().max();
	const Point &bmid = b->bbox().center();
	Point c(control_point_);
	Point o(bmid.x(), bmid.y(), bmin.z());
	Vector v(0, 0, c.z() - o.z());
	if(c.z() > bmin.z() && c.z() < bmax.z() ) {
	  if(view.z() > c.z()) {
	    o.z(bmax.z());
	    v.z(c.z() - o.z());
	  } 
	  draw_z = false;
	  Ray r(o,v);
	  r.planeIntersectParameter(-r.direction(), control_point_, t);
	  b->compute_polygon(r, t, vertex, texcoord, size);
	}
 	draw_polygons(vertex, texcoord, size, true, use_fog, 0);
        draw_level_outline(vertex, size, use_fog, j, shader);
      }
    }
  }

  glPopMatrix();
  
  //-------------------------------------------------------------------------
  // turn off stenciling
  if(use_stencil_)
    glDisable(GL_STENCIL_TEST);

  //--------------------------------------------------------------------------
  // release shaders
  if(shader && shader->valid())
    shader->release();

  //--------------------------------------------------------------------------
  glDisable(GL_ALPHA_TEST);
  glDepthMask(GL_TRUE);

  if(use_cmap2) {
    release_colormap2();
  } else {
    release_colormap1();
  }
#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#ifdef _WIN32
  if (glActiveTexture)
#endif
  glActiveTexture(GL_TEXTURE0_ARB);
#endif
  glDisable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, 0);

  tex_->unlock_bricks();
}



void 
SliceRenderer::draw_wireframe()
{
  tex_->lock_bricks();
  Ray view_ray = compute_view();
  Transform tform = tex_->transform();
  double mvmat[16];
  tform.get_trans(mvmat);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);
  GLboolean lighting = glIsEnabled(GL_LIGHTING);
  glDisable(GL_LIGHTING);
  vector<TextureBrickHandle> bricks;
  tex_->get_sorted_bricks(bricks, view_ray);

  vector<float> vertex;
  vector<float> texcoord;
  vector<int> size;

  for (unsigned int i=0; i<bricks.size(); i++)
  {
    glColor4f(0.8, 0.8, 0.8, 1.0);

    TextureBrickHandle b = bricks[i];
    const Point &pmin(b->bbox().min());
    const Point &pmax(b->bbox().max());
    Point corner[8];
    corner[0] = pmin;
    corner[1] = Point(pmin.x(), pmin.y(), pmax.z());
    corner[2] = Point(pmin.x(), pmax.y(), pmin.z());
    corner[3] = Point(pmin.x(), pmax.y(), pmax.z());
    corner[4] = Point(pmax.x(), pmin.y(), pmin.z());
    corner[5] = Point(pmax.x(), pmin.y(), pmax.z());
    corner[6] = Point(pmax.x(), pmax.y(), pmin.z());
    corner[7] = pmax;

    glBegin(GL_LINES);
    {
      for(int i=0; i<4; i++) {
        glVertex3d(corner[i].x(), corner[i].y(), corner[i].z());
        glVertex3d(corner[i+4].x(), corner[i+4].y(), corner[i+4].z());
      }
    }
    glEnd();
    glBegin(GL_LINE_LOOP);
    {
      glVertex3d(corner[0].x(), corner[0].y(), corner[0].z());
      glVertex3d(corner[1].x(), corner[1].y(), corner[1].z());
      glVertex3d(corner[3].x(), corner[3].y(), corner[3].z());
      glVertex3d(corner[2].x(), corner[2].y(), corner[2].z());
    }
    glEnd();
    glBegin(GL_LINE_LOOP);
    {
      glVertex3d(corner[4].x(), corner[4].y(), corner[4].z());
      glVertex3d(corner[5].x(), corner[5].y(), corner[5].z());
      glVertex3d(corner[7].x(), corner[7].y(), corner[7].z());
      glVertex3d(corner[6].x(), corner[6].y(), corner[6].z());
    }
    glEnd();

    glColor4f(0.4, 0.4, 0.4, 1.0);

    vertex.clear();
    texcoord.clear();
    size.clear();

    const Point view = view_ray.origin();
    const Point &bmin = b->bbox().min();
    const Point &bmax = b->bbox().max();
    const Point &bmid = b->bbox().center();
    const Point c(control_point_);
    double t;
    bool draw_z = false;
    if(draw_cyl_) {
      const double to_rad = M_PI / 180.0;
      BBox bb;
      tex_->get_bounds(bb);
      Point cyl_mid = bb.center();
      if(draw_phi0_) {
	Vector phi(1.,0,0);
	Transform rot;
	rot.pre_rotate(phi0_ * to_rad, Vector(0,0,1.));
	phi = rot.project(phi);
	Ray r(cyl_mid, phi);
        r.planeIntersectParameter(-r.direction(), control_point_, t);
        b->compute_polygon(r, t, vertex, texcoord, size);
      }
      if(draw_phi1_) {
	Vector phi(1.,0,0);
	Transform rot;
	rot.pre_rotate(phi1_ * to_rad, Vector(0,0,1.));
	phi = rot.project(phi);
	Ray r(cyl_mid, phi);
        r.planeIntersectParameter(-r.direction(), control_point_, t);
        b->compute_polygon(r, t, vertex, texcoord, size);
      }
      if(draw_z_) {
        draw_z = true;
      }
    } else {
      if(draw_view_) {
        view_ray.planeIntersectParameter(-view_ray.direction(), control_point_, t);
        b->compute_polygon(view_ray, t, vertex, texcoord, size);
      } else {
	if(draw_x_) {
	  Point o(bmin.x(), bmid.y(), bmid.z());
	  Vector v(c.x() - o.x(), 0,0);
	  if(c.x() > bmin.x() && c.x() < bmax.x() ){
	    if(view.x() > c.x()) {
	      o.x(bmax.x());
	      v.x(c.x() - o.x());
	    } 
	    Ray r(o,v);
            r.planeIntersectParameter(-r.direction(), control_point_, t);
            b->compute_polygon(r, t, vertex, texcoord, size);
	  }
	}
	if(draw_y_) {
	  Point o(bmid.x(), bmin.y(), bmid.z());
	  Vector v(0, c.y() - o.y(), 0);
	  if(c.y() > bmin.y() && c.y() < bmax.y() ){
	    if(view.y() > c.y()) {
	      o.y(bmax.y());
	      v.y(c.y() - o.y());
	    } 
	    Ray r(o,v);
            r.planeIntersectParameter(-r.direction(), control_point_, t);
            b->compute_polygon(r, t, vertex, texcoord, size);
	  }
	}
        if(draw_z_) {
          draw_z = true;
        }
      }
    }
    
    if (draw_z) {
      Point o(bmid.x(), bmid.y(), bmin.z());
      Vector v(0, 0, c.z() - o.z());
      if(c.z() > bmin.z() && c.z() < bmax.z() ) {
	if(view.z() > c.z()) {
	  o.z(bmax.z());
	  v.z(c.z() - o.z());
	} 
	Ray r(o,v);
        r.planeIntersectParameter(-r.direction(), control_point_, t);
        b->compute_polygon(r, t, vertex, texcoord, size);
      }
    }

    draw_polygons_wireframe(vertex, texcoord, size, false, false, 0);
  }

  if(lighting) glEnable(GL_LIGHTING);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  tex_->unlock_bricks();
}

#endif // SCI_OPENGL

void SliceRenderer::draw_level_outline(vector<float>& vertex,
                                       vector<int>& poly,
                                       bool use_fog,
                                       int color_index,
                                       FragmentProgramARB* shader)
{
  if( draw_level_outline_ ){

    if(shader) shader->release(); // shader messes with line drawing. why?
    Color co = outline_colors_[color_index];
    glColor3f(co.r(), co.g(), co.b());

    if(use_stencil_){
      glDisable(GL_STENCIL_TEST);
    }

    GLboolean lighting = glIsEnabled(GL_LIGHTING);
    glDisable(GL_LIGHTING);
    // We don't need these so pass in dummy ones.
    vector<float> texcoord;

    // If we don't disable the texturing we won't get the color we
    // specify here.
    GLint n_tex_regs = 1;
    // first is multi-texturing enabled?
#ifndef _WIN32
    bool multitex = gluCheckExtension((GLubyte*)"GL_ARB_multitexture",
                                      glGetString(GL_EXTENSIONS));
#else
    bool multitex = (glActiveTexture != 0);
#endif
    if( multitex ) //if so, how many texture units?
#if defined(__sgi)
      n_tex_regs = 1;  // <- DON'T KNOW IF THIS IS RIGHT!!! But it
                       //    allows the SGIs to compile.  Kurt (I think)
                       //    please fix!  -Dav
#else      
      glGetIntegerv(GL_MAX_TEXTURE_UNITS_ARB, &n_tex_regs);
#endif


#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
    GLboolean *tex1d = new GLboolean[n_tex_regs];
    GLboolean *tex2d = new GLboolean[n_tex_regs];
    GLboolean *tex3d = new GLboolean[n_tex_regs];
    if(multitex){ // for each texture unit, mark what is enabled
      for( int i = 0; i < n_tex_regs; i++){
#ifdef _WIN32
        if (glActiveTexture)
#endif
        glActiveTexture(GL_TEXTURE0+i);
        tex1d[i] = glIsEnabled(GL_TEXTURE_1D);
        glDisable(GL_TEXTURE_1D);
        tex2d[i] = glIsEnabled(GL_TEXTURE_2D);
        glDisable(GL_TEXTURE_2D);
        tex3d[i] = glIsEnabled(GL_TEXTURE_3D);
        glDisable(GL_TEXTURE_3D);
      }
    } else {
      tex1d[0] = glIsEnabled(GL_TEXTURE_1D);
      glDisable(GL_TEXTURE_1D);
      tex2d[0] = glIsEnabled(GL_TEXTURE_2D);
      glDisable(GL_TEXTURE_2D);
      tex3d[0] = glIsEnabled(GL_TEXTURE_3D);
      glDisable(GL_TEXTURE_3D);
    }
#endif
    //    glColor4f(0.8,0.8,0.8,1.0);
    draw_polygons_wireframe(vertex, texcoord, poly, true, use_fog, 0);

#if defined(GL_ARB_fragment_program) || defined(GL_ATI_fragment_shader)
#ifdef _WIN32
    if (glActiveTexture) {
#endif
    // Renable the texturing.
    if(multitex){ // re-enable textures for each texture unit.
      for(int i = 0; i < n_tex_regs; i++){
        glActiveTexture(GL_TEXTURE0+i);
        if(tex1d[i]) glEnable(GL_TEXTURE_1D);
        if(tex2d[i]) glEnable(GL_TEXTURE_2D);
        if(tex3d[i]) glEnable(GL_TEXTURE_3D);
      }
    } else {
      if(tex1d[0]) glEnable(GL_TEXTURE_1D);
      if(tex2d[0]) glEnable(GL_TEXTURE_2D);
      if(tex3d[0]) glEnable(GL_TEXTURE_3D);
    }

    delete [] tex1d;
    delete [] tex2d;
    delete [] tex3d;
#ifdef _WIN32
    }
#endif
#endif

    if( lighting ) glEnable(GL_LIGHTING);
    if( use_stencil_ ) glEnable(GL_STENCIL_TEST);

    if(shader) shader->bind();   
    glColor4f(1.0,1.0,1.0,1.0);
  }
}

} // namespace SCIRun
