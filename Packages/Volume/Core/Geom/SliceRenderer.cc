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

#include <string>
#include <Core/Geom/GeomOpenGL.h>
#include <Packages/Volume/Core/Geom/SliceRenderer.h>
#include <Packages/Volume/Core/Geom/VolShader.h>
#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Packages/Volume/Core/Util/SliceTable.h>
#include <Packages/Volume/Core/Util/ShaderProgramARB.h>

using std::string;
using SCIRun::DrawInfoOpenGL;

namespace Volume {

SliceRenderer::SliceRenderer(TextureHandle tex,
                             ColorMapHandle cmap1, Colormap2Handle cmap2):
  TextureRenderer(tex, cmap1, cmap2),
  control_point_(Point(0,0,0)),
  draw_x_(false),
  draw_y_(false),
  draw_z_(false),
  draw_view_(false),
  draw_phi0_(false),
  phi0_(0),
  draw_phi1_(false),
  phi1_(0),
  draw_cyl_(false)
{
  lighting_ = 1;
  mode_ = MODE_SLICE;
}

SliceRenderer::SliceRenderer(const SliceRenderer& copy ) :
  TextureRenderer(copy.tex_, copy.cmap1_, copy.cmap2_),
  control_point_( copy.control_point_),
  draw_x_(copy.draw_x_),
  draw_y_(copy.draw_y_),
  draw_z_(copy.draw_x_),
  draw_view_(copy.draw_view_),
  draw_phi0_(copy.phi0_),
  phi0_(copy.phi0_),
  draw_phi1_(copy.draw_phi1_),
  phi1_(copy.phi1_),
  draw_cyl_(copy.draw_cyl_)
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

#ifdef SCI_OPENGL
void
SliceRenderer::draw(DrawInfoOpenGL* di, Material* mat, double)
{
  //AuditAllocator(default_allocator);
  if( !pre_draw(di, mat, lighting_) ) return;
  mutex_.lock();
  di_ = di;
  if(di->get_drawtype() == DrawInfoOpenGL::WireFrame) {
    draw_wireframe();
  } else {
    //AuditAllocator(default_allocator);
    draw();
  }
  di_ = 0;
  mutex_.unlock();
}

void
SliceRenderer::draw()
{
  Ray viewRay;
  compute_view( viewRay );

  vector<Brick*> bricks;
  tex_->get_sorted_bricks(bricks, viewRay);
  vector<Brick*>::iterator it = bricks.begin();
  vector<Brick*>::iterator it_end = bricks.end();
  BBox brickbounds;
  tex_->get_bounds(brickbounds);
  if(bricks.size() == 0) return;
  
  //--------------------------------------------------------------------------

  int nc = (*bricks.begin())->data()->nc();
  int nb0 = (*bricks.begin())->data()->nb(0);
  bool use_cmap2 = cmap2_.get_rep() && nc == 2;
  bool use_cmap1 = cmap1_.get_rep();
  GLboolean use_fog;
  glGetBooleanv(GL_FOG, &use_fog);

  if(!use_cmap1 && !use_cmap2) return;
  
  //--------------------------------------------------------------------------
  // load colormap texture
  if(use_cmap2) {
    // rebuild if needed
    build_colormap2();
    bind_colormap2();
  } else {
    // rebuild if needed
    build_colormap1();
    bind_colormap1();
  }
  
  //--------------------------------------------------------------------------
  // enable data texture unit 0
  glActiveTexture(GL_TEXTURE0_ARB);
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
  shader = vol_shader_factory_->shader(use_cmap2 ? 2 : 1, nb0, false, true,
                                       use_fog, blend_mode);

  if(shader) {
    if(!shader->valid()) {
      shader->create();
    }
    shader->bind();
  }

  //--------------------------------------------------------------------------
  // render bricks
  Polygon*  poly;
  BBox box;
  double t;
  for( ; it != it_end; it++ ) {
    Brick& b = *(*it);
    box = b.bbox();
    Point viewPt = viewRay.origin();
    Point mid = b[0] + (b[7] - b[0])*0.5;
    Point c(control_point_);
    bool draw_z = false;
    if(draw_cyl_) {
      const double to_rad = M_PI / 180.0;
      BBox bb;
      tex_->get_bounds(bb);
      Point cyl_mid = bb.min() + bb.diagonal()*0.5;
      if(draw_phi0_) {
	Vector phi(1.,0,0);
	Transform rot;
	rot.pre_rotate(phi0_ * to_rad, Vector(0,0,1.));
	phi = rot.project(phi);
	Ray r(cyl_mid, phi);
	t = intersectParam(-r.direction(), control_point_, r);
	b.ComputePoly(r, t, poly);
	draw(b, poly, use_fog);
      }
      if(draw_phi1_) {
	Vector phi(1.,0,0);
	Transform rot;
	rot.pre_rotate(phi1_ * to_rad, Vector(0,0,1.));
	phi = rot.project(phi);
	Ray r(cyl_mid, phi);
	t = intersectParam(-r.direction(), control_point_, r);
	b.ComputePoly(r, t, poly);
	draw(b, poly, use_fog);
      }
      if(draw_z_) {
        draw_z = true;
      }
    } else {
      if(draw_view_) {
	t = intersectParam(-viewRay.direction(), control_point_, viewRay);
	b.ComputePoly(viewRay, t, poly);
	draw(b, poly, use_fog);
      } else {
	if(draw_x_) {
	  Point o(b[0].x(), mid.y(), mid.z());
	  Vector v(c.x() - o.x(), 0,0);
	  if(c.x() > b[0].x() && c.x() < b[7].x() ){
	    if( viewPt.x() > c.x() ){
	      o.x(b[7].x());
	      v.x(c.x() - o.x());
	    } 
	    Ray r(o,v);
	    t = intersectParam(-r.direction(), control_point_, r);
	    b.ComputePoly( r, t, poly);
	    draw(b, poly, use_fog);
	  }
	}
	if(draw_y_) {
	  Point o(mid.x(), b[0].y(), mid.z());
	  Vector v(0, c.y() - o.y(), 0);
	  if(c.y() > b[0].y() && c.y() < b[7].y() ){
	    if( viewPt.y() > c.y() ){
	      o.y(b[7].y());
	      v.y(c.y() - o.y());
	    } 
	    Ray r(o,v);
	    t = intersectParam(-r.direction(), control_point_, r);
	    b.ComputePoly( r, t, poly);
	    draw(b, poly, use_fog);
	  }
	}
        if(draw_z_) {
          draw_z = true;
        }
      }
    }
    
    if (draw_z) {
      Point o(mid.x(), mid.y(), b[0].z());
      Vector v(0, 0, c.z() - o.z());
      if(c.z() > b[0].z() && c.z() < b[7].z() ){
	if(viewPt.z() > c.z()) {
	  o.z(b[7].z());
	  v.z(c.z() - o.z());
	} 
	Ray r(o,v);
	t = intersectParam(-r.direction(), control_point_, r);
	b.ComputePoly(r, t, poly);
	draw(b, poly, use_fog);  
      }
    }
  }

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
  glActiveTexture(GL_TEXTURE0_ARB);
  glDisable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, 0);
}
  

void
SliceRenderer::draw(Brick& b, Polygon* poly, bool use_fog)
{
  vector<Polygon *> polys;
  polys.push_back(poly);
  load_brick(b);
  draw_polys(polys, use_fog, 0);
}

void 
SliceRenderer::draw_wireframe()
{
  Ray viewRay;
  compute_view(viewRay);
}

#endif // #if defined(SCI_OPENGL)

} // namespace Volume
