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
#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Packages/Volume/Core/Util/SliceTable.h>
#include <Packages/Volume/Core/Util/ShaderProgramARB.h>

using std::string;

static const string ShaderString1 =
"!!ARBfp1.0 \n"
"TEMP v, c; \n"
"ATTRIB t = fragment.texcoord[0]; \n"
"ATTRIB f = fragment.color; \n"
"TEX v, t, texture[0], 3D; \n"
"TEX c, v, texture[1], 1D; \n"
"MUL result.color, c, f; \n"
"END";

static const string ShaderString4 =
"!!ARBfp1.0 \n"
"TEMP v, c; \n"
"ATTRIB t = fragment.texcoord[0]; \n"
"ATTRIB f = fragment.color; \n"
"TEX v, t, texture[0], 3D; \n"
"TEX c, v.w, texture[1], 1D; \n"
"MUL result.color, c, f; \n"
"END";

static const string FogShaderString1 =
"!!ARBfp1.0 \n"
"TEMP value, color, fogFactor; \n"
"PARAM fogColor = state.fog.color; \n"
"PARAM fogParam = state.fog.params; \n"
"ATTRIB fogCoord = fragment.fogcoord; \n"
"ATTRIB texCoord = fragment.texcoord[0]; \n"
"ATTRIB fragmentColor = fragment.color; \n"
"SUB fogFactor.x, fogParam.z, fogCoord.x; \n"
"MUL_SAT fogFactor.x, fogFactor.x, fogParam.w; \n"
"TEX value, texCoord, texture[0], 3D; \n"
"TEX color, value, texture[1], 1D; \n"
"MUL color, color, fragmentColor; \n"
"LRP color.xyz, fogFactor.x, color.xyzz, fogColor.xyzz; \n"
"MOV result.color, color; \n"
"END";

static const string FogShaderString4 =
"!!ARBfp1.0 \n"
"TEMP value, color, fogFactor, finalColor; \n"
"PARAM fogColor = state.fog.color; \n"
"PARAM fogParam = state.fog.params; \n"
"ATTRIB fogCoord = fragment.fogcoord; \n"
"ATTRIB texCoord = fragment.texcoord[0]; \n"
"ATTRIB fragmentColor = fragment.color; \n"
"SUB fogFactor.x, fogParam.z, fogCoord.x; \n"
"MUL_SAT fogFactor.x, fogFactor.x, fogParam.w; \n"
"TEX value, texCoord, texture[0], 3D; \n"
"TEX color, value.w, texture[1], 1D; \n"
"MUL color, color, fragmentColor; \n"
"LRP color.xyz, fogFactor.x, color.xyzz, fogColor.xyzz; \n"
"MOV result.color, color; \n"
"END";


using namespace Volume;
using SCIRun::DrawInfoOpenGL;

SliceRenderer::SliceRenderer():
  control_point_( Point(0,0,0) ),
  drawX_(false),
  drawY_(false),
  drawZ_(false),
  draw_view_(false),
  draw_phi0_(false),
  phi0_(0),
  draw_phi1_(false),
  phi1_(0),
  draw_cyl_(false)
{
  VolShader1 = new FragmentProgramARB(ShaderString1);
  VolShader4 = new FragmentProgramARB(ShaderString4);
  FogVolShader1 = new FragmentProgramARB(FogShaderString1);
  FogVolShader4 = new FragmentProgramARB(FogShaderString4);

  lighting_ = 1;
}

SliceRenderer::SliceRenderer(TextureHandle tex, ColorMapHandle map, Colormap2Handle cmap2):
  TextureRenderer(tex, map, cmap2),
  control_point_( Point(0,0,0) ),
  drawX_(false),
  drawY_(false),
  drawZ_(false),
  draw_view_(false),
  draw_phi0_(false),
  phi0_(0),
  draw_phi1_(false),
  phi1_(0),
  draw_cyl_(false)
{
  VolShader1 = new FragmentProgramARB(ShaderString1);
  VolShader4 = new FragmentProgramARB(ShaderString4);
  FogVolShader1 = new FragmentProgramARB(FogShaderString1);
  FogVolShader4 = new FragmentProgramARB(FogShaderString4);
  lighting_ = 1;
}

SliceRenderer::SliceRenderer( const SliceRenderer& copy ) :
  TextureRenderer(copy.tex_, copy.cmap_, copy.cmap2_),
  control_point_( copy.control_point_),
  drawX_(copy.drawX_),
  drawY_(copy.drawY_),
  drawZ_(copy.drawX_),
  draw_view_(copy.draw_view_),
  draw_phi0_(copy.phi0_),
  phi0_(copy.phi0_),
  draw_phi1_(copy.draw_phi1_),
  phi1_(copy.phi1_),
  draw_cyl_(copy.draw_cyl_)
{
  VolShader1 = copy.VolShader1;
  VolShader4 = copy.VolShader4;
  FogVolShader1 = copy.FogVolShader1;
  FogVolShader4 = copy.FogVolShader4;
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
  if( di->get_drawtype() == DrawInfoOpenGL::WireFrame ){
    drawWireFrame();
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
  tex_->get_sorted_bricks( bricks, viewRay );

  vector<Brick*>::iterator it = bricks.begin();
  vector<Brick*>::iterator it_end = bricks.end();

  BBox brickbounds;
  tex_->get_bounds( brickbounds );


  if( cmap_.get_rep() ) {
    if( cmap_has_changed_ || r_count_ != 1) {
      BuildTransferFunction();
      // cmap_has_changed_ = false;
    }
  }
  load_colormap();
  
  // First set up the Textures.
  glActiveTexture(GL_TEXTURE0_ARB);
  glEnable(GL_TEXTURE_3D);
  glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_REPLACE); 

  glActiveTexture(GL_TEXTURE1_ARB);
  glEnable(GL_TEXTURE_1D);
  glActiveTexture(GL_TEXTURE0_ARB);

  glColor4f(1.0, 1.0, 1.0, 1.0);
  glDepthMask(GL_TRUE);
  glEnable(GL_ALPHA_TEST);
  glAlphaFunc(GL_GREATER, 0.0);

  

  GLboolean fog;
  glGetBooleanv(GL_FOG, &fog);
  GLboolean lighting;
  glGetBooleanv(GL_LIGHTING, &lighting);
  int nb = (*bricks.begin())->data()->nb(0);

  if(fog) {
    switch (nb) {
    case 1:
      if(!FogVolShader1->valid()) {
        FogVolShader1->create();
      }
      FogVolShader1->bind();
      break;
    case 4:
      if(!FogVolShader4->valid()) {
        FogVolShader4->create();
      }
      FogVolShader4->bind();
      break;
    }
  } else {
    switch(nb) {
    case 1:
      if(!VolShader1->valid()) {
        VolShader1->create();
      }
      VolShader1->bind();
      break;
    case 4:
      if(!VolShader4->valid()) {
        VolShader4->create();
      }
      VolShader4->bind();
      break;
    }
  }
  
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

    if (draw_cyl_) {
      const double to_rad = M_PI / 180.0;
      BBox bb;
      tex_->get_bounds( bb );
      Point cyl_mid = bb.min() + bb.diagonal() * 0.5;
      if(draw_phi0_) {
	Vector phi(1.,0,0);
	
	Transform rot;
	rot.pre_rotate(phi0_ * to_rad, Vector(0,0,1.));
	phi = rot.project(phi);
      
	Ray r(cyl_mid, phi);
	t = intersectParam(-r.direction(), control_point_, r);
	b.ComputePoly( r, t, poly);
	draw( b, poly );
      }
      if(draw_phi1_) {
	Vector phi(1.,0,0);
	
	Transform rot;
	rot.pre_rotate(phi1_ * to_rad, Vector(0,0,1.));
	phi = rot.project(phi);
      
	Ray r(cyl_mid, phi);
	t = intersectParam(-r.direction(), control_point_, r);
	b.ComputePoly( r, t, poly);
	draw( b, poly );
      }
      if(drawZ_){
	draw_z = true;
      }

    } else {

      if(draw_view_){
	t = intersectParam(-viewRay.direction(), control_point_, viewRay);
	b.ComputePoly(viewRay, t, poly);
	draw(b, poly);
      } else {
      
	if(drawX_){
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
	    draw( b, poly );
	  }
	}
	if(drawY_){
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
	    draw( b, poly );
	  }
	}
	if(drawZ_){
	  draw_z = true;
	}
      }
    }
    
    if (draw_z) {
      Point o(mid.x(), mid.y(), b[0].z());
      Vector v(0, 0, c.z() - o.z());
      if(c.z() > b[0].z() && c.z() < b[7].z() ){
	if( viewPt.z() > c.z() ){
	  o.z(b[7].z());
	  v.z(c.z() - o.z());
	} 
	Ray r(o,v);
	t = intersectParam(-r.direction(), control_point_, r);
	b.ComputePoly( r, t, poly);
	draw( b, poly );  
      }
    }
  }

  if(fog) {
    switch(nb) {
    case 1:
      FogVolShader1->release();
      break;
    case 4:
      FogVolShader4->release();
      break;
    }
  } else {
    switch(nb) {
    case 1:
      VolShader1->release();
      break;
    case 4:
      VolShader4->release();
      break;
    }
  }

  glDisable(GL_ALPHA_TEST);
  glDepthMask(GL_TRUE);
  if( cmap_.get_rep() ){
    glActiveTexture(GL_TEXTURE1_ARB);
    glDisable(GL_TEXTURE_1D);
    glActiveTexture(GL_TEXTURE0_ARB);
  }
  glDisable(GL_TEXTURE_3D);
  // glEnable(GL_DEPTH_TEST);  
}
  

void
SliceRenderer::draw(Brick& b, Polygon* poly)
{
  vector<Polygon *> polys;
  polys.push_back(poly);

  load_texture(b);
//   make_texture_matrix( b );
//   enable_tex_coords();
  
//   if( !VolShader->valid() ){
//     VolShader->create();
//   }
//   VolShader->bind();
  drawPolys(polys);
//   VolShader->release();
//   disable_tex_coords();
}

void 
SliceRenderer::drawWireFrame()
{
  Ray viewRay;
  compute_view( viewRay );
  
}

void 
SliceRenderer::load_colormap()
{
  const unsigned char *arr = transfer_function_;

  glActiveTexture(GL_TEXTURE1_ARB);
  {
    glEnable(GL_TEXTURE_1D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if( cmap_texture_ == 0 || cmap_has_changed_ ){
      glDeleteTextures(1, &cmap_texture_);
      glGenTextures(1, &cmap_texture_);
      glBindTexture(GL_TEXTURE_1D, cmap_texture_);
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexImage1D(GL_TEXTURE_1D, 0,
		   GL_RGBA,
		   256, 0,
		   GL_RGBA, GL_UNSIGNED_BYTE,
		   arr);
      cmap_has_changed_ = false;
    } else {
      glBindTexture(GL_TEXTURE_1D, cmap_texture_);
    }
    glActiveTexture(GL_TEXTURE0_ARB);
  }
}
#endif // #if defined(SCI_OPENGL)

void
SliceRenderer::BuildTransferFunction()
{
  const int tSize = 256;
  float mul = 1.0/(tSize - 1);
  double bp = tan( 1.570796327 * 0.5 );
  
  double sliceRatio = 512.0;
  for ( int j = 0; j < tSize; j++ )
  {
    const Color c = cmap_->getColor(j*mul);
    const double alpha = cmap_->getAlpha(j*mul);
    
    const double alpha1 = pow(alpha, bp);
    const double alpha2 = 1.0 - pow((1.0 - alpha1), sliceRatio);
    transfer_function_[4*j + 0] = (unsigned char)(c.r()*alpha2*255);
    transfer_function_[4*j + 1] = (unsigned char)(c.g()*alpha2*255);
    transfer_function_[4*j + 2] = (unsigned char)(c.b()*alpha2*255);
    transfer_function_[4*j + 3] = (unsigned char)(alpha2*255);
  }
}

void
SliceRenderer::BuildTransferFunction2()
{
}
