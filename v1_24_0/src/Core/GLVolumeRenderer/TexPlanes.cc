/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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

#include <sci_gl.h>
#include <sci_glu.h>

#include <Core/GLVolumeRenderer/TexPlanes.h>
#include <Core/GLVolumeRenderer/GLVolumeRenderer.h>
#include <Core/GLVolumeRenderer/FullResIterator.h>
#include <Core/GLVolumeRenderer/Brick.h>
#include <Core/GLVolumeRenderer/SliceTable.h>
#include <Core/GLVolumeRenderer/VolumeUtils.h>
#include <Core/Geometry/Ray.h>
#include <iostream>

namespace SCIRun {


TexPlanes::TexPlanes(const GLVolumeRenderer* glvr ) :
  GLVolRenState( glvr )
{
}


void
TexPlanes::draw()
{
  using std::cerr;
  using std::endl;
  
  if( newbricks_ ){
    glDeleteTextures( textureNames.size(), &(textureNames[0]));
    textureNames.clear();
    newbricks_ = false;
  }

  Ray viewRay;
  Brick* brick;
  computeView(viewRay);
  
  FullResIterator it( volren->tex().get_rep(), viewRay,  volren->control_point());

  Polygon*  poly;
  BBox box;
  double t;
  for( brick = it.Start(); !it.isDone(); brick = it.Next()){
    Brick& b = *brick;
    box = b.bbox();
    Point viewPt = viewRay.origin();
    Point mid = b[0] + (b[7] - b[0])*0.5;
    Point c(volren->control_point());
    bool draw_z = false;

    if (volren->draw_cyl()) {
      const double to_rad = M_PI / 180.0;
      Point cyl_mid = bounding_box_.min() + bounding_box_.diagonal() * 0.5;
      if(volren->draw_phi_0()) {
	Vector phi(1.,0,0);
	
	Transform rot;
	rot.pre_rotate(volren->phi0() * to_rad, Vector(0,0,1.));
	phi = rot.project(phi);
      
	Ray r(cyl_mid, phi);
	t = intersectParam(-r.direction(), volren->control_point(), r);
	b.ComputePoly( r, t, poly);
	draw( b, poly );
      }
      if(volren->draw_phi_1()) {
	Vector phi(1.,0,0);
	
	Transform rot;
	rot.pre_rotate(volren->phi1() * to_rad, Vector(0,0,1.));
	phi = rot.project(phi);
      
	Ray r(cyl_mid, phi);
	t = intersectParam(-r.direction(), volren->control_point(), r);
	b.ComputePoly( r, t, poly);
	draw( b, poly );
      }
      if(volren->drawZ()){
	draw_z = true;
      }

    } else {

      if(volren->drawView()){
	t = intersectParam(-viewRay.direction(), volren->control_point(), viewRay);
	b.ComputePoly(viewRay, t, poly);
	draw(b, poly);
      } else {
      
	if(volren->drawX()){
	  Point o(b[0].x(), mid.y(), mid.z());
	  Vector v(c.x() - o.x(), 0,0);
	  if(c.x() > b[0].x() && c.x() < b[7].x() ){
	    if( viewPt.x() > c.x() ){
	      o.x(b[7].x());
	      v.x(c.x() - o.x());
	    } 
	    Ray r(o,v);
	    t = intersectParam(-r.direction(), volren->control_point(), r);
	    b.ComputePoly( r, t, poly);
	    draw( b, poly );
	  }
	}
	if(volren->drawY()){
	  Point o(mid.x(), b[0].y(), mid.z());
	  Vector v(0, c.y() - o.y(), 0);
	  if(c.y() > b[0].y() && c.y() < b[7].y() ){
	    if( viewPt.y() > c.y() ){
	      o.y(b[7].y());
	      v.y(c.y() - o.y());
	    } 
	    Ray r(o,v);
	    t = intersectParam(-r.direction(), volren->control_point(), r);
	    b.ComputePoly( r, t, poly);
	    draw( b, poly );
	  }
	}
	if(volren->drawZ()){
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
	t = intersectParam(-r.direction(), volren->control_point(), r);
	b.ComputePoly( r, t, poly);
	draw( b, poly );  
      }
    }
    
  }
}
  

void
TexPlanes::draw(Brick& b, Polygon* poly)
{
  vector<Polygon *> polys;
  polys.push_back( poly );
  loadColorMap( b );
  loadTexture( b );
  setAlpha( b );
  makeTextureMatrix( b );
  enableTexCoords();
#if defined( GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
  if( !VolShader->created() ){
    VolShader->create();
  }
  VolShader->bind();
#endif
  drawPolys( polys );
#if defined( GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
  VolShader->release();
#endif
  disableTexCoords();
  reload_ = false;
}

  void
    TexPlanes::setAlpha( const Brick&)
{
  glColor4f(1,1,1,1);
}

void 
TexPlanes::drawWireFrame()
{
  Ray viewRay;
  computeView( viewRay );
  
  FullResIterator it( volren->tex().get_rep(), viewRay,  volren->control_point());

  const Brick* brick;
  for( brick = it.Start(); !it.isDone(); brick = it.Next()){
    GLVolRenState::drawWireFrame( *brick );
  }
}

} // End namespace SCIRun

