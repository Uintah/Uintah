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

#include <Core/Datatypes/TexPlanes.h>
#include <Core/Geometry/Ray.h>
#include <Core/Datatypes/FullResIterator.h>
#include <Core/Datatypes/Brick.h>
#include <Core/Datatypes/SliceTable.h>
#include <Core/Datatypes/GLVolumeRenderer.h>
#include <Core/Datatypes/VolumeUtils.h>
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
  
  FullResIterator it( volren->tex.get_rep(), viewRay,  volren->controlPoint);

  Polygon*  poly;
  BBox box;
  double t;
  for( brick = it.Start(); !it.isDone(); brick = it.Next()){
    Brick& b = *brick;
    box = b.bbox();
    Point viewPt = viewRay.origin();
    Point mid = b[0] + (b[7] - b[0])*0.5;
    Point c(volren->controlPoint);

    if(volren->drawView){
      t = intersectParam(-viewRay.direction(), volren->controlPoint, viewRay);
      b.ComputePoly(viewRay, t, poly);
      draw(b, poly);
    } else {

      if(volren->drawX){
	Point o(b[0].x(), mid.y(), mid.z());
	Vector v(c.x() - o.x(), 0,0);
	if(c.x() > b[0].x() && c.x() < b[7].x() ){
	  if( viewPt.x() > c.x() ){
	    o.x(b[7].x());
	    v.x(c.x() - o.x());
	  } 
	  Ray r(o,v);
	  t = intersectParam(-r.direction(), volren->controlPoint, r);
	  b.ComputePoly( r, t, poly);
	  draw( b, poly );
	}
      }
      if(volren->drawY){
	Point o(mid.x(), b[0].y(), mid.z());
	Vector v(0, c.y() - o.y(), 0);
	if(c.y() > b[0].y() && c.y() < b[7].y() ){
	  if( viewPt.y() > c.y() ){
	    o.y(b[7].y());
	    v.y(c.y() - o.y());
	  } 
	  Ray r(o,v);
	  t = intersectParam(-r.direction(), volren->controlPoint, r);
	  b.ComputePoly( r, t, poly);
	  draw( b, poly );
	}
      }
      if(volren->drawZ){
	Point o(mid.x(), mid.y(), b[0].z());
	Vector v(0, 0, c.z() - o.z());
	if(c.z() > b[0].z() && c.z() < b[7].z() ){
	  if( viewPt.z() > c.z() ){
	    o.z(b[7].z());
	    v.z(c.z() - o.z());
	  } 
	  Ray r(o,v);
	  t = intersectParam(-r.direction(), volren->controlPoint, r);
	  b.ComputePoly( r, t, poly);
	  draw( b, poly );
	}
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
  makeTextureMatrix( b );
  enableTexCoords();
  setAlpha( b );
  drawPolys( polys );
  disableTexCoords();
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
  
  FullResIterator it( volren->tex.get_rep(), viewRay,  volren->controlPoint);

  const Brick* brick;
  for( brick = it.Start(); !it.isDone(); brick = it.Next()){
    GLVolRenState::drawWireFrame( *brick );
  }
}

} // End namespace SCIRun

