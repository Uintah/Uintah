#include <Packages/Kurt/Core/Geom/GridSliceRen.h>
#include <Packages/Kurt/Core/Geom/BrickGrid.h>
#include <Core/GLVolumeRenderer/SliceTable.h>
#include <Core/GLVolumeRenderer/VolumeUtils.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Vector.h>
#include <iostream>
#include <unistd.h>
#include <vector>


namespace Kurt {


using SCIRun::intersectParam;
using SCIRun::sortParameters;
using SCIRun::SliceTable;
using SCIRun::Transform;
using SCIRun::Vector;
using SCIRun::Cross;


GridSliceRen::GridSliceRen(){}

void GridSliceRen::draw(const BrickGrid& bg, int)
{
  if( newbricks_ ){
    glDeleteTextures( textureNames.size(), &(textureNames[0]));
    textureNames.clear();
    newbricks_ = false;
  }
  Ray viewRay;
  computeView(viewRay);

  Polygon* poly;
  BBox box;
  double t;
  BrickGrid::iterator it = bg.begin(viewRay);
  BrickGrid::iterator it_end = bg.end(viewRay);
  for(; it != it_end; ++it) {

    Brick& b = *(*it);
    box = b.bbox();
    Point viewPt = viewRay.origin();
    Point mid = b[0] + (b[7] - b[0])*0.5;
    Point c(controlPoint);

    if(drawView){
      t = intersectParam(-viewRay.direction(), controlPoint, viewRay);
      b.ComputePoly(viewRay, t, poly);
      draw(b, poly);
    } else {

      if(drawX){
	Point o(b[0].x(), mid.y(), mid.z());
	Vector v(c.x() - o.x(), 0,0);
	if(c.x() > b[0].x() && c.x() < b[7].x() ){
	  if( viewPt.x() > c.x() ){
	    o.x(b[7].x());
	    v.x(c.x() - o.x());
	  } 
	  Ray r(o,v);
	  t = intersectParam(-r.direction(), controlPoint, r);
	  b.ComputePoly( r, t, poly);
	  draw( b, poly );
	}
      }
      if(drawY){
	Point o(mid.x(), b[0].y(), mid.z());
	Vector v(0, c.y() - o.y(), 0);
	if(c.y() > b[0].y() && c.y() < b[7].y() ){
	  if( viewPt.y() > c.y() ){
	    o.y(b[7].y());
	    v.y(c.y() - o.y());
	  } 
	  Ray r(o,v);
	  t = intersectParam(-r.direction(), controlPoint, r);
	  b.ComputePoly( r, t, poly);
	  draw( b, poly );
	}
      }
      if(drawZ){
	Point o(mid.x(), mid.y(), b[0].z());
	Vector v(0, 0, c.z() - o.z());
	if(c.z() > b[0].z() && c.z() < b[7].z() ){
	  if( viewPt.z() > c.z() ){
	    o.z(b[7].z());
	    v.z(c.z() - o.z());
	  } 
	  Ray r(o,v);
	  t = intersectParam(-r.direction(), controlPoint, r);
	  b.ComputePoly( r, t, poly);
	  draw( b, poly );
	}
      }
    }
  }
}

void
GridSliceRen::draw(Brick& b, Polygon* poly)
{
  vector<Polygon *> polys;
  polys.push_back( poly );
  loadColorMap( );
  loadTexture( b );
  makeTextureMatrix( b );
  enableTexCoords();
  glColor4f(1,1,1,1); // was  setAlpha( b );
  drawPolys( polys );
  disableTexCoords();
}

void GridSliceRen::drawWireFrame(const BrickGrid& bg )
{
  Ray viewRay;
  computeView(viewRay);

  Polygon* poly;
  BBox box;
  double t;
  BrickGrid::iterator it = bg.begin(viewRay);
  BrickGrid::iterator it_end = bg.end(viewRay);
  for(; it != it_end; ++it) {

    Brick& b = *(*it);
    box = b.bbox();
    Point viewPt = viewRay.origin();
    Point mid = b[0] + (b[7] - b[0])*0.5;
    Point c(controlPoint);

    if(drawView){
      t = intersectParam(-viewRay.direction(), controlPoint, viewRay);
      b.ComputePoly(viewRay, t, poly);
      draw(poly);
    } else {

      if(drawX){
	Point o(b[0].x(), mid.y(), mid.z());
	Vector v(c.x() - o.x(), 0,0);
	if(c.x() > b[0].x() && c.x() < b[7].x() ){
	  if( viewPt.x() > c.x() ){
	    o.x(b[7].x());
	    v.x(c.x() - o.x());
	  } 
	  Ray r(o,v);
	  t = intersectParam(-r.direction(), controlPoint, r);
	  b.ComputePoly( r, t, poly);
	  draw( poly );
	}
      }
      if(drawY){
	Point o(mid.x(), b[0].y(), mid.z());
	Vector v(0, c.y() - o.y(), 0);
	if(c.y() > b[0].y() && c.y() < b[7].y() ){
	  if( viewPt.y() > c.y() ){
	    o.y(b[7].y());
	    v.y(c.y() - o.y());
	  } 
	  Ray r(o,v);
	  t = intersectParam(-r.direction(), controlPoint, r);
	  b.ComputePoly( r, t, poly);
	  draw( poly );
	}
      }
      if(drawZ){
	Point o(mid.x(), mid.y(), b[0].z());
	Vector v(0, 0, c.z() - o.z());
	if(c.z() > b[0].z() && c.z() < b[7].z() ){
	  if( viewPt.z() > c.z() ){
	    o.z(b[7].z());
	    v.z(c.z() - o.z());
	  } 
	  Ray r(o,v);
	  t = intersectParam(-r.direction(), controlPoint, r);
	  b.ComputePoly( r, t, poly);
	  draw( poly );
	}
      }
    }
  }
}

void 
GridSliceRen::draw(Polygon* poly)
{
  vector<Polygon *> polys;
  polys.push_back( poly );
  drawPolys( polys );
}

} // end namespace Kurt
