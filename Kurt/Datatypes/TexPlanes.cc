#include "TexPlanes.h"
#include <SCICore/Geometry/Ray.h>
#include "FullResIterator.h"
#include "Brick.h"
#include "SliceTable.h"
#include "GLVolumeRenderer.h"
#include "VolumeUtils.h"
#include <iostream>
namespace SCICore {
namespace GeomSpace {

using SCICore::Geometry::Ray;
using Kurt::Datatypes::SliceTable;

GLVolRenState* TexPlanes::_instance = 0;

TexPlanes::TexPlanes(const GLVolumeRenderer* glvr ) :
  GLVolRenState( glvr )
{
}


void
TexPlanes::draw()
{
  using std::cerr;
  using std::endl;
  
  Ray viewRay;
  Brick* brick;
  computeView(viewRay);
  
  FullResIterator it( volren->tex, viewRay,  volren->controlPoint);

  Polygon*  poly;
  BBox box;
  double t;
  for( brick = it.Start(); !it.isDone(); brick = it.Next()){
    Brick& b = *brick;
    box = b.bbox();
    Point viewPt = viewRay.origin();
    Point mid = b[0] + (b[7] - b[0])*0.5;
    Point c(volren->controlPoint);

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


void
TexPlanes::draw(Brick& b, Polygon* poly)
{
  vector<Polygon *> polys;
  polys.push_back( poly );
  //  disableBlend();
  loadTexture( b );
  makeTextureMatrix( b );
  enableTexCoords();
  setAlpha( b );
  drawPolys( polys );
  disableTexCoords();
  //  enableBlend();
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
  
  FullResIterator it( volren->tex, viewRay,  volren->controlPoint);

  const Brick* brick;
  for( brick = it.Start(); !it.isDone(); brick = it.Next()){
    GLVolRenState::drawWireFrame( *brick );
  }
}
GLVolRenState* 
TexPlanes::Instance(const GLVolumeRenderer* glvr)
{
  // Not a true Singleton class, but this does make sure that 
  // there is only one instance per volume renderer.
  if( _instance == 0 ){
    _instance = new TexPlanes( glvr );
  }
  
  return _instance;
}



} // end namespace Datatypes
} // end namespace Kurt

