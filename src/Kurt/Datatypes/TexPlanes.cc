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
//     t = intersectParam(-viewRay.direction(), volren->controlPoint, viewRay);

//     if( box.inside( viewRay.parameter( t ) )){
//       if(volren->drawView){
// 	b.ComputePoly( viewRay, t, poly);
// 	draw( b, poly );
//       }
    if(volren->drawX){
      Vector v(1,0,0);
      Point o(volren->controlPoint);
	std::cerr<<"brick min = "<<b[0].x()
		 <<", brick max = "<<b[7].x()<<std::endl;
	std::cerr<<"control Point = "<< o <<std::endl;
      if(o.x() > b[0].x() && o.x() < b[7].x() ){
	if( viewRay.origin().x() > o.x() ){
	  v.x(-1);
	  o.x(b[7].x());
	} else {
	  o.x(b[0].x());
	}
	Ray r(o,v);
	t = intersectParam(-r.direction(), volren->controlPoint, r);
	b.ComputePoly( r, t, poly);
	draw( b, poly );
      }
    }
    if(volren->drawY){
      Vector v(1,0,0);
      Point o(volren->controlPoint);
      if(o.y() > b[0].y() && o.y() < b[7].y() ){
	if( viewRay.origin().y() > o.y() ){
	  v.y(-1);
	  o.y(b[7].y());
	} else {
	  o.y(b[0].y());
	}
	Ray r(o,v);
	t = intersectParam(-r.direction(), volren->controlPoint, r);
	b.ComputePoly( r, t, poly);
	draw( b, poly );
      }
    }
    if(volren->drawZ){
      Vector v(1,0,0);
      Point o(volren->controlPoint);
      if(o.z() > b[0].z() && o.z() < b[7].z() ){
	if( viewRay.origin().z() > o.z() ){
	  v.z(-1);
	  o.z(b[7].z());
	} else {
	  o.z(b[0].z());
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

