
/******************************************
IMPLEMENTATION
   GridLines
     A module that displays gridlines around a scalar field.


GENERAL INFORMATION
   GridLines.h
   Written by:

     Kurt Zimmerman
     Department of Computer Science
     University of Utah
     December 1999

     Copyright (C) 1999 SCI Group

KEYWORDS
   GridLines, ScalarField, VectorField

DESCRIPTION
   This module was created for the Uintah project to display that actual
   structure of the scalar or vector fields that were being used
   during simulation computations.  The number of lines displayed represent
   the actual grid and cannot be manipulated. This module is based on 
   Philip Sutton's cfdGridLines.cc which was based on FieldCage.cc by
   David Weinstein.

***************************************** */


#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldRGCC.h>
#include <SCICore/Datatypes/VectorFieldRGCC.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomText.h>
#include <SCICore/Geom/GeomCylinder.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include "GridLines.h"

namespace Uintah {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Containers;
using namespace SCICore::Math;

GeomObj* GeomLineFactory::Create(int id , const Point& p1, const Point& p2,
				 double rad,  int nu, int nv)
{
  if( id == 0 )
    return new GeomLine( p1, p2 );
  else if(id ==1)
    return new GeomCylinder( p1, p2, rad, nu, nv);
  else
    return 0;
}
      
GridLines::GridLines(const clString& id)
  : Module("GridLines", id, Filter), rad("rad", id, this),
  mode("mode",id,this), lineRep("lineRep", id, this),
  textSpace("textSpace", id, this), dim("dim", id, this),
  plane("plane",id,this), planeLoc("planeLoc", id, this)
{
  // Create the input ports
  insfield=new ScalarFieldIPort(this,"ScalarField",ScalarFieldIPort::Atomic);
  add_iport(insfield);
  invfield=new VectorFieldIPort(this,"VectorField",VectorFieldIPort::Atomic);
  add_iport(invfield);
    
  // Create the output port
  ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom);
  matl = new Material(Color(0,0,0), Color(0.2, 0.2, 0.2),
			 Color(.5,.5,.5), 20);
  white = new Material(Color(0,0,0), Color(1,1,1), Color(1,1,1), 20);

}

GridLines::~GridLines()
{
}

void GridLines::execute()
{
  int numx;
  int numy;
  int numz;

  bool CC = false;
  
  ScalarFieldHandle sfield;
  Point min, max;
  bool haveit=false;
  if(insfield->get(sfield)){
    sfield->get_bounds(min, max);
    ScalarFieldRG *rg = sfield->getRG();
    if( rg != 0 ){
      numx = rg->nx; numy = rg->ny; numz = rg->nz;
      if( ScalarFieldRGCC *sfrgcc =
	  dynamic_cast<ScalarFieldRGCC *> (sfield.get_rep())){
	CC = true;
      }
    } 
    haveit=true;
  }
  VectorFieldHandle vfield;
  if(invfield->get(vfield)){
    vfield->get_bounds(min, max);
    VectorFieldRG *rg = vfield->getRG();
    if( rg != 0 ){
      numx = rg->nx; numy = rg->ny; numz = rg->nz;
      if( VectorFieldRGCC *vfrgcc =
	  dynamic_cast<VectorFieldRGCC *> (vfield.get_rep())) {
	CC = true;
      }
    } 
    haveit=true;
  }
  if(!haveit)
    return;
  GeomGroup* all=new GeomGroup();

  int m = mode.get();
  int lR = lineRep.get();

  // sizes for drawing text
  double xsize = 0;
  double ysize = 0;
  double zsize = 0;

  // deltas for drawing borders;
  double dx = (max.x() - min.x());
  double dy = (max.y() - min.y());
  double dz = (max.z() - min.z());
  
  if( m == 1 || m == 2 || m == 3 ) {
    // inside, outside, or both - add border lines
    for( int iz = 0; iz < 2; iz++){
      double pz = min.z() + dz * iz;
      for( int iy = 0; iy < 2; iy++){
	double py = min.y() + dy * iy;
	
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(max.x(), py, pz),
						  Point(min.x(), py, pz),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
    }
    for( int iy = 0; iy < 2; iy++){
      double py = min.y() + dy * iy;
      for( int ix = 0; ix < 2; ix++){
	double px = min.x() + dx * ix;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(px, py, min.z()),
						  Point(px, py, max.z()),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
    }
    for( int iz = 0; iz < 2; iz++){
      double pz = min.z() + dz * iz;
      for( int ix = 0; ix < 2; ix++){
	double px = min.x() + dx * ix;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(px, min.y(), pz),
						  Point(px, max.y(), pz),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
    }
  }
  
  // deltas for drawing inside lines
  if( CC ) { // cell centered data
    dx /= numx;
    dy /= numy;
    dz /= numz;
  } else {    
    dx /= (numx - 1);
    dy /= (numy - 1);
    dz /= (numz - 1);
  }
  if(dim.get() == 1 && ( m == 1 || m == 3 )){
    if ( m == 3 ){
      xsize = (max.x() - min.x()) / textSpace.get();
      ysize = (max.y() - min.y()) / textSpace.get();
      zsize = (max.z() - min.z()) / textSpace.get();
    }

    // draw inside lines
    for( int iz = 1; iz < numz - 2; iz++){
      double pz = min.z() + dz * iz;
      for( int iy = 0; iy < numy; iy++){
	double py = min.y() + dy * iy;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(max.x(), py, pz),
						  Point(min.x()-xsize,py,pz),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
    }
    for( int iy = 0; iy < numy; iy++){
      double py = min.y() + dy * iy;
      for( int ix = 1; ix < numx - 2; ix ++){
	double px = min.x() + dx * ix;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
							     Point(px, py,
							min.z() - zsize),
						  Point(px, py, max.z()),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
    }
    for( int iz = 0; iz < numz; iz++){
      double pz = min.z() + dz * iz;
      for( int ix = 0; ix < numx; ix++){
	double px = min.x() + dx * ix;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
							     Point(px,
							min.y()-ysize, pz),
						  Point(px, max.y(), pz),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
    }
  }


  if( dim.get() == 0 && (m == 1 || m == 3)){
    if ( m == 3 ){
      xsize = (max.x() - min.x()) / textSpace.get();
      ysize = (max.y() - min.y()) / textSpace.get();
      zsize = (max.z() - min.z()) / textSpace.get();
    }
    
    if( plane.get() == 0 ){ // XY plane
      double pz = min.z() + planeLoc.get()*(max.z()- min.z());
      // draw inside lines
      for( int iy = 0; iy < numy; iy++){
	double py = min.y() + dy * iy;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(max.x(), py, pz),
						  Point(min.x() - xsize,
							py, pz     ),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
      for( int ix = 0; ix < numx; ix++){
	double px = min.x() + dx * ix;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(px, max.y(), pz),
						  Point(px, min.y() - ysize,
						        pz     ),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
    } else if (plane.get() == 1){ //XZ plane
      double py = min.y() + planeLoc.get() *(max.y() - min.y());
      for( int iz = 0; iz < numz; iz++){
	double pz = min.z() + dz * iz;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(max.x(), py, pz),
						  Point(min.x() - xsize,
							py, pz     ),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
      for( int ix = 0; ix < numx; ix++){
	double px = min.x() + dx * ix;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(px, py,
							min.z() - zsize),
						  Point(px, py, max.z()),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
    } else { // YZ plane
      double px = min.x() + planeLoc.get()*(max.x() - min.x());
      for( int iy = 0; iy < numy; iy++){
	double py = min.y() + dy * iy;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(px, py,
							min.z() - zsize),
						  Point(px, py, max.z()),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
      for( int iz = 0; iz < numz; iz++){
	double pz = min.z() + dz * iz;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(px, min.y()-ysize, pz),
						  Point(px, max.y(), pz),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
    }
  }
  if( dim.get() == 1 && m == 2 ){
    // draw outside lines
    xsize = (max.x() - min.x()) / textSpace.get();
    ysize = (max.y() - min.y()) / textSpace.get();
    zsize = (max.z() - min.z()) / textSpace.get();
    for( int iy = 0; iy < numy; iy++){
      double py = min.y() + dy * iy;
      all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						       Point(min.x()-xsize, py,
							     min.z()),
						       Point(min.x(), py,
							     min.z()),
						       rad.get(), 3, 1 ),
				    matl ) );
    }
    for( int ix = 0; ix < numx; ix ++){
      double px = min.x() + dx * ix;
      all->add( new GeomMaterial( GeomLineFactory::Create( lR,
					     Point(px, min.y(), min.z()-zsize),
					     Point(px, min.y(), min.z()),
					     rad.get(), 3, 1 ),
				  matl ) );
    }
    for( int iz = 0; iz < numz; iz++){
      double pz = min.z() + dz * iz;
      all->add( new GeomMaterial( GeomLineFactory::Create( lR,
					     Point(min.x(), min.y()-ysize, pz),
					     Point(min.x(), min.y(), pz),
					     rad.get(), 3, 1 ),
				    matl ) );
    }
  }
  
  if(dim.get() == 0 && m == 2){
    xsize = (max.x() - min.x()) / textSpace.get();
    ysize = (max.y() - min.y()) / textSpace.get();
    zsize = (max.z() - min.z()) / textSpace.get();
    if( plane.get() == 0 ){ // XY
      double pz = min.z() + planeLoc.get()*(max.z()- min.z());
      for( int iy = 0; iy < numy; iy++){
	double py = min.y() + dy * iy;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						       Point(min.x()-xsize, py,
							     pz),
						       Point(min.x(), py,
							     pz),
						       rad.get(), 3, 1 ),
				    matl ) );
      }
      for( int ix = 0; ix < numx; ix++){
	double px = min.x() + dx * ix;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(px, min.y()-ysize, pz),
						  Point(px, min.y(), pz),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
    } else if ( plane.get() == 1 ){ // XZ
      double py = min.y() + planeLoc.get()*(max.y() - min.y());
      for( int ix = 0; ix < numx; ix++){
	double px = min.x() + dx * ix;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(px, py, min.z()-zsize),
						  Point(px, py, min.z()),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
      for( int iz = 0; iz < numz; iz++){
	double pz = min.z() + dz * iz;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(min.x(), py, pz),
						  Point(min.x()-xsize,py,pz),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
    } else { // YZ
      double px = min.x() + planeLoc.get()*(max.x() - min.x() );
      for( int iy = 0; iy < numy; iy++){
	double py = min.y() + dy * iy;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(px, py, min.z()-zsize),
						  Point(px, py, min.z()),
						  rad.get(), 3, 1 ),
				    matl ) );
      }
      for( int iz = 0; iz < numz; iz++){
	double pz = min.z() + dz * iz;
	all->add( new GeomMaterial( GeomLineFactory::Create( lR,
						  Point(px, min.y(), pz),
						  Point(px, min.y()-ysize, pz),
						  rad.get(), 3, 1 ),
				      matl ) );
      }	
    }
  }
  if( m != 0 ) {
    int i;

    char txt[80];
    // add numbers by 10's
    if( m == 1 ) {
      // inside only - place numbers close to field
      for( i = 0; i < numx; i+=textSpace.get() ) {
	sprintf( txt, "%d", i );
	all->add(new GeomMaterial(new GeomText(txt,
					       Point(min.x()+i*dx
						     ,min.y()-ysize/2.0,
						     min.z() - zsize/2.0)),
				  white) );
      }
      sprintf( txt, "%d", numx );
      all->add(new GeomMaterial(new GeomText(txt,
					     Point(max.x(), min.y()-ysize/2.0,
						   min.z() - zsize/2.0 )),
				white) );
      for( i = 0; i < numy; i+=textSpace.get() ) {
	sprintf( txt, "%d", i );
	all->add(new GeomMaterial(new GeomText(txt,
					       Point(min.x()-xsize/2.0,
						     min.y()+i*dy,
						     min.z() - zsize/2.0)),
				  white) );
      }
      sprintf( txt, "%d", numy );
      all->add(new GeomMaterial(new GeomText(txt,
					     Point(min.x()-xsize/2.0,
						   max.y(),min.z() - zsize/2.0)),
				white) );
      
      for( i = 0; i < numz; i+=textSpace.get() ) {
	sprintf( txt, "%d", i );
	all->add(new GeomMaterial(new GeomText(txt,
					       Point(min.x()-xsize/2.0,
						     min.y() - ysize/2.0,
						     min.z()+i*dz)),
				  white) );
      }
      sprintf( txt, "%d", numy );
      all->add(new GeomMaterial(new GeomText(txt,
					     Point(min.x()-xsize/2.0,
						   min.z() - zsize/2.0,
						   max.z())),
				white));
	       
    } else if ( m != 0 ) {
      // outside & both - place numbers further away because of outside lines
      for( i = 0; i < numx; i+=textSpace.get() ) {
	sprintf( txt, "%d", i );
	all->add(new GeomMaterial(new GeomText(txt,
					       Point(min.x()+i*dx
						     ,min.y()-ysize,
						     min.z() - zsize)),
				  white));
      }
      sprintf( txt, "%d", numx );
      all->add(new GeomMaterial(new GeomText(txt,
					     Point(max.x(), min.y()-ysize,
						   min.z() - zsize )),
				white) );
      for( i = 0; i < numy; i+=textSpace.get() ) {
	sprintf( txt, "%d", i );
	all->add(new GeomMaterial(new GeomText(txt,
					       Point(min.x()-xsize,
						     min.y()+i*dy,
						     min.z() - zsize)),
				  white) );
      }
      sprintf( txt, "%d", numy );
      all->add(new GeomMaterial(new GeomText(txt,
					     Point(min.x()-xsize,
						   max.y(),min.z() - zsize)),
				white) );
      for( i = 0; i < numz; i+=textSpace.get() ) {
	sprintf( txt, "%d", i );
	all->add(new GeomMaterial(new GeomText(txt,
					       Point(min.x()-xsize,
						     min.y() - ysize,
						     min.z()+i*dz)),
				  white) );
      }
      sprintf( txt, "%d", numy );
      all->add(new GeomMaterial(new GeomText(txt,
					     Point(min.x()-xsize,
						   min.z() - zsize,
						   max.z())),
				white));
    }
  }  
  ogeom->delAll();
  ogeom->addObj(all, "Grid Lines");
}

PSECore::Dataflow::Module*
    make_GridLines( const SCICore::Containers::clString& id ) {
  return new Uintah::Modules::GridLines( id );
}

} // End namespace Modules
} // End namespace Uintah

