
/*
 *  cfdGridLines.cc: display gridlines around a scalar field
 *    assumes a 2d field oriented in x-y.
 *
 *  Written by:
 *   Philip Sutton (based on FieldCage.cc by David Weinstein)
 *   Department of Computer Science
 *   University of Utah
 *   May 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/VectorFieldPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/VectorFieldRG.h>
#include <Geom/Geom.h>
#include <Geom/Line.h>
#include <Geom/Group.h>
#include <Geom/Text.h>
#include <Geom/Cylinder.h>
#include <Geom/Material.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>
#include <iostream.h>
#include <stdio.h>

class cfdGridLines : public Module {
  ScalarFieldIPort* insfield;
  VectorFieldIPort* invfield;
  GeometryOPort* ogeom;
  MaterialHandle matl;
  MaterialHandle white;
  TCLdouble rad;
  TCLdouble scalex;
  TCLdouble scaley;
  TCLint expy;
  TCLint expx;
  TCLint mode;

public:
  cfdGridLines(const clString& id);
  cfdGridLines(const cfdGridLines&, int deep);
  virtual ~cfdGridLines();
  virtual Module* clone(int deep);
  virtual void execute();
};

extern "C" {
Module* make_cfdGridLines(const clString& id)
{
  return new cfdGridLines(id);
}
}

cfdGridLines::cfdGridLines(const clString& id)
  : Module("cfdGridLines", id, Filter), rad("rad", id, this),
    scalex("scalex", id, this), scaley("scaley", id, this),
    expy("expy",id,this), expx("expx",id,this), mode("mode",id,this)
{
  // Create the input ports
  insfield=new ScalarFieldIPort(this,"ScalarField",ScalarFieldIPort::Atomic);
  add_iport(insfield);
  invfield=new VectorFieldIPort(this,"VectorField",VectorFieldIPort::Atomic);
  add_iport(invfield);
    
  // Create the output port
  ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom);
  matl = scinew Material(Color(0,0,0), Color(0.2, 0.2, 0.2),
			 Color(.5,.5,.5), 20);
  white = scinew Material(Color(0,0,0), Color(1,1,1), Color(1,1,1), 20);

  //  rad.set( (max.x() - min.x()) / (10*numx) );
  rad.set(0);
  scalex.set(1.0);
  scaley.set(1.0);
  expx.set(0);
  expy.set(0);
  mode.set(0);
}

cfdGridLines::cfdGridLines(const cfdGridLines& copy, int deep)
: Module(copy, deep), rad("rad", id, this), scalex("scalex", id, this),
  scaley("scaley", id, this),  expy("expy",id,this), expx("expx",id,this),
  mode("mode",id,this)
{
  NOT_FINISHED("cfdGridLines::cfdGridLines");
}

cfdGridLines::~cfdGridLines()
{
}

Module* cfdGridLines::clone(int deep)
{
  return new cfdGridLines(*this, deep);
}

void cfdGridLines::execute()
{
  int numx;
  int numy;
  
  ScalarFieldHandle sfield;
  Point min, max;
  bool haveit=false;
  if(insfield->get(sfield)){
    sfield->get_bounds(min, max);
    ScalarFieldRG *rg = sfield->getRG();
    if( rg != 0 ){
      numx = rg->nx; numy = rg->ny;
    } 
    haveit=true;
  }
  VectorFieldHandle vfield;
  if(invfield->get(vfield)){
    vfield->get_bounds(min, max);
    VectorFieldRG *rg = vfield->getRG();
    if( rg != 0 ){
      numx = rg->nx; numy = rg->ny;
    } 
    haveit=true;
  }
  if(!haveit)
    return;
  GeomGroup* all=new GeomGroup();

  double zm = min.z();
  double zM = max.z();
  double z = (zm+zM)/2.0;
  double xsize = (max.x() - min.x()) / 10.0;
  double ysize = (max.y() - min.y()) / 10.0;
  
  int m = mode.get();
  if( m == 1 || m == 3 ) {
    // inside or both - add border lines
    all->add( new GeomMaterial( new GeomCylinder( Point(max.x(), min.y(), z),
						  Point(min.x(), min.y(), z),
						  rad.get(), 3, 1 ),
				matl ) );
    all->add( new GeomMaterial( new GeomCylinder( Point(min.x(), max.y(), z),
						  Point(max.x(), max.y(), z),
						  rad.get(), 3, 1 ),
				matl ) );
    all->add( new GeomMaterial( new GeomCylinder( Point(min.x(), min.y(), z),
						  Point(min.x(), max.y(), z),
						  rad.get(), 3, 1 ),
				matl ) );
    all->add( new GeomMaterial( new GeomCylinder( Point(max.x(), max.y(), z),
						  Point(max.x(), min.y(), z),
						  rad.get(), 3, 1 ),
				matl ) );
  }
  if( m == 2 || m == 3 ) {
    // outside or both - add border lines
    all->add( new GeomMaterial( new GeomLine( Point( min.x(), min.y(), z ),
					      Point( min.x()-xsize, min.y(),z)
					      ),
				white ) );
    all->add( new GeomMaterial( new GeomLine( Point( max.x(), min.y(), z ),
					      Point( max.x()+xsize, min.y(),z)
					      ),
				white ) );
    all->add( new GeomMaterial( new GeomLine( Point( min.x(), min.y(), z ),
					      Point( min.x(), min.y()-ysize,z)
					      ),
				white ) );
    all->add( new GeomMaterial( new GeomLine( Point( max.x(), min.y(), z ),
					      Point( max.x(), min.y()-ysize,z)
					      ),
				white ) );
    all->add( new GeomMaterial( new GeomLine( Point( min.x(), max.y(), z ),
					      Point( min.x()-xsize, max.y(),z)
					      ),
				white ) );
    all->add( new GeomMaterial( new GeomLine( Point( max.x(), max.y(), z ),
					      Point( max.x()+xsize, max.y(),z)
					      ),
				white ) );
    all->add( new GeomMaterial( new GeomLine( Point( min.x(), max.y(), z ),
					      Point( min.x(), max.y()+ysize,z)
					      ),
				white ) );
    all->add( new GeomMaterial( new GeomLine( Point( max.x(), max.y(), z ),
					      Point( max.x(), max.y()+ysize,z)
					      ),
				white ) );    
					      
  }

  if( m != 0 ) {
    int i;

    char txt[80];
    // add numbers by 10's
    if( m == 1 ) {
      // inside only - place numbers close to field
      for( i = 0; i < numx; i+=10 ) {
	sprintf( txt, "%d", i );
	all->add(new GeomMaterial(new GeomText(txt,
					       Point(i,min.y()-ysize/2.0,z)),
				  white) );
      }
      sprintf( txt, "%d", numx );
      all->add(new GeomMaterial(new GeomText(txt,Point(max.x(),min.y()-ysize/2.0,z)), white) );
      for( i = 0; i < numy; i+=10 ) {
	sprintf( txt, "%d", i );
	all->add(new GeomMaterial(new GeomText(txt,
					       Point(min.x()-xsize/2.0,i,z)),
				  white) );
      }
       sprintf( txt, "%d", numy );
      all->add(new GeomMaterial(new GeomText(txt,Point(min.x()-xsize/2.0,max.y(),z)), white) );

    } else if ( m != 0 ) {
      // outside & both - place numbers further away because of outside lines
      for( i = 0; i < numx; i+=10 ) {
	sprintf( txt, "%d", i );
	all->add(new GeomMaterial(new GeomText(txt,Point(i,min.y()-ysize,z)),
				  white) );
      }
      sprintf( txt, "%d", numx );
      all->add(new GeomMaterial(new GeomText(txt,Point(max.x(),min.y()-ysize,z)), white) );
      for( i = 0; i < numy; i+=10 ) {
	sprintf( txt, "%d", i );
	all->add(new GeomMaterial(new GeomText(txt,Point(min.x()-xsize,i,z)),
				  white) );
      }
      sprintf( txt, "%d", numy );
      all->add(new GeomMaterial(new GeomText(txt,Point(min.x()-xsize,max.y(),z)), white) );
      
    }
    
    int nx = (int)( (double)numx * scalex.get() * pow(10,expx.get()) );
    double denom = (double)(numx+1) * scalex.get() * pow(10,expx.get());

    // add lines between the borders
    for(i = 0; i < nx; i++){
      double x=Interpolate(min.x(), max.x(), double(i+1)/denom );
      if( m == 1 || m == 3 )
	// add inside lines
	all->add( new GeomMaterial( new GeomCylinder( Point(x, min.y(), z),
						      Point(x, max.y(), z),
						      rad.get(), 3, 1 ),
				    matl ) );
      if( m == 2 || m == 3 ) {
	// add outside lines
	all->add( new GeomMaterial(new GeomLine( Point( x, min.y(), z),
						 Point(x,min.y()-ysize,z) ),
				   white ) );
	all->add( new GeomMaterial(new GeomLine( Point( x, max.y(), z),
						 Point(x,max.y()+ysize,z) ),
				   white ) );
      }
    }
    int ny = (int)(scaley.get() * (double)numy * pow(10,expy.get()) );
    denom = (double)(numy+1) * scaley.get() * pow(10,expy.get());
    for(i = 0; i < ny; i++){
      double y=Interpolate(min.y(), max.y(), double(i+1)/denom );
      if( m == 1 || m == 3 )
	// add inside lines
	all->add( new GeomMaterial( new GeomCylinder( Point(min.x(), y, z),
						      Point(max.x(), y, z),
						      rad.get(), 3, 1 ),
				    matl ) );
      if( m == 2 || m == 3 ) {
	// add outside lines
	all->add( new GeomMaterial(new GeomLine( Point( min.x(), y, z),
						 Point(min.x()-xsize,y,z) ),
				   white ) );
	all->add( new GeomMaterial(new GeomLine( Point( max.x(), y, z),
						 Point(max.x()+xsize,y,z) ),
				   white ) );
      }
    } // end for i < ny
  } // end if m != 0
  
  ogeom->delAll();
  ogeom->addObj(all, "Grid Lines");
}
