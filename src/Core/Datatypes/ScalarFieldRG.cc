
/*
 *  ScalarFieldRG.cc: double Scalar Field defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker (& David Weinstein)
 *   Department of Computer Science
 *   University of Utah
 *   March 1994 (& January 1996)
 *
 *  Copyright (C) 1994 SCI Group
 *
 *  WARNING: This file was automatically generated from:
 *           ScalarFieldRGtype.cc (<- "type" should be in all caps
 *           but I don't want it replaced by the sed during
 *           the generation process.)
 */

#include <CoreDatatypes/ScalarFieldRG.h>
//#include <Containers/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

namespace SCICore {
namespace CoreDatatypes {

static Persistent* maker()
{
    return scinew ScalarFieldRG;
}

PersistentTypeID ScalarFieldRG::type_id("ScalarFieldRG", "ScalarField", maker);

ScalarFieldRG::ScalarFieldRG()
: ScalarFieldRGBase("double")
{
}

ScalarFieldRG::ScalarFieldRG(const ScalarFieldRG& copy)
: ScalarFieldRGBase(copy), grid(copy.grid)
{
}

ScalarFieldRG::~ScalarFieldRG()
{
}

void ScalarFieldRG::resize(int x, int y, int z) {
    nx=x; ny=y; nz=z;
    grid.newsize(x,y,z);
}

#define ScalarFieldRG_VERSION 2

void ScalarFieldRG::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;
    using SCICore::Geometry::Pio;

    int version=stream.begin_class("ScalarFieldRG", ScalarFieldRG_VERSION);
    if(version == 1){
	// From before, when the ScalarFieldRGBase didn't exist...
	ScalarField::io(stream);

	// Save these since the ScalarField doesn't
	Pio(stream, bmin);
	Pio(stream, bmax);
	if(stream.reading()){
	    have_bounds=1;
	    diagonal=bmax-bmin;
	}

	// Save the rest..
	Pio(stream, nx);
	Pio(stream, ny);
	Pio(stream, nz);
    } else {
	// Do the base class first...
	ScalarFieldRGBase::io(stream);
    }
    Pio(stream, grid);
    stream.end_class();
}

void ScalarFieldRG::compute_minmax()
{
    using SCICore::Math::Min;
    using SCICore::Math::Max;

    if(nx==0 || ny==0 || nz==0)return;
    double min=grid(0,0,0);
    double max=grid(0,0,0);
    for(int i=0;i<nx;i++){
	for(int j=0;j<ny;j++){
	    for(int k=0;k<nz;k++){
		double val=grid(i,j,k);
		min=Min(min, val);
		max=Max(max, val);
	    }
	}
    }
    data_min=min;
    data_max=max;
}

Vector ScalarFieldRG::gradient(const Point& p)
{
    using SCICore::Math::Interpolate;

    Vector pn=p-bmin;
    double diagx=diagonal.x();
    double diagy=diagonal.y();
    double diagz=diagonal.z();
    double x=pn.x()*(nx-1)/diagx;
    double y=pn.y()*(ny-1)/diagy;
    double z=pn.z()*(nz-1)/diagz;
    int ix=(int)x;
    int iy=(int)y;
    int iz=(int)z;
    int ix1=ix+1;
    int iy1=iy+1;
    int iz1=iz+1;
    if(ix<0 || ix1>=nx)return Vector(0,0,0);
    if(iy<0 || iy1>=ny)return Vector(0,0,0);
    if(iz<0 || iz1>=nz)return Vector(0,0,0);
    double fx=x-ix;
    double fy=y-iy;
    double fz=z-iz;
    double z00=Interpolate(grid(ix, iy, iz), grid(ix, iy, iz1), fz);
    double z01=Interpolate(grid(ix, iy1, iz), grid(ix, iy1, iz1), fz);
    double z10=Interpolate(grid(ix1, iy, iz), grid(ix1, iy, iz1), fz);
    double z11=Interpolate(grid(ix1, iy1, iz), grid(ix1, iy1, iz1), fz);
    double yy0=Interpolate(z00, z01, fy);
    double yy1=Interpolate(z10, z11, fy);
    double dx=(yy1-yy0)*(nx-1)/diagonal.x();
    double x00=Interpolate(grid(ix, iy, iz), grid(ix1, iy, iz), fx);
    double x01=Interpolate(grid(ix, iy, iz1), grid(ix1, iy, iz1), fx);
    double x10=Interpolate(grid(ix, iy1, iz), grid(ix1, iy1, iz), fx);
    double x11=Interpolate(grid(ix, iy1, iz1), grid(ix1, iy1, iz1), fx);
    double y0=Interpolate(x00, x10, fy);
    double y1=Interpolate(x01, x11, fy);
    double dz=(y1-y0)*(nz-1)/diagonal.z();
    double z0=Interpolate(x00, x01, fz);
    double z1=Interpolate(x10, x11, fz);
    double dy=(z1-z0)*(ny-1)/diagonal.y();
    return Vector(dx, dy, dz);
}

int ScalarFieldRG::interpolate(const Point& p, double& value, int&,
				   double epsilon1, double epsilon2,
				   int) {
    return interpolate(p, value, epsilon1, epsilon2);
}

int ScalarFieldRG::interpolate(const Point& p, double& value, double,
				   double)
{
    using SCICore::Math::Interpolate;

    Vector pn=p-bmin;
    double dx=diagonal.x();
    double dy=diagonal.y();
    double dz=diagonal.z();
    double x=pn.x()*(nx-1)/dx;
    double y=pn.y()*(ny-1)/dy;
    double z=pn.z()*(nz-1)/dz;
    int ix=(int)x;
    int iy=(int)y;
    int iz=(int)z;
    int ix1=ix+1;
    int iy1=iy+1;
    int iz1=iz+1;
    if(ix<0 || ix1>=nx)return 0;
    if(iy<0 || iy1>=ny)return 0;
    if(iz<0 || iz1>=nz)return 0;
    double fx=x-ix;
    double fy=y-iy;
    double fz=z-iz;
    double x00=Interpolate(grid(ix, iy, iz), grid(ix1, iy, iz), fx);
    double x01=Interpolate(grid(ix, iy, iz1), grid(ix1, iy, iz1), fx);
    double x10=Interpolate(grid(ix, iy1, iz), grid(ix1, iy1, iz), fx);
    double x11=Interpolate(grid(ix, iy1, iz1), grid(ix1, iy1, iz1), fx);
    double y0=Interpolate(x00, x10, fy);
    double y1=Interpolate(x01, x11, fy);
    value=Interpolate(y0, y1, fz);
    return 1;
}

ScalarField* ScalarFieldRG::clone()
{
    return scinew ScalarFieldRG(*this);
}

Vector
ScalarFieldRG::get_normal( const Point& ivoxel )
{
  return get_normal( ivoxel.x(), ivoxel.y(), ivoxel.z() );
}

Vector
ScalarFieldRG::get_normal( int x, int y, int z )
{
  Vector normal;

  if ( x == 0 )
    normal.x( ( get_value( x+1, y, z ) - get_value( x, y, z ) ) / 2 );
  else if ( x == nx-1 )
    normal.x( ( get_value( x, y, z ) - get_value( x-1, y, z ) ) / 2 );
  else
    normal.x( ( get_value( x-1, y, z ) + get_value( x+1, y, z ) ) / 2 );
    
  
  if ( y == 0 )
    normal.y( ( get_value( x, y+1, z ) - get_value( x, y, z ) ) / 2 );
  else if ( y == ny-1 )
    normal.y( ( get_value( x, y, z ) - get_value( x, y-1, z ) ) / 2 );
  else
    normal.y( ( get_value( x, y-1, z ) + get_value( x, y+1, z ) ) / 2 );
    
  
  if ( z == 0 )
    normal.z( ( get_value( x, y, z+1 ) - get_value( x, y, z ) ) / 2 );
  else if ( z == nz-1 )
    normal.z( ( get_value( x, y, z ) - get_value( x, y, z-1 ) ) / 2 );
  else
    normal.z( ( get_value( x, y, z-1 ) + get_value( x, y, z+1 ) ) / 2 );

  return normal;
}

Vector 
ScalarFieldRG::gradient(int x, int y, int z)
{
  // this tries to use central differences...

  Vector rval; // return value

  Vector h(0.5*(nx-1)/diagonal.x(),
	   0.5*(ny-1)/diagonal.y(),
	   0.5*(nz-1)/diagonal.z());
  // h is distances one over between nodes in each dimension...

  if (!x || (x == nx-1)) { // boundary...
    if (!x) {
      rval.x(2*(grid(x+1,y,z)-grid(x,y,z))*h.x()); // end points are rare
    } else {
      rval.x(2*(grid(x,y,z)-grid(x-1,y,z))*h.x());
    }
  } else { // just use central diferences...
    rval.x((grid(x+1,y,z)-grid(x-1,y,z))*h.x());
  }

  if (!y || (y == ny-1)) { // boundary...
    if (!y) {
      rval.y(2*(grid(x,y+1,z)-grid(x,y,z))*h.y()); // end points are rare
    } else {
      rval.y(2*(grid(x,y,z)-grid(x,y-1,z))*h.y());
    }
  } else { // just use central diferences...
    rval.y((grid(x,y+1,z)-grid(x,y-1,z))*h.y());
  }

  if (!z || (z == nz-1)) { // boundary...
    if (!z) {
      rval.z(2*(grid(x,y,z+1)-grid(x,y,z))*h.z()); // end points are rare
    } else {
      rval.z(2*(grid(x,y,z)-grid(x,y,z-1))*h.z());
    }
  } else { // just use central diferences...
    rval.z((grid(x,y,z+1)-grid(x,y,z-1))*h.z());
  }

  return rval;

}

// this stuff is for augmenting the random distributions...

void ScalarFieldRG::fill_gradmags() // these guys ignor the vf
{
  cerr << " Filling Gradient Magnitudes...\n";
  
  total_gradmag = 0.0;
  
  int nelems = (nx-1)*(ny-1)*(nz-1); // must be 3d for now...

  grad_mags.resize(nelems);

  // MAKE PARALLEL

  for(int i=0;i<nelems;i++) {
    int x,y,z;

    cell_pos(i,x,y,z); // x,y,z is lleft corner...

    Vector vs[8]; // 8 points define a cell...

    vs[0] = gradient(x,y,z);
    vs[1] = gradient(x,y+1,z);
    vs[2] = gradient(x,y,z+1);
    vs[3] = gradient(x,y+1,z+1);

    vs[4] = gradient(x+1,y,z);
    vs[5] = gradient(x+1,y+1,z);
    vs[6] = gradient(x+1,y,z+1);
    vs[7] = gradient(x+1,y+1,z+1);

    // should try different types of averages...
    // average magnitudes and directions seperately???
    // for this we just need magnitudes though.

    double ml=0;
    for(int j=0;j<8;j++) {
      ml += vs[j].length();
    }  

    grad_mags[i] = ml/8.0;;
    
    total_gradmag += grad_mags[i];
  }
}

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:24  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:39  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/05/06 19:55:50  dav
// added back .h files
//
// Revision 1.1  1999/04/25 04:07:12  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

