
/*
 *  ScalarFieldRGCC.cc:  Scalar Field defined on a Regular grid
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

#include <SCICore/Datatypes/ScalarFieldRGCC.h>
//#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>

namespace SCICore {
namespace Datatypes {

static Persistent* maker()
{
    return scinew ScalarFieldRGCC;
}

PersistentTypeID ScalarFieldRGCC::type_id("ScalarFieldRGCC", "ScalarField", maker);

ScalarFieldRGCC::ScalarFieldRGCC()
: ScalarFieldRG()
{
}

ScalarFieldRGCC::ScalarFieldRGCC(const ScalarFieldRGCC& copy)
: ScalarFieldRG(copy)
{
}

ScalarFieldRGCC::~ScalarFieldRGCC()
{
}


#define ScalarFieldRGCC_VERSION 2

void ScalarFieldRGCC::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;
    using SCICore::Geometry::Pio;

    int version=stream.begin_class("ScalarFieldRGCC", ScalarFieldRGCC_VERSION);
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


Vector ScalarFieldRGCC::gradient(const Point& p)
{
  /*            bmax ( boundary max point )
		|
    -------------
    | + | + | + |    in this example nx = 3, ny = 3, nz = 3
    -------------    diagonal = bmax - bmin
    | + | + | + |
    -------------
    | + | + | + |
    -------------
    |
    bmin (boundary min point)
  */

    
  Vector pn=p-bmin; // vector from field origin to point p

  // compute index into field
  int ix=(int)(pn.x()*nx/diagonal.x());
  int iy=(int)(pn.y()*ny/diagonal.y());
  int iz=(int)(pn.z()*nz/diagonal.z());

  return gradient( ix, iy, iz );
}

int ScalarFieldRGCC::interpolate(const Point& p, double& value, int&,
				   double epsilon1, double epsilon2,
				   int) {
    return interpolate(p, value, epsilon1, epsilon2);
}

int ScalarFieldRGCC::interpolate(const Point& p, double& value, double,
				   double)
{
    Vector pn=p-bmin;
    int ix=(int)(pn.x()*nx/diagonal.x());
    int iy=(int)(pn.y()*ny/diagonal.y());
    int iz=(int)(pn.z()*nz/diagonal.z());
    if(ix<0 || ix>=nx)return 0;
    if(iy<0 || iy>=ny)return 0;
    if(iz<0 || iz>=nz)return 0;
    value=grid(ix, iy, iz);
    return 1;
}

ScalarField* ScalarFieldRGCC::clone()
{
    return scinew ScalarFieldRGCC(*this);
}



Vector 
ScalarFieldRGCC::gradient(int ix, int iy, int iz)
{
  // this uses central differences...

  // compute Deltas
  double dx = diagonal.x()/nx;
  double dy = diagonal.y()/ny;
  double dz = diagonal.z()/nz;

  // return the zero vector if the point lies outside the boundaries
  if(ix<0 || ix>=nx)return Vector(0,0,0);
  if(iy<0 || iy>=ny)return Vector(0,0,0);
  if(iz<0 || iz>=nz)return Vector(0,0,0);



  // compute gradients
  double gradX, gradY, gradZ;

  // compute the X component
  if (ix == 0){			// point is in left face cell
    gradX = ( -3.0 * grid(ix,iy,iz) + 4.0*grid(ix+1,iy,iz)
	      - grid(ix+2,iy,iz))/ dx;
  } else if( ix == nx - 1){	// point is in right face cell
    gradX = ( 3.0 * grid(ix,iy,iz) - 4.0*grid(ix-1,iy,iz)
	      - grid(ix-2,iy,iz))/ dx;
  } else {			// point is NOT in left or right face cell
    gradX = (grid(ix+1, iy, iz) - grid(ix-1, iy, iz))/(2.0 * dx );
  }
  // compute the Y component
  if (iy == 0){			// point is in bottom face cell
    gradY = ( -3.0 * grid(ix,iy,iz) + 4.0*grid(ix,iy+1,iz)
	      - grid(ix,iy+2,iz))/ dy;
  } else if( iy == ny - 1){	// point is in top face cell
    gradY = ( 3.0 * grid(ix,iy,iz) - 4.0*grid(ix,iy-1,iz)
	      - grid(ix,iy-2,iz))/ dy;
  } else {			// point is NOT in top or bottom face cell
    gradY = (grid(ix, iy+1, iz) - grid(ix, iy-1, iz))/(2.0 * dy );
  }
  // compute the Z component
  if (iz == 0){			// point is in a back face cell
    gradZ = ( -3.0 * grid(ix,iy,iz) + 4.0*grid(ix,iy,iz+1)
	      - grid(ix,iy,iz+2))/ dz;
  } else if( iz == nz - 1){	// point is in a front face cell
    gradZ = ( 3.0 * grid(ix,iy,iz) - 4.0*grid(ix,iy,iz-1)
	      - grid(ix,iy,iz-2))/ dz;
  } else {			//point is NOT in a front or back face cell
    gradZ = (grid(ix, iy, iz+1) - grid(ix, iy, iz-1))/(2.0 * dz );
  }
    
  return Vector( gradX, gradY, gradZ );

}

// this stuff is for augmenting the random distributions...

void ScalarFieldRGCC::fill_gradmags() // these guys ignor the vf
{
  
  total_gradmag = 0.0;
  
  int nelems = (nx)*(ny)*(nz); // must be 3d for now...

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

} // End namespace Datatypes
} // End namespace SCICore


