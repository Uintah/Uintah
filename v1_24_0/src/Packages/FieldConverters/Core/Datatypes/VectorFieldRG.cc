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

/*
 *  VectorFieldRG.cc: Vector Fields defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <FieldConverters/Core/Datatypes/VectorFieldRG.h>
#include <Core/Malloc/Allocator.h>

namespace FieldConverters {

static Persistent* maker()
{
    return scinew VectorFieldRG(0, 0, 0);
}

PersistentTypeID VectorFieldRG::type_id("VectorFieldRG", "VectorField", maker);

VectorFieldRG::VectorFieldRG(int x, int y, int z)
  : VectorField(RegularGrid), nx(x), ny(y), nz(z), grid(x, y, z)
{
}

VectorFieldRG::VectorFieldRG(const VectorFieldRG& copy)
  : VectorField(RegularGrid), nx(copy.nx), ny(copy.ny), nz(copy.nz)
{
  grid.copy(copy.grid);
}

VectorFieldRG::~VectorFieldRG()
{
}

Point VectorFieldRG::get_point(int i, int j, int k)
{
    double x=bmin.x()+diagonal.x()*double(i)/double(nx-1);
    double y=bmin.y()+diagonal.y()*double(j)/double(ny-1);
    double z=bmin.z()+diagonal.z()*double(k)/double(nz-1);
    return Point(x,y,z);
}

bool VectorFieldRG::locate(int *ijk, const Point& p)
{
    Vector pn=p-bmin;
    double dx=diagonal.x();
    double dy=diagonal.y();
    double dz=diagonal.z();
    double x=pn.x()*(nx-1)/dx;
    double y=pn.y()*(ny-1)/dy;
    double z=pn.z()*(nz-1)/dz;
    ijk[0]=(int)x;
    ijk[1]=(int)y;
    ijk[2]=(int)z;

    if (ijk[0] < 0 || ijk[0] >= nx ||
	ijk[1] < 0 || ijk[1] >= ny ||
	ijk[2] < 0 || ijk[2] >= nz)
    {
      return false;
    }
    else
    {
      return true;
    }
}

#define VectorFIELDRG_VERSION 1

void VectorFieldRG::io(Piostream& stream)
{

    /*int version=*/
    stream.begin_class("VectorFieldRG", VectorFIELDRG_VERSION);

    // Do the base class first...
    VectorField::io(stream);

    // Save these since the VectorField doesn't
    SCIRun::Pio(stream, bmin);
    SCIRun::Pio(stream, bmax);
    if(stream.reading()){
	have_bounds=1;
	diagonal=bmax-bmin;
    }

    // Save the rest..
    SCIRun::Pio(stream, nx);
    SCIRun::Pio(stream, ny);
    SCIRun::Pio(stream, nz);
    SCIRun::Pio(stream, grid);
    stream.end_class();
}	

void VectorFieldRG::compute_bounds()
{
    // Nothing to do - we store the bounds in the base class...
}

int VectorFieldRG::interpolate(const Point& p, Vector& value, int&, int)
{
    return interpolate(p, value);
}

int VectorFieldRG::interpolate(const Point& p, Vector& value)
{

    Vector pn=p-bmin;
    double dx=diagonal.x();
    double dy=diagonal.y();
    double dz=diagonal.z();

    ASSERT( dx > 0 && dy > 0 && dz > 0 );

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

    Vector x00=Interpolate(grid(ix, iy, iz), grid(ix1, iy, iz), fx);
    Vector x01=Interpolate(grid(ix, iy, iz1), grid(ix1, iy, iz1), fx);
    Vector x10=Interpolate(grid(ix, iy1, iz), grid(ix1, iy1, iz), fx);
    Vector x11=Interpolate(grid(ix, iy1, iz1), grid(ix1, iy1, iz1), fx);
    Vector y0=Interpolate(x00, x10, fy);
    Vector y1=Interpolate(x01, x11, fy);
    value=Interpolate(y0, y1, fz);
    return 1;
}

void VectorFieldRG::set_bounds(const Point& min,
			       const Point& max)
{
    bmin=min;
    bmax=max;
    have_bounds=1;
    diagonal=bmax-bmin;
}

void VectorFieldRG::get_boundary_lines(Array1<Point>& lines)
{
    Point min, max;
    get_bounds(min, max);
    int i;
    for(i=0;i<4;i++){
	double x=(i&1)?min.x():max.x();
	double y=(i&2)?min.y():max.y();
	lines.add(Point(x, y, min.z()));
	lines.add(Point(x, y, max.z()));
    }
    for(i=0;i<4;i++){
	double y=(i&1)?min.y():max.y();
	double z=(i&2)?min.z():max.z();
	lines.add(Point(min.x(), y, z));
	lines.add(Point(max.x(), y, z));
    }
    for(i=0;i<4;i++){
	double x=(i&1)?min.x():max.x();
	double z=(i&2)?min.z():max.z();
	lines.add(Point(x, min.y(), z));
	lines.add(Point(x, max.y(), z));
    }
}


VectorField* VectorFieldRG::clone()
{
    return scinew VectorFieldRG(*this);
}

} // end namespace FieldConverters


