
/*
 *  ScalarFieldRG.h: Scalar Fields defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <ScalarFieldRG.h>
#include <Classlib/String.h>

static Persistent* maker()
{
    return new ScalarFieldRG;
}

PersistentTypeID ScalarFieldRG::typeid("ScalarFieldRG", "ScalarField", maker);

ScalarFieldRG::ScalarFieldRG()
: ScalarField(RegularGrid), nx(0), ny(0), nz(0)
{
}

ScalarFieldRG::~ScalarFieldRG()
{
}

Point ScalarFieldRG::get_point(int i, int j, int k)
{
    double x=bmin.x()+diagonal.x()*double(i)/double(nx);
    double y=bmin.y()+diagonal.y()*double(j)/double(ny);
    double z=bmin.z()+diagonal.z()*double(k)/double(nz);
    return Point(x,y,z);
}

void ScalarFieldRG::locate(const Point& p, int& ix, int& iy, int& iz)
{
    Vector pn=p-bmin;
    double dx=diagonal.x();
    double dy=diagonal.y();
    double dz=diagonal.z();
    double x=pn.x()*nx/dx;
    double y=pn.y()*ny/dy;
    double z=pn.z()*nz/dz;
    ix=(int)x;
    iy=(int)y;
    iz=(int)z;
}

#define SCALARFIELDRG_VERSION 1

void ScalarFieldRG::io(Piostream& stream)
{
    int version=stream.begin_class("ScalarFieldRG", SCALARFIELDRG_VERSION);
    // Do the base class first...
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
    Pio(stream, grid);
    stream.end_class();
}	
