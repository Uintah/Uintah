
/*
 *  Field3D.cc: The Field3D Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Field3D.h>
#include <NotFinished.h>
#include <Classlib/String.h>
#include <Math/MinMax.h>

#define FIELD3D_VERSION 2

int Field3D::get_nx()
{
    return nx;
}

int Field3D::get_ny()
{
    return ny;
}

int Field3D::get_nz()
{
    return nz;
}

Vector*** Field3D::get_dataptr()
{
    return v_grid.get_dataptr();
}


Field3D::Representation Field3D::get_rep()
{
    return rep;
}

Field3D::FieldType Field3D::get_type()
{
    return fieldtype;
}

Field3D::Field3D()
: rep(RegularGrid), fieldtype(ScalarField), nx(0), ny(0), nz(0), ntetra(0),
  ref_cnt(0)
{
}

Field3D::~Field3D()
{
}

double Field3D::get(int i, int j, int k)
{
    return s_grid(i, j, k);
}

Point Field3D::get_point(int i, int j, int k)
{
    double x=min.x()+diagonal.x()*double(i)/double(nx);
    double y=min.y()+diagonal.y()*double(j)/double(ny);
    double z=min.z()+diagonal.z()*double(k)/double(nz);
    return Point(x,y,z);
}

void Field3D::set_size(int _nx, int _ny, int _nz)
{
    nx=_nx;
    ny=_ny;
    nz=_nz;
    if(fieldtype==ScalarField)
	s_grid.newsize(nx, ny, nz);
    else
	v_grid.newsize(nx, ny, nz);
}

void Field3D::set(int i, int j, int k, const Vector& v)
{
    v_grid(i,j,k)=v;
}

void Field3D::set(int i, int j, int k, double d)
{
    s_grid(i,j,k)=d;
}

void Field3D::set_rep(Representation _rep)
{
    rep=_rep;
}

void Field3D::set_type(FieldType _fieldtype)
{
    fieldtype=_fieldtype;
}

void Field3D::io(Piostream& stream)
{
    int version=stream.begin_class("Field3D", FIELD3D_VERSION);
    int* repp=(int*)&rep;
    stream.io(*repp);
    int* fp=(int*)&fieldtype;
    stream.io(*fp);
    switch(rep){
    case RegularGrid:
	stream.io(nx);
	stream.io(ny);
	stream.io(nz);
	if(stream.reading()){
	    // Allocate the array...
	    set_size(nx, ny, nz);
	}
	if(version==1){
	    min=Point(0,0,0);
	    max=Point(nx-1, ny-1, nz-1);
	} else {
	    stream.io(min);
	    stream.io(max);
	    if(stream.reading()){
		// Set the diagonal
		diagonal=max-min;
	    }
	}
	if(fieldtype==ScalarField){
	    for(int i=0;i<nx;i++){
		for(int j=0;j<ny;j++){
		    for(int k=0;k<nz;k++){
			stream.io(s_grid(i, j, k));
		    }
		}
	    }
	} else {
	    for(int i=0;i<nx;i++){
		for(int j=0;j<ny;j++){
		    for(int k=0;k<nz;k++){
			stream.io(v_grid(i, j, k));
		    }
		}
	    }
	}
	break;
    case TetraHedra:
	NOT_FINISHED("Persistent representation for tetrahedra");
	break;
    }
    stream.end_class();
}

int Field3D::interpolate(const Point& p, double& value)
{
    if(fieldtype != ScalarField)return 0;
    if(rep != RegularGrid){
	NOT_FINISHED("interpolate for non-regular grids");
	return 0;
    }
    Vector pn=p-min;
    double x=pn.x()*nx/diagonal.x();
    double y=pn.y()*ny/diagonal.y();
    double z=pn.z()*nz/diagonal.z();
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
    double x00=Interpolate(s_grid(ix, iy, iz), s_grid(ix1, iy, iz), fx);
    double x01=Interpolate(s_grid(ix, iy, iz1), s_grid(ix1, iy, iz1), fx);
    double x10=Interpolate(s_grid(ix, iy1, iz), s_grid(ix1, iy1, iz), fx);
    double x11=Interpolate(s_grid(ix, iy1, iz1), s_grid(ix1, iy1, iz1), fx);
    double y0=Interpolate(x00, x10, fy);
    double y1=Interpolate(x01, x11, fy);
    value=Interpolate(y0, y1, fz);
    return 1;
}

int Field3D::interpolate(const Point& p, Vector& value)
{
    if(fieldtype != ScalarField)return 0;
    if(rep != RegularGrid){
	NOT_FINISHED("interpolate for non-regular grids");
	return 0;
    }
    Vector pn=p-min;
    double x=pn.x()*nx/diagonal.x();
    double y=pn.y()*ny/diagonal.y();
    double z=pn.z()*nz/diagonal.z();
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
    Vector x00=Interpolate(v_grid(ix1, iy, iz), v_grid(ix, iy, iz), fx);
    Vector x01=Interpolate(v_grid(ix1, iy, iz1), v_grid(ix, iy, iz1), fx);
    Vector x10=Interpolate(v_grid(ix1, iy1, iz), v_grid(ix1, iy1, iz), fx);
    Vector x11=Interpolate(v_grid(ix1, iy1, iz1), v_grid(ix1, iy1, iz1), fx);
    Vector y0=Interpolate(x10, x00, fy);
    Vector y1=Interpolate(x11, x01, fy);
    value=Interpolate(y1, y0, fz);
    return 1;
}

void Field3D::get_minmax(double& min, double& max)
{
    if(fieldtype != ScalarField)return;
    if(rep != RegularGrid){
	NOT_FINISHED("interpolate for non-regular grids");
	return;
    }
    min=max=s_grid(0,0,0);
    for(int i=0;i<nx;i++){
	for(int j=0;j<ny;j++){
	    for(int k=0;k<nz;k++){
		double v=s_grid(i,j,k);
		min=Min(min, v);
		max=Max(max, v);
	    }
	}
    }
}

void Field3D::locate(const Point& p, int& ix, int& iy, int& iz)
{
    if(rep != RegularGrid){
	NOT_FINISHED("interpolate for non-regular grids");
	return;
    }
    Vector pn=p-min;
    double x=pn.x()*nx/diagonal.x();
    double y=pn.y()*ny/diagonal.y();
    double z=pn.z()*nz/diagonal.z();
    ix=(int)x;
    iy=(int)y;
    iz=(int)z;
}
