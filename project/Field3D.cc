
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

Field3DHandle::Field3DHandle(Field3D* rep)
: rep(rep)
{
    if(rep)
	rep->ref_cnt++;
}

Field3DHandle::~Field3DHandle()
{
    if(rep && --rep->ref_cnt==0)
	delete rep;
}

Field3D* Field3DHandle::operator->() const
{
    return rep;
}

Field3D::Field3D()
: rep(RegularGrid), fieldtype(ScalarField), nx(0), ny(0), nz(0), ntetra(0)
{
}

Field3D::~Field3D()
{
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
    stream.begin_class("Field3D");
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
	    if(fieldtype==ScalarField)
		s_grid.newsize(nx, ny, nz);
	    else
		v_grid.newsize(nx, ny, nz);
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

