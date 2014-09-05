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
 *  ScalarFieldRGBase.cc: Scalar Fields defined on a Regular grid base class
 *
 *  Written by:
 *   Steven G. Parker (& David Weinstein)
 *   Department of Computer Science
 *   University of Utah
 *   March 1994 (& January 1996)
 *
 *  Copyright (C) 1994, 1996 SCI Group 
 */

#include <FieldConverters/Core/Datatypes/ScalarFieldRGBase.h>
#include <iostream>
using std::cerr;
using std::endl;

#ifdef _WIN32
#include <stdlib.h>
#define drand48() rand()
#endif

namespace FieldConverters {

PersistentTypeID ScalarFieldRGBase::type_id("ScalarFieldRGBase", "ScalarField", 0);


ScalarFieldRGBase::ScalarFieldRGBase(Representation r, int x, int y, int z)
  : ScalarField(RegularGridBase),
    nx(x), ny(y), nz(z),
    is_augmented(0),
    next(0),
    rep(r)
{
}

ScalarFieldRGBase::ScalarFieldRGBase(const ScalarFieldRGBase& copy)
: ScalarField(copy), nx(copy.nx), ny(copy.ny), nz(copy.nz), 
  is_augmented(0), next(0),
  rep(copy.rep)
{
}

ScalarFieldRGBase::~ScalarFieldRGBase()
{
  if (next)
    delete next; // propogate down the links...
}


const char *
ScalarFieldRGBase::getType() const {
    if (rep==Double)
        return ("double");
    else if (rep==Float)
        return ("float");
    else if (rep==Int)
        return ("int");
    else if (rep==Short)
        return ("short");
    else if (rep==Ushort)
        return ("ushort");
    else if (rep==Char)
        return ("char");
    else if (rep==Uchar)
        return ("uchar");
    else if (rep==Void)
	return ("void");
    else
        return ("unknown");
}

ScalarFieldRGdouble* ScalarFieldRGBase::getRGDouble()
{
    if (rep==Double)
	return (ScalarFieldRGdouble*) this;
    else
	return 0;
}

ScalarFieldRGfloat* ScalarFieldRGBase::getRGFloat()
{
    if (rep==Float)
	return (ScalarFieldRGfloat*) this;
    else
	return 0;
}

ScalarFieldRGint* ScalarFieldRGBase::getRGInt()
{
    if (rep==Int)
	return (ScalarFieldRGint*) this;
    else
	return 0;
}

ScalarFieldRGshort* ScalarFieldRGBase::getRGShort()
{
    if (rep==Short)
	return (ScalarFieldRGshort*) this;
    else
	return 0;
}
ScalarFieldRGushort* ScalarFieldRGBase::getRGUshort()
{
    if (rep==Ushort)
	return (ScalarFieldRGushort*) this;
    else
	return 0;
}

ScalarFieldRGchar* ScalarFieldRGBase::getRGChar()
{
    if (rep==Char)
	return (ScalarFieldRGchar*) this;
    else
	return 0;
}

ScalarFieldRGuchar* ScalarFieldRGBase::getRGUchar()
{
    if (rep==Uchar)
	return (ScalarFieldRGuchar*) this;
    else
	return 0;
}

Point ScalarFieldRGBase::get_point(int i, int j, int k)
{
    double x=bmin.x()+diagonal.x()*double(i)/double(nx-1);
    double y=bmin.y()+diagonal.y()*double(j)/double(ny-1);
    double z=bmin.z()+diagonal.z()*double(k)/double(nz-1);
    if (!is_augmented)
      return Point(x,y,z);
    return (*aug_data.get_rep())(i,j,k); // just grab it...
}

void ScalarFieldRGBase::set_bounds(const Point &min, const Point &max) {
    bmin=min;
    bmax=max;
    diagonal=max-min;
}
    
bool ScalarFieldRGBase::locate(int *loc, const Point& p)
{
    Vector pn=p-bmin;
    double dx=diagonal.x();
    double dy=diagonal.y();
    double dz=diagonal.z();
    double x=pn.x()*(nx-1)/dx;
    double y=pn.y()*(ny-1)/dy;
    double z=pn.z()*(nz-1)/dz;
    loc[0]=(int)x;
    loc[1]=(int)y;
    loc[2]=(int)z;
    
    if (loc[0] < 0 || loc[0] >= nx ||
	loc[1] < 0 || loc[1] >= ny ||
	loc[2] < 0 || loc[2] >= nz)
    {
      return false;
    }
    else
    {
      return true;
    }
}


void ScalarFieldRGBase::midLocate(const Point& p, int& ix, int& iy, int& iz)
{
    Vector pn=p-bmin;
    double dx=diagonal.x();
    double dy=diagonal.y();
    double dz=diagonal.z();
    double x=pn.x()*(nx-1)/dx+.5;
    double y=pn.y()*(ny-1)/dy+.5;
    double z=pn.z()*(nz-1)/dz+.5;
    ix=(int)x;
    iy=(int)y;
    iz=(int)z;
}

int
ScalarFieldRGBase::get_voxel( const Point& p, Point& ivoxel )
{
    Vector pn=p-bmin;
    
    ivoxel.x( int( pn.x()*(nx-1)/diagonal.x() ) );
    ivoxel.y( int( pn.y()*(ny-1)/diagonal.y() ) );
    ivoxel.z( int( pn.z()*(nz-1)/diagonal.z() ) );

    if(ivoxel.x()<0 || ivoxel.x()>=nx)
      cerr << "nx = " << nx << "  " << ivoxel << endl;
    if(ivoxel.y()<0 || ivoxel.y()>=ny)
      cerr << "ny = " << ny << "  " << ivoxel << endl;
    if(ivoxel.z()<0 || ivoxel.z()>=nz)
      cerr << "nz = " << nz << "  " << ivoxel << endl;

    if(ivoxel.x()<0 || ivoxel.x()>=nx)return 0;
    if(ivoxel.y()<0 || ivoxel.y()>=ny)return 0;
    if(ivoxel.z()<0 || ivoxel.z()>=nz)return 0;
 
   return 1;
}


#define ScalarFieldRGBase_VERSION 1

void ScalarFieldRGBase::io(Piostream& stream)
{

    /*int version=*/
    stream.begin_class("ScalarFieldRGBase", ScalarFieldRGBase_VERSION);

    // Do the base class first...
    ScalarField::io(stream);

    // Save these since the ScalarField doesn't
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
    stream.end_class();
}	

void ScalarFieldRGBase::compute_bounds()
{
    // Nothing to do - we store the bounds in the base class...
}

void ScalarFieldRGBase::get_boundary_lines(Array1<Point>& lines)
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


inline Point RandomPoint(Vector& dv, Point& o) 
{
  return Point(o.x() + drand48()*dv.x(),
	       o.y() + drand48()*dv.y(),
	       o.z() + drand48()*dv.z());
}


void ScalarFieldRGBase::compute_samples(int nsamp)
{
  cerr << "Computing Samples RG\n";

  Vector dv(diagonal.x()/(nx-1),
	    diagonal.y()/(ny-1),
	    diagonal.z()/(nz-1));
  double celvol = dv.x()*dv.y()*dv.z();
  
  aug_elems.resize((nx-1)*(ny-1)*(nz-1));

  for(int i=0;i<(nx-1)*(ny-1)*(nz-1);i++) {
    aug_elems[i].importance = celvol;
  }
  samples.resize(nsamp);

  cerr << "Volume: " << celvol*(nx-1)*(ny-1)*(nz-1) << "\n";

}

void ScalarFieldRGBase::cell_pos(int i, int& x, int& y, int& z)
{
  z = i/((ny-1)*(nx-1));
  int delta = i-z*(ny-1)*(nx-1);
  y = delta/(nx-1);
  delta = delta - y*(nx-1);
  x = delta;  // this is the coord to use...
}

void ScalarFieldRGBase::distribute_samples()
{
  // number is already assigned, but weights have changed
  cerr << "Distributing Samples RG\n";

  Vector dv(diagonal.x()/(nx-1),
	    diagonal.y()/(ny-1),
	    diagonal.z()/(nz-1));
  double celvol = dv.x()*dv.y()*dv.z();

  cerr << diagonal << " vol: " << celvol << "\n";

  double total_importance =0.0;

  Array1<double> psum(aug_elems.size());
  
  int i;
  for(i=0;i<aug_elems.size();i++) {
    total_importance += aug_elems[i].importance;
    psum[i] = total_importance;
    aug_elems[i].pt_samples.remove_all();
  }

  // now just jump into the prefix sum table...
  // this is a bit faster, especialy initialy...

  int pi=0;
  int nsamp = samples.size();
  double factor = 1.0/(nsamp-1)*total_importance;

  //  double jitter = 1.0; // can jitter half an interval in either direction

  for(i=0;i<nsamp;i++) {
    double ni = i + drand48()-0.5;
    if (ni<0) ni = 0;
    if (ni >= (nsamp-1)) ni = nsamp-1; // jittered comb...

    double val = (ni*factor);
    while ( (pi < aug_elems.size()) && 
	   (psum[pi] < val))
      pi++;
    if (pi == aug_elems.size()) {
      cerr << "Over flow!\n";
    } else {
      aug_elems[pi].pt_samples.add(i);
      int x,y,z;

      cell_pos(pi,x,y,z); // get coordinates for this cell...

      Point nv(bmin.x()+x*dv.x(),
	       bmin.y()+y*dv.y(),
	       bmin.z()+z*dv.z());

      samples[i].loc = RandomPoint(dv,nv);
    }
  }
}

} // End namespace FieldConverters

