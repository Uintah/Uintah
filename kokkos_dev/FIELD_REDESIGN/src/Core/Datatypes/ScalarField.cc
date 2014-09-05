//static char *id="@(#) $Id$";

/*
 *  ScalarField.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Containers/String.h>
#include <iostream>
using std::cerr;

namespace SCICore {
namespace Datatypes {

PersistentTypeID ScalarField::type_id("ScalarField", "Datatype", 0);

ScalarField::ScalarField(Representation rep)
  : have_bounds(0), have_minmax(0), rep(rep), separate_raw(0), raw_filename("")
{
}

ScalarField::~ScalarField()
{
}

ScalarFieldRG* ScalarField::getRG()
{
    if(rep==RegularGrid)
	return (ScalarFieldRG*)this;
    else if (rep==RegularGridBase) {
	ScalarFieldRGBase *sfrgb=(ScalarFieldRGBase*) this;
	if (sfrgb->getRGDouble()) return (ScalarFieldRG*)this; else return 0;
    } else
	return 0;
}

ScalarFieldUG* ScalarField::getUG()
{
    if(rep==UnstructuredGrid)
	return (ScalarFieldUG*)this;
    else
	return 0;
}

ScalarFieldHUG* ScalarField::getHUG()
{
    if(rep==HexGrid)
	return (ScalarFieldHUG*)this;
    else
	return 0;
}

ScalarFieldRGBase* ScalarField::getRGBase()
{
cerr << "rep="<<rep<<"\n";
    if(rep==RegularGridBase)
	return (ScalarFieldRGBase*)this;
    else
	return 0;
}

void ScalarField::set_minmax(double min, double max)
{
    have_minmax=1;
    data_min=min;
    data_max=max;
}

void ScalarField::get_minmax(double& min, double& max)
{
    if(!have_minmax){
	compute_minmax();
	have_minmax=1;
    }
    min=data_min;
    max=data_max;
}

double ScalarField::longest_dimension()
{
    using SCICore::Math::Max;

    if(!have_bounds){
	compute_bounds();
	have_bounds=1;
	diagonal=bmax-bmin;
    }
    return Max(diagonal.x(), diagonal.y(), diagonal.z());
}

void ScalarField::get_bounds(Point& min, Point& max)
{
    if(!have_bounds){
	compute_bounds();
	have_bounds=1;
	diagonal=bmax-bmin;
    }
    max=bmax;
    min=bmin;
}

void ScalarField::set_bounds(const Point& min, const Point& max)
{
  bmax=max;
  bmin=min;
  have_bounds=1;
  diagonal=bmax-bmin;
}

// stuff for random distributions

void ScalarField::compute_samples(int nsamp)
{
  cerr << nsamp << "In quasi-virtual base class ScalarField::compute_samples\n";
}

void ScalarField::distribute_samples()
{
  cerr << "In quasi-virtual base class ScalarField::distribute_samples\n";
}

void ScalarField::fill_gradmags()
{
  cerr << "In quasi-virtual base class  ScalarField::fill_gradmags\n";
}

void ScalarField::over_grad_augment(double vol_wt, double grad_wt, 
				      double /*crit_scale*/)
{
  // this is done in 2 passes, first just get gradients and compute totals
  double vol_total=0.0;
  double grad_total=0.0;
  Array1<double> grads(grad_mags.size());
  double max=-1;
  Array1<int> zeros;
  
  int i;
  for(i=0;i<grad_mags.size();i++) {
    grads[i] = grad_mags[i]; // already compute this...

    if (grads[i] == 0.0) {
      cerr << "Have a 0...\n";
      zeros.add(i);
    } else {
      grads[i] = 1.0/grads[i];
      if (grads[i] > max)
	max = grads[i];
    }
    vol_total += aug_elems[i].importance;
    grad_total += grad_mags[i];
  }

  // ok...  now handle zeros...
  
  for(i=0;i<zeros.size();i++) {
    grad_total += max;
    grads[zeros[i]] = max;
  }
  
  // now do a second pass, normalizing the above...
  
  // normalization is done through recomputing the weights...
  
  vol_wt /= vol_total;
  grad_wt /= grad_total;
  
  for(i=0;i<aug_elems.size();i++) {
    aug_elems[i].importance =
      vol_wt*aug_elems[i].importance + grads[i]*grad_wt;
  }

  cerr << vol_total << " Volume " << grad_total << " Gradient\n";
}

void ScalarField::grad_augment(double vol_wt, double grad_wt)
{
  // this is done in 2 passes, first just get gradients and compute totals
  double vol_total=0.0;
  double grad_total=total_gradmag;
  
  int i;
  for(i=0;i<aug_elems.size();i++) {
    vol_total += aug_elems[i].importance;
  }

  // now do a second pass, normalizing the above...

  // normalization is done through recomputing the weights...

  vol_wt /= vol_total;
  grad_wt /= grad_total;

  for(i=0;i<aug_elems.size();i++) {
      aug_elems[i].importance =
	vol_wt*aug_elems[i].importance + grad_mags[i]*grad_wt;
  }

  cerr << vol_total << " Volume " << grad_total << " Gradient\n";
}

void ScalarField::hist_grad_augment(double vol_wt, double grad_wt,
				      const int HSIZE)
{
  Array1<int> histogram;

  histogram.resize(HSIZE);

  int i;
  for(i=0;i<HSIZE;i++) {
    histogram[i] = 0;
  }

  double vol_total=0.0;
  double grad_total=0;
  
  double min=122434435, max=-1.0;


  for(i=0;i<aug_elems.size();i++) {
    if (grad_mags[i] < min) min = grad_mags[i];
    if (grad_mags[i] > max) max = grad_mags[i];
    
    vol_total += aug_elems[i].importance;
  }

  double temp = 1.0/(max-min)*(HSIZE-1);

  for(i=0;i<aug_elems.size();i++) {
      histogram[(grad_mags[i]-min)*temp]++;
  }
  
  Array1<double> S;
  S.resize(HSIZE);

  for( i=0;i<HSIZE;i++) {
    double Number=0.0;
    for (int j=0; j<=i; j++) Number+=histogram[j];
    S[i] = Number/aug_elems.size();
  }

  Array1<double> grads(grad_mags.size());

  for( i=0;i<aug_elems.size();i++) {
    grads[i] = S[(grad_mags[i]-min)*temp];
    grad_total += grads[i];
  }

  vol_wt /= vol_total;
  grad_wt /= grad_total;

  for(i=0;i<aug_elems.size();i++) {
    aug_elems[i].importance =
      vol_wt*aug_elems[i].importance + grads[i]*grad_wt;
  }

  cerr << vol_total << " Volume " << grad_total << " Gradient\n";
}

#define SCALARFIELD_VERSION 1

void ScalarField::io(Piostream& stream)
{
    /*int version=*/stream.begin_class("ScalarField", SCALARFIELD_VERSION);
    int* repp=(int*)&rep;
    stream.io(*repp);
    if(stream.reading()){
	have_bounds=0;
	have_minmax=0;
    }
    stream.end_class();
}

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.6  2000/02/04 00:19:32  yarden
// enable to store the grid part of a ScalarField in a seperate file.
// a flag (sererate_raw) signal if this ScalarField was read from a split
// input file or should be writen as two. raw_filename specify the secondary
// file name. if no filename is given during writing, the output routines
// will try to extract the name from the output stream and attach a '.raw'
// extension to the binary portion.
//
// replaced ScalarFieldRGxxx with ScalarFieldRGTYPE. it also replaces
// the ScalarFieldRG files (i.e. the old RG with no type specification
// which defaults to 'double'
//
// Revision 1.5  1999/10/07 02:07:32  sparker
// use standard iostreams and complex type
//
// Revision 1.4  1999/08/29 00:46:52  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.3  1999/08/25 03:48:35  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:48  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:23  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:10  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//
