
/*
 *  ScalarField.h: The Scalar Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ScalarField_h
#define SCI_project_ScalarField_h 1

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::Containers::Array1;
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class ScalarFieldRGBase;
class ScalarFieldRG;
class ScalarFieldUG;
class ScalarFieldHUG;
class ScalarField;
typedef LockingHandle<ScalarField> ScalarFieldHandle;

// this augmented information is for doing random distributions
// simplifications and things like that

// augmented info for elements
// array of indices for point samples, etc.

// the simplest defenition if importance is just
// the percentage of the volume of this element

struct AugElement {  // on a per element basis
  Array1<int>   pt_samples; // indeces into sample array
  double        importance; // relative importance
};  

struct SampInfo {    // these are just the raw samples
  Point    loc;      // only thing set with random distributions...
  double   orig_val;
  double   cur_val;
  double   weight;
  // a vector could be here as well???
};


class SCICORESHARE ScalarField : public Datatype {
protected:
    int have_bounds;
    Point bmin;
    Point bmax;
    Vector diagonal;
    virtual void compute_bounds()=0;

    int have_minmax;
    double data_min;
    double data_max;
    virtual void compute_minmax()=0;
protected:
    enum Representation {
	RegularGridBase,
	UnstructuredGrid,
	RegularGrid,
	HexGrid,
	Zones,
	HP
    };
    ScalarField(Representation);
private:
    Representation rep;
public:
    virtual ~ScalarField();
    virtual ScalarField* clone()=0;

    ScalarFieldRG* getRG();
    ScalarFieldRGBase* getRGBase();
    ScalarFieldUG* getUG();
    ScalarFieldHUG* getHUG();
    void get_minmax(double&, double&);
    void get_bounds(Point&, Point&);
    void set_minmax(double, double);
//    void copy_bounds(ScalarField* c);

    void set_bounds(const Point&, const Point&);

    double longest_dimension();
    virtual Vector gradient(const Point&)=0;
    virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6)=0;
    virtual int interpolate(const Point&, double&, int& ix, double epsilon1=1.e-6, double epsilon2=1.e-6, int exhaustive=0)=0;
    virtual void get_boundary_lines(Array1<Point>& lines)=0;

    // exposed for now - maybe this should be abstraced at a later
    // time...

    Array1<AugElement> aug_elems;  // augmented information...

    Array1<SampInfo>    samples;   // samples...

    Array1<double>      grad_mags; // per-element gradient magnitudes
    double              total_gradmag;  // doesn't change...
    

    // fill in the samples

    virtual void compute_samples(int nsamp);  // make sure you define this...

    virtual void distribute_samples();        // weights have been modified
    
    // this has to be called before 2 functions below...

    virtual void fill_gradmags();

    // diferent ways to augment a mesh...
    // 1/grad, grad, histogram of gradients

    virtual void over_grad_augment(double vol_wt, double grad_wt, 
				   double crit_scale);

    virtual void grad_augment(double vol_wt, double grad_wt);

    virtual void hist_grad_augment(double vol_wt, double grad_wt,
				   const int HSIZE=4096); //size of histogram

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/08/29 00:46:52  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.3  1999/08/25 03:48:36  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:48  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:24  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:49  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:41  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:10  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif /* SCI_project_ScalarField_h */
