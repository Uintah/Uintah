
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

#include <Datatypes/Datatype.h>
#include <Classlib/Array1.h>
#include <Classlib/LockingHandle.h>
#include <Geometry/Vector.h>
#include <Geometry/Point.h>

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


class ScalarField : public Datatype {
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
    double longest_dimension();
    virtual Vector gradient(const Point&)=0;
    virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6)=0;
    virtual int interpolate(const Point&, double&, int& ix, double epsilon1=1.e-6, double epsilon2=1.e-6)=0;
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

#endif /* SCI_project_ScalarField_h */
