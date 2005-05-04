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

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

namespace FieldConverters {

using namespace SCIRun;

class ScalarFieldRGBase;
template <class T> class ScalarFieldRGT;
typedef class ScalarFieldRGT<double> ScalarFieldRG;
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

    int separate_raw;
    string raw_filename;
    string filename;

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

    // separate raw files
    void set_raw(int v) { separate_raw = v; }
    int get_raw() { return separate_raw; }
    void set_raw_filename( string &f ) { raw_filename = f; separate_raw =1;}
    string &get_raw_filename() { return raw_filename; }

    string get_filename() { return filename; }
    void set_filename( string &f) { filename = f ; }


    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // end namespace FieldConverters


#endif /* SCI_project_ScalarField_h */
