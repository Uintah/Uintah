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
 *  ScalarFieldUG.h: Scalar Fields defined on an unstructured grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ScalarFieldUG_h
#define SCI_project_ScalarFieldUG_h 1

#include <FieldConverters/Core/Datatypes/ScalarField.h>
#include <FieldConverters/Core/Datatypes/Mesh.h>
#include <Core/Containers/Array1.h>

namespace FieldConverters {

using namespace SCIRun;

class FieldConvertersSHARE ScalarFieldUG : public ScalarField {
public:
    MeshHandle mesh;
    Array1<double> data;

    enum Type {
	NodalValues,
	ElementValues
    };
    Type typ;

    ScalarFieldUG(Type typ);
    ScalarFieldUG(const MeshHandle&, Type typ);
    virtual ~ScalarFieldUG();
    virtual ScalarField* clone();

    virtual void compute_bounds();
    virtual void compute_minmax();
    virtual Vector gradient(const Point&);
    virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6);
    virtual int interpolate(const Point&, double&, int& ix, double epsilon1=1.e-6, double epsilon2=1.e-6, int exhaustive=0);
    virtual void get_boundary_lines(Array1<Point>& lines);

    virtual void compute_samples(int);  // for random distributions in fields
    virtual void distribute_samples();

    // this has to be called before 2 functions below...

    virtual void fill_gradmags();

    // diferent ways to augment a mesh...
    // 1/grad, grad, histogram of gradients

    // this is just special cased because of potential missing elements
    // you might want to get rid of it and use the base class...

    virtual void over_grad_augment(double vol_wt, double grad_wt, 
				   double crit_scale);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace FieldConverters

#endif
