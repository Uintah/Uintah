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

#include <Dataflow/share/share.h>

#include <Dataflow/Ports/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

class ScalarFieldUG;
class ScalarField;
typedef LockingHandle<ScalarField> ScalarFieldHandle;

class PSECORESHARE ScalarField : public Datatype {
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
	RegularGrid,
	UnstructuredGrid,
    };
    ScalarField(Representation);
private:
    Representation rep;
public:
    virtual ~ScalarField();
    virtual ScalarField* clone()=0;

    ScalarFieldUG* getUG();
    void get_minmax(double&, double&);
    void get_bounds(Point&, Point&);
    double longest_dimension();
    virtual Vector gradient(const Point&)=0;
    virtual int interpolate(const Point&, double&)=0;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_project_ScalarField_h */
