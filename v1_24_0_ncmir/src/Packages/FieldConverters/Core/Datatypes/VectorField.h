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
 *  VectorField.h: The Vector Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_FieldConverters_VectorField_h
#define SCI_FieldConverters_VectorField_h 1

#include <FieldConverters/share/share.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

namespace FieldConverters {

using namespace SCIRun;

class VectorFieldRG;
class VectorFieldUG;
class VectorField;
typedef LockingHandle<VectorField> VectorFieldHandle;

class FieldConvertersSHARE VectorField : public Datatype {
protected:
    int have_bounds;
    Point bmin;
    Point bmax;
    Vector diagonal;
    virtual void compute_bounds()=0;

protected:
    enum Representation {
	RegularGrid,
	UnstructuredGrid
    };
    VectorField(Representation);
private:
    Representation rep;
public:
    virtual ~VectorField();
    virtual VectorField* clone()=0;

    VectorFieldRG* getRG();
    VectorFieldUG* getUG();
    void get_bounds(Point&, Point&);
    double longest_dimension();
    virtual int interpolate(const Point&, Vector&)=0;
    virtual int interpolate(const Point&, Vector&, int& cache, int exhaustive=0)=0;
    virtual void get_boundary_lines(Array1<Point>& lines)=0;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace FieldConverters


#endif /* SCI_FieldConverters_VectorField_h */
