/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  ContourSet.h: The ContourSet Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Packages_DaveW_Datatypes_ContourSet_h
#define SCI_Packages_DaveW_Datatypes_ContourSet_h 1

#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace DaveW {
using namespace SCIRun;


class ContourSet;
typedef LockingHandle<ContourSet> ContourSetHandle;

class ContourSet : public Datatype {
public:
    Array1<Array1<Array1<Point> > > contours;
    Array1<Array1<double> > conductivity;
    Array1<Array1<Array1<Point> > > levels;
    Array1<Array1<int> > level_map;
    int bdry_type;
    Vector basis[3];
    Vector origin;
    BBox bbox;
    double space;
    Array1<clString> name;
    Array2<int> split_join; //split_join[line#][param]
    Array1<int> matl;	// what's the matl idx for each named contour

    ContourSet();
    ContourSet(const ContourSet &copy);
    virtual ~ContourSet();
    virtual ContourSet* clone();
    void translate(const Vector &v);
    void scale(double sc);
    void rotate(const Vector &rot);
    void build_bbox();
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};
} // End namespace DaveW



#endif /* SCI_Packages/DaveW_Datatypes_ContourSet_h */
