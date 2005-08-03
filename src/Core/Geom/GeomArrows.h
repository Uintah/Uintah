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
 *  GeomArrows.h: Arrows objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Arrows_h
#define SCI_Geom_Arrows_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {


class GeomArrows : public GeomObj {
    double headwidth;
    double headlength;
    Array1<MaterialHandle> shaft_matls;
    Array1<MaterialHandle> back_matls;
    Array1<MaterialHandle> head_matls;
    Array1<Point> positions;
    Array1<Vector> directions;
    Array1<Vector> v1, v2;
    double rad; // radius of arrow shaft if cylinders are drawn
    int drawcylinders; // switch to use lines or cylinders for the arrow
    // The size of the of the head is proportional to the length of the vector.
    // When this flag is set the same size head is used for all the arrows 
    int normalize_headsize;
public:
    GeomArrows(double headwidth, double headlength=0.7, int cyl=0, double r=0,
	       int normhead = 0);
    GeomArrows(const GeomArrows&);
    virtual ~GeomArrows();

    virtual GeomObj* clone();

    void set_material(const MaterialHandle& shaft_matl,
                      const MaterialHandle& back_matl,
                      const MaterialHandle& head_matl);
    void add(const Point& pos, const Vector& dir);
    void add(const Point& pos, const Vector& dir,
	     const MaterialHandle& shaft, const MaterialHandle& back,
	     const MaterialHandle& head);
    inline int size() { return positions.size(); }

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void get_bounds(BBox&);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun

// $Log

#endif /* SCI_Geom_Arrows_h */
