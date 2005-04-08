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
 *  TriStrip.h: Triangle Strip object
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_TriStrip_h
#define SCI_Geom_TriStrip_h 1

#include <Core/Geom/GeomVertexPrim.h>

namespace SCIRun {

class GeomTriStrip : public GeomVertexPrim {
public:
    GeomTriStrip();
    GeomTriStrip(const GeomTriStrip&);
    virtual ~GeomTriStrip();

    virtual GeomObj* clone();

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    int size(void);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class GeomTriStripList : public GeomObj {
    int n_strips;
    Array1<float> pts;
    Array1<float> nrmls;
    Array1<int>   strips;
public:
    GeomTriStripList();
    virtual ~GeomTriStripList();

    virtual GeomObj* clone();

    void add(const Point&);
    void add(const Point&, const Vector&);
    
    void end_strip(void); // ends a tri-strip

    Point get_pm1(void);
    Point get_pm2(void);

    void permute(int,int,int);
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void get_bounds(BBox&);

   int size(void);
   int num_since(void);

   virtual void io(Piostream&);
   static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif /* SCI_Geom_TriStrip_h */
