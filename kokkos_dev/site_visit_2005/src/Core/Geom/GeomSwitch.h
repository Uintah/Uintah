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
 *  Switch.h:  Turn Geometry on and off
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Switch_h
#define SCI_Geom_Switch_h 1

#include <Core/Geom/GeomContainer.h>

namespace SCIRun {

class GeomSwitch : public GeomContainer {
    int state;
    GeomSwitch(const GeomSwitch&);

public:
    GeomSwitch(GeomHandle, int state=1);
    virtual GeomObj* clone();

    void set_state(int st);
    int get_state();
    virtual void get_bounds(BBox&);

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


class GeomTimeSwitch : public GeomContainer {
    double tbeg;
    double tend;
    GeomTimeSwitch(const GeomTimeSwitch&);

public:
    GeomTimeSwitch(GeomHandle, double tbeg, double tend);
    virtual GeomObj* clone();

    // For OpenGL
#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};


} // End namespace SCIRun


#endif /* SCI_Geom_Switch_h */
