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
 *  SpotLight.h:  A Spot light source
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_SpotLight_h
#define SCI_Geom_SpotLight_h 1

#include <Core/Geom/Light.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {

class SpotLight : public Light {
    Point p;
    Vector v;
    float cutoff; // must be [0-90], or 180
    Color c;
public:
    SpotLight(const string& name, const Point&, const Vector&, 
	      float, const Color&, bool on = true, bool transformed = true);
    virtual ~SpotLight();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
#ifdef SCI_OPENGL
    virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx);
#endif
  void move( const Point& newP) { p = newP; }
  void setDirection( const Vector& newV) {v = newV; }
  void setCutoff( float co ) { 
    if ( co >=0 && co <=90 ) 
      cutoff = co;
    else
      cutoff = 180;
  }
  void setColor( const Color& newC) { c = newC; }
};

} // End namespace SCIRun


#endif /* SCI_Geom_SpotLight_h */
