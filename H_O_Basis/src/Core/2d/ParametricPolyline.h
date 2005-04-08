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
 *  ParametricPolyline.h: Displayable 2D object
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_ParametricPolyline_h
#define SCI_ParametricPolyline_h 

#include <Core/Datatypes/Color.h>
#include <Core/2d/DrawObj.h>

#include <map>

namespace SCIRun {
  
using std::map;
using std::pair;

class ParametricPolyline : public DrawObj {

  typedef map<double, pair<double, double> >::iterator iter;

protected:
  map< double, pair<double,double> > data_;
  double tmin_, tmax_, xmin_, xmax_, ymin_, ymax_;
  Color color_;

public:
  ParametricPolyline( const string &name="") : DrawObj(name) {} 
  ParametricPolyline( int i );
  ParametricPolyline( const map<double, pair<double,double> >&, 
		      const string &name="" );
  virtual ~ParametricPolyline();

  virtual bool at( double, pair<double,double>& );
  
  void compute_minmax();
  void add( double, double, double);
  void clear() { data_.clear(); }
  string tcl_color();

  void set_color( const Color &);
  Color get_color() { return color_; }

  virtual void get_bounds(BBox2d&);

  virtual void add(const vector<double>&);

  // For OpenGL
#ifdef SCI_OPENGL
  virtual void draw( bool = false );
#endif
  static PersistentTypeID type_id;

  virtual void io(Piostream&);    

};

void Pio(Piostream&, ParametricPolyline*&);

} // namespace SCIRun

#endif // SCI_ParametricPolyline_h
