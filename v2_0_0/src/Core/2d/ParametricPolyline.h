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

class SCICORESHARE ParametricPolyline : public DrawObj {

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
