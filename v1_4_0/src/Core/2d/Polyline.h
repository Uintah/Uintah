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
 *  Polyline.h: Displayable 2D object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_Polyline_h
#define SCI_Polyline_h 

#include <Core/2d/DrawObj.h>

namespace SCIRun {

    using std::vector;
  
class SCICORESHARE Polyline : public DrawObj {
protected:
  vector<double> data_;
  double min_, max_;

public:
  Polyline( const string &name="") : DrawObj(name) {} 
  Polyline( int i );
  Polyline( const vector<double> &, const string &name="" );
  virtual ~Polyline();

  virtual double at( double );
  
  void compute_minmax();
  void add( double );
  void clear() { data_.clear(); }

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

void Pio(Piostream&, Polyline*&);

} // namespace SCIRun

#endif // SCI_Polyline_h
