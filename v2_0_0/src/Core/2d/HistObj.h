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
 *  HistObj.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_HistObj_h
#define SCI_HistObj_h 

#include <Core/Datatypes/Color.h>
#include <Core/Containers/Array1.h>
#include <Core/2d/Polyline.h>

namespace SCIRun {
  
class SCICORESHARE HistObj : public Polyline {
protected:
  Array1<double> ref_;
  int bins_;
  double ref_min_, ref_max_;

public:
  HistObj( const string &name="");
  HistObj( const Array1<double> &, const string &name="" );
  virtual ~HistObj();

  void set_bins( int );
  void set_data( const Array1<double> &);

  virtual double at( double );
  virtual void get_bounds( BBox2d &bb );

public:
  // For OpenGL
#ifdef SCI_OPENGL
  virtual void draw( bool = false );
#endif
  static PersistentTypeID type_id;

  virtual void io(Piostream&);    

private:
  void compute();


};

void Pio(Piostream&, HistObj*&);

} // namespace SCIRun

#endif // SCI_HistObj_h
