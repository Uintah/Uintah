
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
 *  Axes.h: Axes for 2D graphs
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   August 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCIRun_Core_2d_Axes_h 
#define SCIRun_Core_2d_Axes_h

#include <Core/Geom/Color.h>
#include <Core/Containers/Array1.h>
#include <Core/2d/DrawObj.h>
#include <Core/share/share.h>
#include <Core/2d/glprintf.h>
 
namespace SCIRun {

class SCICORESHARE Axes : public DrawObj {
private:

  int num_v_tics;
  int num_h_tics;
  Color color; 
  bool initialized;
  
public:

  Axes(const string &name="") : DrawObj(name), initialized(false) {}
  Axes(int h_tics, int v_tics, const string &name="") : DrawObj(name),
    initialized(false)
  { num_v_tics = v_tics; num_h_tics = h_tics; }
  
  virtual ~Axes();
  void set_color(const Color&);
  virtual void get_bounds(BBox2d&);
  
#ifdef SCI_OPENGL
  virtual void draw( bool = false );
#endif

  static PersistentTypeID type_id;
  virtual void io(Piostream&);

};

  void Pio(Piostream&, Axes*&);

} // namespace SCIRun

#endif
