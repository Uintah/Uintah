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
 *  Pickable.h: ???
 *
 *  Written by:
 *   Dav de St. Germain...
 *   Department of Computer Science
 *   University of Utah
 *   Mar 1999
 *
 *  Copyright (C) 1999 University of Utah
 */

#ifndef SCI_Geom_Pickable_h
#define SCI_Geom_Pickable_h 1

#include <Core/share/share.h>
#include <Core/Geom/GeomPick.h>

namespace SCIRun {

class Vector;
class Point;
class ViewWindow;
class GeomPick;

struct BState {
   unsigned int control:1;
   unsigned int alt:1;
   unsigned int shift:1;
   unsigned int btn:2;
};


class SCICORESHARE WidgetPickable {

public:
  virtual ~WidgetPickable();

  virtual void geom_pick(GeomPickHandle, ViewWindow*, int widget_data, 
			 const BState& bs);
  virtual void geom_release(GeomPickHandle, int, const BState& bs);
  virtual void geom_moved(GeomPickHandle, int, double, const Vector&, int,
			  const BState&, const Vector &pick_offset);
};


class SCICORESHARE ModulePickable {

public:
  virtual ~ModulePickable();

  virtual void geom_pick(GeomPickHandle, void*, GeomHandle);
  virtual void geom_release(GeomPickHandle, void*, GeomHandle);
  virtual void geom_moved(GeomPickHandle, int, double, const Vector&,
			  void*, GeomHandle);
};

} // End namespace SCIRun

#endif /* SCI_Geom_Pickable_h */
