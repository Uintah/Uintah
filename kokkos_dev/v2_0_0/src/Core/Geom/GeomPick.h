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
 *  GeomPick.h: Picking information for Geometry objects
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Pick_h
#define SCI_Geom_Pick_h 1

#include <Core/Geom/GeomContainer.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Vector.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
using std::vector;


class BaseWidget;
class MessageBase;
class ViewWindow;
class ModulePickable;
class WidgetPickable;
class BState;

class SCICORESHARE GeomPick : public GeomContainer {
private:
  ModulePickable*   module_;
  void*             cbdata_;
  GeomHandle        picked_obj_;
  vector<Vector>    directions_;
  WidgetPickable*   widget_;
  int               widget_data_;
  bool              selected_;
  bool              ignore_;
  MaterialHandle    highlight_;
  bool              draw_only_on_pick_;
  
  GeomPick(const GeomPick&);

public:
  GeomPick(GeomHandle, ModulePickable* module);
  GeomPick(GeomHandle, ModulePickable* module,
	   WidgetPickable*, int widget_data);
  GeomPick(GeomHandle, ModulePickable* module, const Vector&);
  GeomPick(GeomHandle, ModulePickable* module, const Vector&, const Vector&);
  GeomPick(GeomHandle, ModulePickable* module,
	   const Vector&, const Vector&, const Vector&);
  GeomPick(GeomHandle, ModulePickable* module, const Array1<Vector>&);
  virtual GeomObj* clone();

  int nprincipal();
  const Vector &principal(int i);
  void set_principal(const Vector&);
  void set_principal(const Vector&, const Vector&);
  void set_principal(const Vector&, const Vector&, const Vector&);
  void set_highlight(const MaterialHandle& matl);
  void set_module_data(void*);
  void set_widget_data(int);
  
  void set_picked_obj(GeomHandle);
  void pick(ViewWindow* viewwindow, const BState& bs);
  void moved(int axis, double distance, const Vector& delta, const BState& bs,
	     const Vector &pick_offset);
  void release(const BState& bs);
  
  void ignore_until_release();
  
  // For OpenGL
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

typedef LockingHandle<GeomPick> GeomPickHandle;
  
} // End namespace SCIRun


#endif /* SCI_Geom_Pick_h */
