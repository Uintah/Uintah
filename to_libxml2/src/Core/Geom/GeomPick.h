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

class GeomPick : public GeomContainer {
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
