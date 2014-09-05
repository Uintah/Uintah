
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

#include <SCICore/Geom/GeomContainer.h>
#include <SCICore/Geom/Pickable.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Geometry/Vector.h>

namespace PSECore {
  namespace Widgets {
    class BaseWidget;
  }
  namespace Comm {
    class MessageBase;
  }
  namespace Modules {
    class Roe;
  }
}

namespace SCICore {
namespace GeomSpace {

using PSECore::Widgets::BaseWidget;
using PSECore::Comm::MessageBase;

class SCICORESHARE GeomPick : public GeomContainer {
  Pickable* module;
  void* cbdata;
  //int pick_index;
  GeomObj* picked_obj;
  Array1<Vector> directions;
  Pickable* widget;
  int widget_data;
  int selected;
  int ignore;
  MaterialHandle highlight;
  
  GeomPick(const GeomPick&);
public:
  bool drawOnlyOnPick;
  GeomPick(GeomObj*, Pickable* module);
  GeomPick(GeomObj*, Pickable* module, Pickable*, int widget_data);
  GeomPick(GeomObj*, Pickable* module,
	   const Vector&);
  GeomPick(GeomObj*, Pickable* module,
	   const Vector&, const Vector&);
  GeomPick(GeomObj*, Pickable* module,
	   const Vector&, const Vector&, const Vector&);
  GeomPick(GeomObj*, Pickable* module, const Array1<Vector>&);
  virtual ~GeomPick();
  virtual GeomObj* clone();
  int nprincipal();
  Vector principal(int i);
  void set_principal(const Vector&);
  void set_principal(const Vector&, const Vector&);
  void set_principal(const Vector&, const Vector&, const Vector&);
  void set_highlight(const MaterialHandle& matl);
  void set_module_data(void*);
  void set_widget_data(int);
  
  void set_picked_obj(GeomObj *);
  void pick(Roe* roe, const BState& bs);
  void moved(int axis, double distance, const Vector& delta, const BState& bs);
  void release(const BState& bs);
  
  void ignore_until_release();
  
  // For OpenGL
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};
  
} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.4.2.2  2000/10/26 17:18:37  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.5  2000/08/11 15:49:06  bigler
// Removed the int index and replaced it with a GeomObj* picked_obj.
// The index can be accessed though picked_obj->getID(int or IntVector).
//
// Revision 1.4  1999/10/07 02:07:43  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/28 17:54:41  sparker
// Integrated new Thread library
//
// Revision 1.2  1999/08/17 06:39:10  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:42  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:06  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:59  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//

#endif /* SCI_Geom_Pick_h */
