
/*
 *  BaseWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Base_Widget_h
#define SCI_project_Base_Widget_h 1

#include <PSECore/Constraints/BaseVariable.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/Pickable.h>
#include <SCICore/Geom/GeomPick.h>
#include <SCICore/Geom/Switch.h>
#include <SCICore/Geom/TCLGeom.h>
#include <SCICore/TclInterface/TCL.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace SCICore {
  namespace Thread {
    class CrowdMonitor;
  }
}

namespace PSECore {
  namespace Dataflow {
    class Module;
  }
  namespace Datatypes {
    class GeometryOPort;
  }
  namespace Constraints {
    class ConstraintSolver;
    class BaseVariable;
    class BaseConstraint;
  }
}
namespace PSECommon {
  namespace Modules {
    class Roe;
  }
}


namespace PSECore {
namespace Widgets {

using SCICore::TclInterface::TCL;
using SCICore::TclInterface::TCLArgs;
using SCICore::Thread::CrowdMonitor;
using SCICore::Containers::clString;
using SCICore::Containers::Array1;
using SCICore::GeomSpace::GeomSwitch;
using SCICore::GeomSpace::GeomPick;
using SCICore::GeomSpace::Pickable;
using SCICore::GeomSpace::GeomObj;
using SCICore::GeomSpace::GeomMaterial;
using SCICore::GeomSpace::BState;
using SCICore::GeomSpace::MaterialHandle;
using SCICore::GeomSpace::TCLMaterial;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

using PSECore::Datatypes::GeometryOPort;
using PSECore::Dataflow::Module;
using PSECore::Constraints::ConstraintSolver;
using PSECore::Constraints::BaseVariable;
using PSECore::Constraints::BaseConstraint;
using PSECore::Constraints::Index;
using PSECore::Constraints::Real;
using PSECommon::Modules::Roe;

class BaseWidget : public TCL, public Pickable {
public:
   BaseWidget( Module* module, CrowdMonitor* lock,
	       const clString& name,
	       const Index NumVariables,
	       const Index NumConstraints,
	       const Index NumGeometries,
	       const Index NumPicks,
	       const Index NumMaterials,
	       const Index NumModes,
	       const Index NumSwitches,
	       const Real widget_scale );
   BaseWidget( const BaseWidget& );
   virtual ~BaseWidget();

   void *userdata;	// set this after construction if you want to use it

   void SetScale( const Real scale );
   double GetScale() const;

   void SetEpsilon( const Real Epsilon );

   GeomSwitch* GetWidget();
   void Connect(GeometryOPort*);

   virtual void MoveDelta( const Vector& delta ) = 0;
   virtual void Move( const Point& p );  // Not pure virtual
   virtual Point ReferencePoint() const = 0;
   
   int GetState();
   void SetState( const int state );

   // This rotates through the "minimizations" or "gaudinesses" for the widget.
   // Doesn't have to be overloaded.
   virtual void NextMode();
   virtual void SetCurrentMode(const Index);
   Index GetMode() const;

   void SetMaterial( const Index mindex, const MaterialHandle& matl );
   MaterialHandle GetMaterial( const Index mindex ) const;

   void SetDefaultMaterial( const Index mindex, const MaterialHandle& matl );
   MaterialHandle GetDefaultMaterial( const Index mindex ) const;

   virtual clString GetMaterialName( const Index mindex ) const=0;
   virtual clString GetDefaultMaterialName( const Index mindex ) const;

   inline Point GetPointVar( const Index vindex ) const;
   inline Real GetRealVar( const Index vindex ) const;
   
   virtual void geom_pick(GeomPick*, Roe*, int, const BState& bs);
  //   virtual void geom_pick(GeomPick*, void*, int);
   virtual void geom_pick(GeomPick*, void*, GeomObj*);
   virtual void geom_pick(GeomPick*, void*);
   virtual void geom_release(GeomPick*, int, const BState& bs);
  //   virtual void geom_release(GeomPick*, void*, int);
   virtual void geom_release(GeomPick*, void*, GeomObj*);
   virtual void geom_release(GeomPick*, void*);

   virtual void geom_moved(GeomPick*, int, double, const Vector&, void*);
  //virtual void geom_moved(GeomPick*, int, double, const Vector&, void*, int);
   virtual void geom_moved(GeomPick*, int, double, const Vector&, void*, GeomObj*);
   virtual void geom_moved(GeomPick*, int, double, const Vector&, int, const BState& bs);
   virtual void geom_moved(GeomPick*, int, double, const Vector&, const BState&, int);

   BaseWidget& operator=( const BaseWidget& );
   int operator==( const BaseWidget& );

   void print( std::ostream& os ) const;

   void init_tcl();
   void tcl_command( TCLArgs&, void* );
   
   // Use this to pop up the widget ui.
   void ui() const;
   
protected:
   Module* module;
   CrowdMonitor* lock;
   clString name;
   clString id;

   ConstraintSolver* solve;
   
   void execute(int always_callback);
   virtual void redraw()=0;
   Index NumVariables;
   Index NumConstraints;
   Index NumGeometries;
   Index NumPicks;
   Index NumMaterials;

   Array1<BaseConstraint*> constraints;
   Array1<BaseVariable*> variables;
   Array1<GeomObj*> geometries;
   Array1<GeomPick*> picks;
   Array1<GeomMaterial*> materials;

   enum {Mode0,Mode1,Mode2,Mode3,Mode4,Mode5,Mode6,Mode7,Mode8,Mode9};
   Index NumModes;
   Index NumSwitches;
   Array1<long> modes;
   Array1<GeomSwitch*> mode_switches;
   Index CurrentMode;
   // modes contains the bitwise OR of Switch0-Switch8
   enum {
	Switch0 = 0x0001,
        Switch1 = 0x0002,
        Switch2 = 0x0004,
        Switch3 = 0x0008,
        Switch4 = 0x0010,
        Switch5 = 0x0020,
        Switch6 = 0x0040,
        Switch7 = 0x0080,
        Switch8 = 0x0100
  };

   GeomSwitch* widget;
   Real widget_scale;
   Real epsilon;

   Array1<GeometryOPort*> oports;
   void flushViews() const;
   
   // Individual widgets use this for tcl if necessary.
   // tcl command in args[1], params in args[2], args[3], ...
   virtual void widget_tcl( TCLArgs& );

   void CreateModeSwitch( const Index snum, GeomObj* o );
   void SetMode( const Index mode, const long swtchs );
   void FinishWidget();

   // Used to pass a material to .tcl file.
   TCLMaterial tclmat;

   // These affect ALL widgets!!!
   static MaterialHandle DefaultPointMaterial;
   static MaterialHandle DefaultEdgeMaterial;
   static MaterialHandle DefaultSliderMaterial;
   static MaterialHandle DefaultResizeMaterial;
   static MaterialHandle DefaultSpecialMaterial;
   static MaterialHandle DefaultHighlightMaterial;
};

std::ostream& operator<<( std::ostream& os, BaseWidget& w );

} // End namespace Widgets
} // End namespace PSECore

//
// $Log$
// Revision 1.6.2.2  2000/10/26 14:16:56  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.7  2000/08/11 15:44:44  bigler
// Changed geom_* functions that took an int index to take a GeomObj* picked_obj.
//
// Revision 1.6  1999/10/07 02:07:24  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/09/08 02:26:43  sparker
// Various #include cleanups
//
// Revision 1.4  1999/08/28 17:54:33  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/25 03:48:28  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:27  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:05  mcq
// Initial commit
//
// Revision 1.4  1999/05/13 18:27:45  dav
// Added geom_moved methods to BaseWidget
//
// Revision 1.3  1999/05/06 20:17:23  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:29  dav
// Import sources
//
//

#endif
