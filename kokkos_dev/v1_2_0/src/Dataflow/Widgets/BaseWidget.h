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

#include <Dataflow/Constraints/BaseVariable.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/Pickable.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/Switch.h>
#include <Core/Geom/GuiGeom.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/GuiInterface/GuiVar.h>

namespace SCIRun {
  class CrowdMonitor;
}

namespace SCIRun {
  class Module;
  class GeometryOPort;
  class ConstraintSolver;
  class BaseVariable;
  class BaseConstraint;
}
namespace SCIRun {
  class ViewWindow;
}


namespace SCIRun {



class BaseWidget : public TCL, public Pickable {
public:
   BaseWidget( Module* module, CrowdMonitor* lock,
	       const string& name,
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

   virtual string GetMaterialName( const Index mindex ) const=0;
   virtual string GetDefaultMaterialName( const Index mindex ) const;

   inline Point GetPointVar( const Index vindex ) const;
   inline Real GetRealVar( const Index vindex ) const;
   
   virtual void geom_pick(GeomPick*, ViewWindow*, int, const BState& bs);
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
   string name;
   string id;

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
   GuiMaterial tclmat;

   // These affect ALL widgets!!!
   static MaterialHandle DefaultPointMaterial;
   static MaterialHandle DefaultEdgeMaterial;
   static MaterialHandle DefaultSliderMaterial;
   static MaterialHandle DefaultResizeMaterial;
   static MaterialHandle DefaultSpecialMaterial;
   static MaterialHandle DefaultHighlightMaterial;
};

std::ostream& operator<<( std::ostream& os, BaseWidget& w );

} // End namespace SCIRun

#endif
