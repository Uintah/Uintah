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
#include <Core/GeomInterface/Pickable.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GuiGeom.h>
#include <Core/GuiInterface/GuiCallback.h>
#include <Core/GuiInterface/GuiVar.h>

#include <Dataflow/Widgets/share.h>

namespace SCIRun {

class CrowdMonitor;
class Module;
class GeometryOPort;
class ConstraintSolver;
class BaseVariable;
class BaseConstraint;

class ViewWindow;


class SCISHARE BaseWidget : public GuiCallback, public WidgetPickable {
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
	      const double widget_scale,
	      const bool stipple_occluded=false);
  BaseWidget( const BaseWidget& );
  virtual ~BaseWidget();

  void *userdata;	// set this after construction if you want to use it

  void SetScale( const double scale );
  double GetScale() const;

  void SetEpsilon( const double Epsilon );

  GeomHandle GetWidget();
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
  inline double GetRealVar( const Index vindex ) const;
   
  virtual void geom_pick(GeomPickHandle, ViewWindow*, int, const BState& bs);
  virtual void geom_release(GeomPickHandle, int, const BState& bs);
  virtual void geom_moved(GeomPickHandle, int, double, const Vector&, int,
			  const BState& button_state,
			  const Vector &pick_offset);

  //BaseWidget& operator=( const BaseWidget& );
  int operator==( const BaseWidget& );

  void print( std::ostream& os ) const;

  void init_tcl();
  void tcl_command( GuiArgs&, void* );
   
  // Use this to pop up the widget ui.
  void ui() const;

  virtual string GetStateString();
  virtual int SetStateString(const string &str);
  string name() { return name_; }

protected:
  Module*       module_;
  CrowdMonitor* lock_;
  string        name_;
  string        id_;

  ConstraintSolver* solve;
   
  void execute(int always_callback);
  virtual void redraw()=0;

  vector<BaseConstraint*> constraints;
  vector<BaseVariable*>   variables;
  vector<GeomHandle>      geometries;
  vector<GeomHandle>      picks_;
  vector<GeomMaterial*>   materials;

#ifdef sgi  
#pragma set woff 1424  // shut up sgi compiler.  
#endif
  template <class T> T geometry(int i) {
    ASSERT(geometries[i].get_rep());
    T tmp = dynamic_cast<T>(geometries[i].get_rep());
    ASSERT(tmp);
    return tmp;
  }

  GeomPickHandle picks(int i) {
    ASSERT(picks_[i].get_rep());
    GeomPick *p = dynamic_cast<GeomPick*>(picks_[i].get_rep());
    ASSERT(p);
    return p;
  }
#ifdef sgi  
#pragma reset woff 1424  // shut up sgi compiler.  
#endif

  enum {Mode0=0,Mode1,Mode2,Mode3,Mode4,Mode5,Mode6,Mode7,Mode8,Mode9};
  vector<long>        modes;
  vector<GeomSwitch*> mode_switches;

  Index current_mode_;

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

  GeomHandle widget_;
  double widget_scale_;
  double epsilon_;

  vector<GeometryOPort*> oports_;
  void flushViews() const;
   
  // Individual widgets use this for tcl if necessary.
  // tcl command in args[1], params in args[2], args[3], ...
  virtual void widget_tcl( GuiArgs& );

  void CreateModeSwitch( const Index snum, GeomHandle o );
  void SetMode( const Index mode, const long swtchs );
  void SetNumModes(int num) { modes.resize(num); }
  void FinishWidget();

  // Used to pass a material to .tcl file.
  GuiInterface* gui;
  GuiContext* ctx;
  GuiMaterial tclmat_;

  bool stipple_occluded_;

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
