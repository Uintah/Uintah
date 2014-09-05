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
 *  BaseWidget.cc : ?
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifdef _WIN32
#pragma warning(disable:4355)
#endif

#include <Dataflow/Widgets/BaseWidget.h>
#include <Dataflow/Constraints/BaseConstraint.h>
#include <Dataflow/Constraints/ConstraintSolver.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/NotFinished.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
using std::ostream;

namespace SCIRun {


static const Index NumDefaultMaterials = 6;

MaterialHandle BaseWidget::DefaultPointMaterial(scinew Material(Color(0,0,0),
							     Color(.54,.60,1),
							     Color(.5,.5,.5),
							     20));
MaterialHandle BaseWidget::DefaultEdgeMaterial(scinew Material(Color(0,0,0),
							    Color(.54,.60,.66),
							    Color(.5,.5,.5),
							    20));
MaterialHandle BaseWidget::DefaultSliderMaterial(scinew Material(Color(0,0,0),
							      Color(.66,.60,.40),
							      Color(.5,.5,.5),
							      20));
MaterialHandle BaseWidget::DefaultResizeMaterial(scinew Material(Color(0,0,0),
							      Color(.54,1,.60),
							      Color(.5,.5,.5),
							      20));
MaterialHandle BaseWidget::DefaultSpecialMaterial(scinew Material(Color(0,0,0),
							       Color(1,.54,.60),
							       Color(.5,.5,.5),
							       20));
MaterialHandle BaseWidget::DefaultHighlightMaterial(scinew Material(Color(0,0,0),
								 Color(.8,0,0),
								 Color(.5,.5,.5),
								 20));

static clString make_id(const clString& name)
{
   static int next_widget_number=0;
   static Mutex idlock("Widget ID lock");
   idlock.lock();
   clString id ( name+"_"+to_string(next_widget_number++) );
   idlock.unlock();
   return id;
   
}

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */
BaseWidget::BaseWidget( Module* module, CrowdMonitor* lock,
			const clString& name,
			const Index NumVariables,
			const Index NumConstraints,
			const Index NumGeometries,
			const Index NumPicks,
			const Index NumMaterials,
			const Index NumModes,
			const Index NumSwitches,
			const Real widget_scale )
  : module(module),
    lock(lock),
    name(name),
    id(make_id(name)),
    solve(scinew ConstraintSolver), 
    NumVariables(NumVariables),
    NumConstraints(NumConstraints),
    NumGeometries(NumGeometries),
    NumPicks(NumPicks),
    NumMaterials(NumMaterials),
    constraints(NumConstraints),
    variables(NumVariables),
    geometries(NumGeometries),
    picks(NumPicks),
    materials(NumMaterials),
    NumModes(NumModes),
    NumSwitches(NumSwitches),
    modes(NumModes),
    mode_switches(NumSwitches),
    CurrentMode(0),
    widget_scale(widget_scale),
    epsilon(1e-3),
    tclmat("material", id, this)
{

   Index i;
   for (i=0; i<NumSwitches; i++)
      mode_switches[i] = NULL;
   for (i=0; i<NumConstraints; i++)
      constraints[i] = NULL;
   for (i=0; i<NumVariables; i++)
      variables[i] = NULL;
   for (i=0; i<NumGeometries; i++)
      geometries[i] = NULL;
   for (i=0; i<NumPicks; i++)
      picks[i] = NULL;
   for (i=0; i<NumMaterials; i++)
      materials[i] = NULL;
   for (i=0; i<NumModes; i++)
      modes[i] = -1;

   init_tcl();
}


/***************************************************************************
 * The destructor frees the widget's allocated structures.
 * The BaseWidget's destructor frees the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Therefore, most widgets' destructors will not need to do anything.
 */
BaseWidget::~BaseWidget()
{
   Index index;
   
   for (index = 0; index < NumVariables; index++) {
      delete variables[index];
   }
   
   for (index = 0; index < NumConstraints; index++) {
      delete constraints[index];
   }

   // Geometry, picks, materials, and switches are freed automatically.
   // Modes don't require freeing.
   // Schemes are freed as part of the variables and constraints.
}


void
BaseWidget::init_tcl()
{
   TCL::add_command(id+"-c", this, 0);
   TCL::execute(name+" "+id);
}


void
BaseWidget::ui() const
{
   TCL::execute(id+" ui");
}


void
pmat( const MaterialHandle& mat )
{
   cout << "Material{" << endl;
   cout << "  ambient(" << mat->ambient.r() << "," << mat->ambient.g() << "," << mat->ambient.b() << ")" << endl;
   cout << "  diffuse(" << mat->diffuse.r() << "," << mat->diffuse.g() << "," << mat->diffuse.b() << ")" << endl;
   cout << "  specular(" << mat->specular.r() << "," << mat->specular.g() << "," << mat->specular.b() << ")" << endl;
   cout << "  shininess(" << mat->shininess << ")" << endl;
   cout << "  emission(" << mat->emission.r() << "," << mat->emission.g() << "," << mat->emission.b() << ")" << endl;
   cout << "  reflectivity(" << mat->reflectivity << ")" << endl;
   cout << "  transparency(" << mat->transparency << ")" << endl;
   cout << "  refraction_index(" << mat->refraction_index << ")" << endl;
   cout << "}" << endl;
}


void
BaseWidget::tcl_command(TCLArgs& args, void*)
{
   if(args.count() < 2){
      args.error("widget needs a minor command");
      return;
   }
   int mati;
   
   if (args[1] == "nextmode") {
      if (args.count() != 2) {
	 args.error("widget doesn't need a minor command");
	 return;
      }
      NextMode();
   } else if (args[1] == "print") { 
       print(cout);
   } else if (args[1] == "defmaterials") {
      if (args.count() != 2) {
	 args.error("widget doesn't need a minor command");
	 return;
      }
      Array1<clString> defmateriallist(NumDefaultMaterials);
      
      for(Index i=0;i<NumDefaultMaterials;i++){
	 defmateriallist[i]=GetDefaultMaterialName(i);
      }   
      args.result(args.make_list(defmateriallist));
   } else if (args[1] == "materials") {
      if (args.count() != 2) {
	 args.error("widget doesn't nedd a minor command");
	 return;
      }
      Array1<clString> materiallist(NumMaterials);
      
      for(Index i=0;i<NumMaterials;i++){
	 materiallist[i]=GetMaterialName(i);
      }   
      args.result(args.make_list(materiallist));
   } else if (args[1] == "getdefmat") {
      if (args.count() != 3) {
	 args.error("widget needs default material index");
	 return;
      }
      if (!args[2].get_int(mati)){
	 args.error("widget can't parse default material index `"+args[2]+"'");
	 return;
      }
      if ((mati < 0) || ((unsigned int)mati >= NumDefaultMaterials)) {
	 args.error("widget default material index out of range `"+args[2]+"'");
	 return;
      }
      pmat(GetDefaultMaterial(mati));
      tclmat.set(*(GetDefaultMaterial(mati).get_rep()));
   } else if (args[1] == "getmat") {
      if (args.count() != 3) {
	 args.error("widget needs material index");
 	 return;
      }
      if (!args[2].get_int(mati)) {
	 args.error("widget can't parse material index `"+args[2]+"'");
	 return;
      }
      if ((mati < 0) || ((unsigned int)mati >= NumMaterials)) {
	 args.error("widget material index out of range `"+args[2]+"'");
	 return;
      }
      tclmat.set(*(GetMaterial(mati).get_rep()));
   } else if(args[1] == "setdefmat"){
      if (args.count() != 3) {
	 args.error("widget needs material index");
	 return;
      }
      if (!args[2].get_int(mati)) {
	 args.error("widget can't parse material index `"+args[2]+"'");
	 return;
      }
      if ((mati <0) || ((unsigned int)mati >= NumDefaultMaterials)) {
	 args.error("widget material index out of range `"+args[2]+"'");
	 return;
      }
      reset_vars();
      MaterialHandle mat(scinew Material(tclmat.get()));
      pmat(mat);
      SetDefaultMaterial(mati, mat);
   } else if(args[1] == "setmat"){
      if (args.count() != 3) {
	 args.error("widget needs material index");
	 return;
      }
      if (!args[2].get_int(mati)) {
	 args.error("widget can't parse material index `"+args[2]+"'");
	 return;
      }
      if ((mati < 0) || ((unsigned int)mati >= NumMaterials)) {
	 args.error("widget material index out of range `"+args[2]+"'");
	 return;
      }
      reset_vars();
      MaterialHandle mat(scinew Material(tclmat.get()));
      pmat(mat);
      SetMaterial(mati, mat);
   } else if(args[1] == "scale"){
      if (args.count() != 3) {
	 args.error("widget needs user scale");
	 return;
      }
      Real us;
      if (!args[2].get_double(us)) {
	 args.error("widget can't parse user scale `"+args[2]+"'");
	 return;
      }
      SetScale(GetScale()*us);
   } else if(args[1] == "dialdone"){
      module->widget_moved(1);
   } else {
      widget_tcl(args);
   }

   reset_vars();
}


void
BaseWidget::widget_tcl( TCLArgs& args )
{
   args.error("widget unknown tcl command `"+args[1]+"'");
}


void
BaseWidget::SetScale( const double scale )
{
   widget_scale = scale;
   solve->SetEpsilon(epsilon*widget_scale);
   // TCL::execute(id+" scale_changed "+to_string(widget_scale));
   execute(0);
}


double
BaseWidget::GetScale() const
{
   return widget_scale;
}


void
BaseWidget::SetEpsilon( const Real Epsilon )
{
   epsilon = Epsilon;
   solve->SetEpsilon(epsilon*widget_scale);
}


GeomSwitch*
BaseWidget::GetWidget()
{
   return widget;
}


void
BaseWidget::Connect(GeometryOPort* oport)
{
   oports.add(oport);
}


void
BaseWidget::Move( const Point& p )
{
   MoveDelta(p - ReferencePoint());
}


int
BaseWidget::GetState()
{
   return widget->get_state();
}


void
BaseWidget::SetState( const int state )
{
   widget->set_state(state);
}


void
BaseWidget::NextMode()
{
   CurrentMode = (CurrentMode+1) % NumModes;
   for (Index s=0; s<NumSwitches; s++)
      if (modes[CurrentMode]&(1<<s))
	 mode_switches[s]->set_state(1);
      else
	 mode_switches[s]->set_state(0);

   execute(0);
}

void
BaseWidget::SetCurrentMode(const Index mode)
{
   CurrentMode = mode % NumModes;
   for (Index s=0; s<NumSwitches; s++)
      if (modes[CurrentMode]&(1<<s))
	 mode_switches[s]->set_state(1);
      else
	 mode_switches[s]->set_state(0);

   execute(0);
}

Index
BaseWidget::GetMode() const
{
   return CurrentMode;
}


void
BaseWidget::SetMaterial( const Index mindex, const MaterialHandle& matl )
{
   ASSERT(mindex<NumMaterials);
   materials[mindex]->setMaterial(matl);
   flushViews();
}


MaterialHandle
BaseWidget::GetMaterial( const Index mindex ) const
{
   ASSERT(mindex<NumMaterials);
   return materials[mindex]->getMaterial();
}


void
BaseWidget::SetDefaultMaterial( const Index mindex, const MaterialHandle& matl )
{
   ASSERT(mindex<NumDefaultMaterials);
   switch(mindex){
   case 0:
      *DefaultPointMaterial.get_rep() = *matl.get_rep();
      break;
   case 1:
      *DefaultEdgeMaterial.get_rep() = *matl.get_rep();
      break;
   case 2:
      *DefaultSliderMaterial.get_rep() = *matl.get_rep();
      break;
   case 3:
      *DefaultResizeMaterial.get_rep() = *matl.get_rep();
      break;
   case 4:
      *DefaultSpecialMaterial.get_rep() = *matl.get_rep();
      break;
   default:
      *DefaultHighlightMaterial.get_rep() = *matl.get_rep();
      break;
   }
   flushViews();
}


MaterialHandle
BaseWidget::GetDefaultMaterial( const Index mindex ) const
{
   ASSERT(mindex<NumDefaultMaterials);
   switch(mindex){
   case 0:
      return DefaultPointMaterial;
   case 1:
      return DefaultEdgeMaterial;
   case 2:
      return DefaultSliderMaterial;
   case 3:
      return DefaultResizeMaterial;
   case 4:
      return DefaultSpecialMaterial;
   default:
      return DefaultHighlightMaterial;
   }
}


clString
BaseWidget::GetDefaultMaterialName( const Index mindex ) const
{
   ASSERT(mindex<NumDefaultMaterials);
   
   switch(mindex){
   case 0:
      return "Point";
   case 1:
      return "Edge";
   case 2:
      return "Slider";
   case 3:
      return "Resize";
   case 4:
      return "Special";
   default:
      return "Highlight";
   }
}


Point
BaseWidget::GetPointVar( const Index vindex ) const
{
   ASSERT(vindex<NumVariables);

   return variables[vindex]->point();
}


Real
BaseWidget::GetRealVar( const Index vindex ) const
{
   ASSERT(vindex<NumVariables);

   return variables[vindex]->real();
}


void
BaseWidget::flushViews() const
{
   for(int i=0;i<oports.size();i++)
      oports[i]->flushViews();
}


void
BaseWidget::execute(int always_callback)
{
   if (always_callback || solve->VariablesChanged()) {
      module->widget_moved(0);
      solve->ResetChanged();
   }

   lock->writeLock();
   redraw();
   lock->writeUnlock();

   flushViews();
}

void
//BaseWidget::geom_pick(GeomPick*, void*, int)
BaseWidget::geom_pick(GeomPick*, void*, GeomObj*)
{
  NOT_FINISHED("Module::geom_pick: This version of geom_pick is only here to stop the compiler from complaining, it should never be used.");
}
void
BaseWidget::geom_pick(GeomPick*, void*)
{
  NOT_FINISHED("Module::geom_pick: This version of geom_pick is only here to stop the compiler from complaining, it should never be used.");
}

void
BaseWidget::geom_pick( GeomPick* pick, ViewWindow* /*roe*/, int /* cbdata */, const BState& state )
{
   cerr << "btn=" << state.btn << endl;
   cerr << "alt=" << state.alt << endl;
   cerr << "ctl=" << state.control << endl;
   if (state.btn == 3 && !state.alt && !state.control) {
      ui();
      pick->ignore_until_release();
   } else if (state.btn == 1 && !state.alt && state.control) {
      BBox bbox;
      widget->get_bounds(bbox);
//SGP      viewwindow->autoview(bbox);
      pick->ignore_until_release();
   } else if (state.btn == 2 && !state.alt && !state.control) {
      NextMode();
      pick->ignore_until_release();
   }
}


void
//BaseWidget::geom_release(GeomPick*, void*, int)
BaseWidget::geom_release(GeomPick*, void*, GeomObj*)
{
  NOT_FINISHED("Module::geom_release: This version of geom_release is only here to stop the compiler from complaining, it should never be used.");
}

void
BaseWidget::geom_release(GeomPick*, void*)
{
  NOT_FINISHED("Module::geom_release: This version of geom_release is only here to stop the compiler from complaining, it should never be used.");
}

void
BaseWidget::geom_release( GeomPick*, int /* cbdata */, const BState& )
{
    module->widget_moved(1);
}

void
BaseWidget::geom_moved(GeomPick*, int, double, const Vector&,
		       void*)
{
  NOT_FINISHED("Module::geom_release: This version of geom_release is only here to stop the compiler from complaining, it should never be used.");
}

void
BaseWidget::geom_moved(GeomPick*, int, double, const Vector&,
		       int, const BState& )
{
  NOT_FINISHED("Module::geom_release: This version of geom_release is only here to stop the compiler from complaining, it should never be used.");
}

void
BaseWidget::geom_moved(GeomPick*, int, double, const Vector&,
		       //void*, int)
		       void*, GeomObj*)
{
  NOT_FINISHED("Module::geom_release: This version of geom_release is only here to stop the compiler from complaining, it should never be used.");
}
void
BaseWidget::geom_moved(GeomPick*, int, double, const Vector&, 
		       const BState&, int)
{
  NOT_FINISHED("Module::geom_release: This version of geom_release is only here to stop the compiler from complaining, it should never be used.");
}

void
BaseWidget::CreateModeSwitch( const Index snum, GeomObj* o )
{
   ASSERT(snum<NumSwitches);
   ASSERT(mode_switches[snum]==NULL);
   mode_switches[snum] = scinew GeomSwitch(o);
}


void
BaseWidget::SetMode( const Index mode, const long swtchs )
{
   ASSERT(mode<NumModes);
   modes[mode] = swtchs;
}


/***************************************************************************
 * This performs consistency checking to ensure full initialization.
 * It should be called as the last line of each widget's constructor.
 */
void
BaseWidget::FinishWidget()
{
   Index i;
   for (i=0; i<NumModes; i++)
      if (modes[i] == -1) {
	 cerr << "BaseWidget Error:  Mode " << i << " is unitialized!" << endl;
	 exit(-1);
      }
   for (i=0; i<NumSwitches; i++)
      if (mode_switches[i] == NULL) {
	 cerr << "BaseWidget Error:  Switch " << i << " is unitialized!" << endl;
	 exit(-1);
      }
   for (i=0; i<NumConstraints; i++)
      if (constraints[i] == NULL) {
	 cerr << "BaseWidget Error:  Constraint " << i << " is unitialized!" << endl;
	 exit(-1);
      }
   for (i=0; i<NumVariables; i++)
      if (variables[i] == NULL) {
	 cerr << "BaseWidget Error:  Variable " << i << " is unitialized!" << endl;
	 exit(-1);
      }
   for (i=0; i<NumGeometries; i++)
      if (geometries[i] == NULL) {
	 cerr << "BaseWidget Error:  Geometry " << i << " is unitialized!" << endl;
	 exit(-1);
      }
   for (i=0; i<NumPicks; i++)
      if (picks[i] == NULL) {
	 cerr << "BaseWidget Error:  Pick " << i << " is unitialized!" << endl;
	 exit(-1);
      }
   for (i=0; i<NumMaterials; i++)
      if (materials[i] == NULL) {
	 cerr << "BaseWidget Error:  Material " << i << " is unitialized!" << endl;
	 exit(-1);
      }
   
   GeomGroup* sg = scinew GeomGroup;
   for (i=0; i<NumSwitches; i++) {
      if (modes[CurrentMode]&(1<<i))
	 mode_switches[i]->set_state(1);
      else
	 mode_switches[i]->set_state(0);
      sg->add(mode_switches[i]);
   }
   widget = scinew GeomSwitch(sg);

   // Init variables.
   for (Index vindex=0; vindex<NumVariables; vindex++)
      variables[vindex]->Order();
}


void
BaseWidget::print( ostream& os ) const
{
   Index index;
   
   for (index=0; index< NumVariables; index++) {
      os << *(variables[index]) << endl;
   }
   os << endl;
   
   for (index=0; index< NumConstraints; index++) {
      os << *(constraints[index]) << endl;
   }
   os << endl;
}


ostream&
operator<<( ostream& os, BaseWidget& w )
{
   w.print(os);
   return os;
}


BaseWidget& BaseWidget::operator=( const BaseWidget& )
{
    NOT_FINISHED("BaseWidget::operator=");
    return *this;
}

} // End namespace SCIRun
