
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


#include <Widgets/BaseWidget.h>

MaterialHandle BaseWidget::DefaultPointMaterial(new Material(Color(0,0,0),
							     Color(.54,.60,1),
							     Color(.5,.5,.5),
							     20));
MaterialHandle BaseWidget::DefaultEdgeMaterial(new Material(Color(0,0,0),
							    Color(.54,.60,.66),
							    Color(.5,.5,.5),
							    20));
MaterialHandle BaseWidget::DefaultSliderMaterial(new Material(Color(0,0,0),
							      Color(.66,.60,.40),
							      Color(.5,.5,.5),
							      20));
MaterialHandle BaseWidget::DefaultResizeMaterial(new Material(Color(0,0,0),
							      Color(.54,1,.60),
							      Color(.5,.5,.5),
							      20));
MaterialHandle BaseWidget::DefaultSpecialMaterial(new Material(Color(0,0,0),
							       Color(1,.54,.60),
							       Color(.5,.5,.5),
							       20));
MaterialHandle BaseWidget::DefaultHighlightMaterial(new Material(Color(0,0,0),
								 Color(.8,0,0),
								 Color(.5,.5,.5),
								 20));

BaseWidget::BaseWidget( Module* module, CrowdMonitor* lock,
			const Index NumVariables,
			const Index NumConstraints,
			const Index NumGeometries,
			const Index NumPicks,
			const Index NumMaterials,
			const Index NumModes,
			const Index NumSwitches,
			const Real widget_scale )
: module(module), lock(lock),
  solve(new ConstraintSolver), 
  NumVariables(NumVariables), NumConstraints(NumConstraints),
  NumGeometries(NumGeometries), NumPicks(NumPicks), NumMaterials(NumMaterials),
  constraints(NumConstraints), variables(NumVariables),
  geometries(NumGeometries), picks(NumPicks), materials(NumMaterials),
  NumModes(NumModes), NumSwitches(NumSwitches),
  modes(NumModes), mode_switches(NumSwitches), CurrentMode(0),
  widget_scale(widget_scale),
  epsilon(1e-6)
{
   for (Index i=0; i<NumSwitches; i++)
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
}


BaseWidget::~BaseWidget()
{
   Index index;
   
   for (index = 0; index < NumVariables; index++) {
      delete variables[index];
   }
   
   for (index = 0; index < NumConstraints; index++) {
      delete constraints[index];
   }
}


void
BaseWidget::SetScale( const double scale )
{
   widget_scale = scale;
   solve->SetEpsilon(epsilon*widget_scale);
   execute();
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
   Index s;
   for (s=0; s<NumSwitches; s++)
      if (modes[CurrentMode]&(1<<s))
	 mode_switches[s]->set_state(0);
   CurrentMode = (CurrentMode+1) % NumModes;
   for (s=0; s<NumSwitches; s++)
      if (modes[CurrentMode]&(1<<s))
	 mode_switches[s]->set_state(1);

   execute();
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
}


const MaterialHandle&
BaseWidget::GetMaterial( const Index mindex ) const
{
   ASSERT(mindex<NumMaterials);
   return materials[mindex]->getMaterial();
}


void
BaseWidget::execute()
{
   lock->write_lock();
   widget_execute();
   lock->write_unlock();
}


void
BaseWidget::geom_pick( int /* cbdata */)
{
}


void
BaseWidget::geom_release( int /* cbdata */)
{
}

void
BaseWidget::CreateModeSwitch( const Index snum, GeomObj* o )
{
   ASSERT(snum<NumSwitches);
   ASSERT(mode_switches[snum]==NULL);
   mode_switches[snum] = new GeomSwitch(o);
}


void
BaseWidget::SetMode( const Index mode, const long swtchs )
{
   ASSERT(mode<NumModes);
   modes[mode] = swtchs;
}


void
BaseWidget::FinishWidget()
{
   for (Index i=0; i<NumModes; i++)
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
   
   GeomGroup* sg = new GeomGroup;
   for (i=0; i<NumSwitches; i++) {
      if (modes[CurrentMode]&(1<<i))
	 mode_switches[i]->set_state(1);
      else
	 mode_switches[i]->set_state(0);
      sg->add(mode_switches[i]);
   }
   widget = new GeomSwitch(sg);

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
