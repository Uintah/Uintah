
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

MaterialHandle BaseWidget::PointWidgetMaterial(new Material(Color(0,0,0),
							    Color(.54,.60,1),
							    Color(.5,.5,.5),
							    20));
MaterialHandle BaseWidget::EdgeWidgetMaterial(new Material(Color(0,0,0),
							   Color(.54,.60,.66),
							   Color(.5,.5,.5),
							   20));
MaterialHandle BaseWidget::SliderWidgetMaterial(new Material(Color(0,0,0),
							     Color(.66,.60,.40),
							     Color(.5,.5,.5),
							     20));
MaterialHandle BaseWidget::ResizeWidgetMaterial(new Material(Color(0,0,0),
							     Color(.54,1,.60),
							     Color(.5,.5,.5),
							     20));
MaterialHandle BaseWidget::SpecialWidgetMaterial(new Material(Color(0,0,0),
							      Color(1,.54,.60),
							      Color(.5,.5,.5),
							      20));
MaterialHandle BaseWidget::HighlightWidgetMaterial(new Material(Color(0,0,0),
								Color(.8,0,0),
								Color(.5,.5,.5),
								20));

BaseWidget::BaseWidget( Module* module, CrowdMonitor* lock,
			const Index NumVariables,
			const Index NumConstraints,
			const Index NumGeometries,
			const Index NumPicks,
			const Index NumModes,
			const Index NumSwitches,
			const double widget_scale )
: NumVariables(NumVariables), NumConstraints(NumConstraints),
  NumGeometries(NumGeometries), NumPicks(NumPicks),
  NumModes(NumModes), NumSwitches(NumSwitches),
  constraints(NumConstraints), variables(NumVariables),
  geometries(NumGeometries), picks(NumPicks),
  modes(NumModes), mode_switches(NumSwitches),
  module(module), widget_scale(widget_scale), lock(lock),
  solve(new ConstraintSolver), CurrentMode(0),
  PointMaterial(PointWidgetMaterial),
  EdgeMaterial(EdgeWidgetMaterial),
  SliderMaterial(SliderWidgetMaterial),
  ResizeMaterial(ResizeWidgetMaterial),
  SpecialMaterial(SpecialWidgetMaterial),
  HighlightMaterial(HighlightWidgetMaterial),
  epsilon(1e-6)
{
   for (Index i=0; i<NumSwitches; i++)
      mode_switches[i] = NULL;
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
BaseWidget::MoveDelta( const Vector& )
{
   Error("BaseWidget can't MoveDelta!");
}


Point
BaseWidget::ReferencePoint() const
{
   Error("BaseWidget can't find ReferencPoint!");
   return Point(0,0,0);
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
