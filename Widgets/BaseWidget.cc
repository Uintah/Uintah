
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

MaterialHandle BaseWidget::PointWidgetMaterial(new Material(Color(0,0,0), Color(.54,.60,1), Color(.5,.5,.5), 20));
MaterialHandle BaseWidget::EdgeWidgetMaterial(new Material(Color(0,0,0), Color(.54,.60,.66), Color(.5,.5,.5), 20));
MaterialHandle BaseWidget::SliderWidgetMaterial(new Material(Color(0,0,0), Color(.66,.60,.40), Color(.5,.5,.5), 20));
MaterialHandle BaseWidget::ResizeWidgetMaterial(new Material(Color(0,0,0), Color(.54,1,.60), Color(.5,.5,.5), 20));
MaterialHandle BaseWidget::SpecialWidgetMaterial(new Material(Color(0,0,0), Color(1,.54,.60), Color(.5,.5,.5), 20));
MaterialHandle BaseWidget::HighlightWidgetMaterial(new Material(Color(0,0,0), Color(.7,.7,.7), Color(0,0,.6), 20));

BaseWidget::BaseWidget( Module* module, CrowdMonitor* lock,
			const Index NumVariables,
			const Index NumConstraints,
			const Index NumGeometries,
			const Index NumMaterials,
			const Index NumPicks,
			const double widget_scale )
: NumVariables(NumVariables), NumConstraints(NumConstraints),
  NumGeometries(NumGeometries), NumMaterials(NumMaterials),
  NumPicks(NumPicks),
  constraints(NumConstraints), variables(NumVariables),
  geometries(NumGeometries), materials(NumMaterials),
  picks(NumPicks),
  module(module), widget_scale(widget_scale), lock(lock)
{
   solve = new ConstraintSolver();
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
}


double
BaseWidget::GetScale() const
{
   return widget_scale;
}


GeomSwitch*
BaseWidget::GetWidget()
{
   return widget;
}


void
BaseWidget::MoveDelta( const Vector& )
{
   Error("BaseWidget: Can't Move!");
}


Point
BaseWidget::ReferencePoint() const
{
   Error("BaseWidget: Can't Determine ReferencePoint!");
   return Point(0,0,0);
}


void
BaseWidget::SetMaterial( const Index mindex, const MaterialHandle m )
{
   ASSERT(mindex<NumMaterials);

   materials[mindex] = m;
}


MaterialHandle&
BaseWidget::GetMaterial( const Index mindex ) const
{
   ASSERT(mindex<NumMaterials);

   return materials[mindex];
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
BaseWidget::execute()
{
   lock->write_lock();
   widget_execute();
   lock->write_unlock();
}


void
BaseWidget::geom_pick( void* /* cbdata */)
{
}


void
BaseWidget::geom_release( void* /* cbdata */)
{
}


void
BaseWidget::geom_moved( int /* axis*/, double /* dist */,
			const Vector& /* delta */, void* /* cbdata */)
{
}


void
BaseWidget::FinishWidget(GeomObj* w)
{
   widget = new GeomSwitch(w);

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
