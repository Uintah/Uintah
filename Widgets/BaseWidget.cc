
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
   
   for (vindex=0; vindex<NumVariables; vindex++)
      variables[vindex]->Resolve();
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

