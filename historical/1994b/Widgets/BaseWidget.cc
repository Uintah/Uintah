
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


BaseWidget::BaseWidget( Module* module,
			const Index NumVariables,
			const Index NumConstraints,
			const Index NumGeometries,
			const Index NumMaterials,
			const double widget_scale )
: NumVariables(NumVariables), NumConstraints(NumConstraints),
  NumGeometries(NumGeometries), NumMaterials(NumMaterials),
  constraints(NumConstraints), variables(NumVariables),
  geometries(NumGeometries), materials(NumMaterials),
  module(module), widget_scale(widget_scale)
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

   for (index = 0; index < NumGeometries; index++) {
      delete geometries[index];
   }
}


void
BaseWidget::execute()
{
}


const Point&
BaseWidget::GetVar( const Index vindex ) const
{
   ASSERT(vindex<NumVariables);

   return variables[vindex]->Get();
}


void
BaseWidget::geom_pick( void* cbdata )
{
}


void
BaseWidget::geom_release( void* cbdata )
{
}


void
BaseWidget::geom_moved( int axis, double dist, const Vector& delta,
			void* cbdata )
{
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

