
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
			const Index vars, const Index cons,
			const Index geoms, const Index mats )
{
   this->module = module;
   NumVariables = vars;
   NumConstraints = cons;
   NumGeometries = geoms;
   NumMaterials = mats;

   constraints = new BaseConstraint*[NumConstraints];
   variables = new Variable*[NumVariables];
   geometries = new GeomObj*[NumGeometries];
   materials = new MaterialProp*[NumMaterials];
}


BaseWidget::~BaseWidget()
{
   Index index;
   
   for (index = 0; index < NumVariables; index++) {
      delete variables[index];
   }
   delete variables;
   
   for (index = 0; index < NumConstraints; index++) {
      delete constraints[index];
   }
   delete constraints;

   for (index = 0; index < NumGeometries; index++) {
      delete geometries[index];
   }
   delete geometries;

   for (index = 0; index < NumMaterials; index++) {
      delete materials[index];
   }
   delete materials;
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

