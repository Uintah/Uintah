
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

#include <Constraints/manifest.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Constraints/BaseConstraint.h>
#include <Geom.h>



class BaseWidget {
public:
   BaseWidget( Module* module,
	       const Index vars, const Index cons,
	       const Index geoms, const Index mats );
   BaseWidget( const BaseWidget& );
   ~BaseWidget();

   inline void SetScale( const double scale );
   inline double GetScale() const;

   inline ObjGroup* GetWidget();

   virtual void execute();
   const Point& GetVar( const Index vindex ) const;
   
   virtual void geom_pick(void*);
   virtual void geom_release(void*);
   virtual void geom_moved(int, double, const Vector&, void*);

   BaseWidget& operator=( const BaseWidget& );
   int operator==( const BaseWidget& );

   void print( ostream& os=cout ) const;

protected:
   Index NumConstraints;
   Index NumVariables;
   Index NumGeometries;
   Index NumMaterials;

   BaseConstraint** constraints;
   Variable** variables;
   GeomObj** geometries;
   MaterialProp** materials;

   ObjGroup* widget;
   double widget_scale;

   Module* module;
};

inline ostream& operator<<( ostream& os, BaseWidget& w );


inline void
BaseWidget::SetScale( const double scale )
{
   widget_scale = scale;
}


inline double
BaseWidget::GetScale() const
{
   return widget_scale;
}


inline ObjGroup*
BaseWidget::GetWidget()
{
   return widget;
}


inline ostream&
operator<<( ostream& os, BaseWidget& w )
{
   w.print(os);
   return os;
}


#endif
