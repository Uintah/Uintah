
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
#include <Constraints/BaseConstraint.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Pick.h>
#include <Geom/Switch.h>

class CrowdMonitor;
class Module;

class BaseWidget {
public:
   BaseWidget( Module* module, CrowdMonitor* lock,
	       const Index vars, const Index cons,
	       const Index geoms, const Index mats,
	       const Index picks,
	       const Real widget_scale );
   BaseWidget( const BaseWidget& );
   ~BaseWidget();

   inline void SetScale( const Real scale );
   inline double GetScale() const;

   inline void SetEpsilon( const Real Epsilon );

   inline GeomSwitch* GetWidget();
   
   inline void SetMaterial( const Index mindex, const MaterialHandle m );
   inline MaterialHandle& GetMaterial( const Index mindex ) const;

   inline int GetState();
   inline void SetState( const int state );

   inline const Point& GetVar( const Index vindex ) const;
   
   void execute();

   virtual void geom_pick(void*);
   virtual void geom_release(void*);
   virtual void geom_moved(int, double, const Vector&, void*);

   BaseWidget& operator=( const BaseWidget& );
   int operator==( const BaseWidget& );

   void print( ostream& os=cout ) const;

protected:
   virtual void widget_execute()=0;
   Index NumConstraints;
   Index NumVariables;
   Index NumGeometries;
   Index NumMaterials;
   Index NumPicks;

   Array1<BaseConstraint*> constraints;
   Array1<Variable*> variables;
   Array1<GeomObj*> geometries;
   Array1<MaterialHandle> materials;
   Array1<GeomPick*> picks;

   GeomSwitch* widget;
   Real widget_scale;

   Module* module;
   CrowdMonitor* lock;

   void FinishWidget(GeomObj* w);

protected:
   // These affect ALL widgets!!!
   inline void SetPointWidgetMaterial( const MaterialHandle m );
   inline void SetEdgeWidgetMaterial( const MaterialHandle m );
   inline void SetSliderWidgetMaterial( const MaterialHandle m );
   inline void SetSpecialWidgetMaterial( const MaterialHandle m );
   inline void SetHighlightWidgetMaterial( const MaterialHandle m );
   
   static MaterialHandle PointWidgetMaterial;
   static MaterialHandle EdgeWidgetMaterial;
   static MaterialHandle SliderWidgetMaterial;
   static MaterialHandle SpecialWidgetMaterial;
   static MaterialHandle HighlightWidgetMaterial;
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


inline void
BaseWidget::SetEpsilon( const Real Epsilon )
{
   for (Index i=0; i<NumVariables; i++)
      variables[i]->SetEpsilon(Epsilon);
}


inline GeomSwitch*
BaseWidget::GetWidget()
{
   return widget;
}


inline int
BaseWidget::GetState()
{
   return widget->get_state();
}

   
inline void
BaseWidget::SetState( const int state )
{
   widget->set_state(state);
}

   
inline void
BaseWidget::SetMaterial( const Index mindex, const MaterialHandle m )
{
   ASSERT(mindex<NumMaterials);

   materials[mindex] = m;
}


inline MaterialHandle&
BaseWidget::GetMaterial( const Index mindex ) const
{
   ASSERT(mindex<NumMaterials);

   return materials[mindex];
}


inline ostream&
operator<<( ostream& os, BaseWidget& w )
{
   w.print(os);
   return os;
}


inline const Point&
BaseWidget::GetVar( const Index vindex ) const
{
   ASSERT(vindex<NumVariables);

   return variables[vindex]->Get();
}


inline void
BaseWidget::SetPointWidgetMaterial( const MaterialHandle m )
{
   PointWidgetMaterial = m;
}


inline void
BaseWidget::SetEdgeWidgetMaterial( const MaterialHandle m )
{
   EdgeWidgetMaterial = m;
}


inline void
BaseWidget::SetSliderWidgetMaterial( const MaterialHandle m )
{
   SliderWidgetMaterial = m;
}


inline void
BaseWidget::SetSpecialWidgetMaterial( const MaterialHandle m )
{
   SpecialWidgetMaterial = m;
}


inline void
BaseWidget::SetHighlightWidgetMaterial( const MaterialHandle m )
{
   HighlightWidgetMaterial = m;
}


#endif
