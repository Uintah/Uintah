
/*
 *  PointWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Point_Widget_h
#define SCI_project_Point_Widget_h 1

#include <Widgets/BaseWidget.h>
#include <Geom/Material.h>

class PointWidget : public BaseWidget {
   GeomMaterial *sphMaterial;
public:
   PointWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   PointWidget( const PointWidget& );
   virtual ~PointWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, int);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetPosition( const Point& );
   Point GetPosition() const;

   void SetMaterial( const MaterialHandle& );
   MaterialHandle GetMaterial();

   // Variable indexs
   enum { PointVar };
};


#endif
