
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


class PointWidget : public BaseWidget {
public:
   PointWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   PointWidget( const PointWidget& );
   ~PointWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetPosition( const Point& );
   const Point& GetPosition() const;

   // Variable indexs
   enum { PointVar };
   // Material indexs
   enum { PointMatl, HighMatl };
};


#endif
