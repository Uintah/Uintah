
/*
 *  CrosshairWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Crosshair_Widget_h
#define SCI_project_Crosshair_Widget_h 1

#include <Widgets/BaseWidget.h>


class CrosshairWidget : public BaseWidget {
public:
   CrosshairWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   CrosshairWidget( const CrosshairWidget& );
   ~CrosshairWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetPosition( const Point& );
   const Point& GetPosition() const;

   // They should be orthogonal.
   void SetAxes( const Vector& v1, const Vector& v2, const Vector& v3 );
   void GetAxes( Vector& v1, Vector& v2, Vector& v3 ) const;

   // Variable indexs
   enum { CenterVar };
   
private:
   Vector axis1, axis2, axis3;
};


#endif
