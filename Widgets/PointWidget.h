
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


// Variable indexs
enum { PointW_Point };
// Material indexs
enum { PointW_PointMatl, PointW_HighMatl };


class PointWidget : public BaseWidget {
public:
   PointWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   PointWidget( const PointWidget& );
   ~PointWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);
};


#endif
