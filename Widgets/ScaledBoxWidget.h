
/*
 *  ScaledBoxWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_ScaledBox_Widget_h
#define SCI_project_ScaledBox_Widget_h 1

#include <Widgets/BaseWidget.h>


// Variable indexs
enum { SBoxW_PointIUL, SBoxW_PointIUR, SBoxW_PointIDR, SBoxW_PointIDL,
       SBoxW_PointOUL, SBoxW_PointOUR, SBoxW_PointODR, SBoxW_PointODL,
       SBoxW_Dist, SBoxW_Hypo, SBoxW_Diag };
// Material indexs
enum { SBoxW_PointMatl, SBoxW_EdgeMatl, SBoxW_SliderMatl, SBoxW_HighMatl };


class ScaledBoxWidget : public BaseWidget {
public:
   ScaledBoxWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   ScaledBoxWidget( const ScaledBoxWidget& );
   ~ScaledBoxWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);
};


#endif
