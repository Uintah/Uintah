
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

enum { SBoxW_PointIUL, SBoxW_PointIUR, SBoxW_PointIDR, SBoxW_PointIDL,
       SBoxW_PointOUL, SBoxW_PointOUR, SBoxW_PointODR, SBoxW_PointODL,
       SBoxW_Dist, SBoxW_Hypo, SBoxW_Diag };

class ScaledBoxWidget : public BaseWidget {
public:
   ScaledBoxWidget( Module* module, double widget_scale );
   ScaledBoxWidget( const ScaledBoxWidget& );
   ~ScaledBoxWidget();

   virtual void execute();
   virtual void geom_moved(int, double, const Vector&, void*);
};


#endif
