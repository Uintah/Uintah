
/*
 *  ScaledCubeWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_ScaledCube_Widget_h
#define SCI_project_ScaledCube_Widget_h 1

#include <Widgets/BaseWidget.h>

enum { SCubeW_PointIUL, SCubeW_PointIUR, SCubeW_PointIDR, SCubeW_PointIDL,
       SCubeW_PointOUL, SCubeW_PointOUR, SCubeW_PointODR, SCubeW_PointODL,
       SCubeW_Dist, SCubeW_Hypo, SCubeW_Diag };

class ScaledCubeWidget : public BaseWidget {
public:
   ScaledCubeWidget( Module* module, double widget_scale );
   ScaledCubeWidget( const ScaledCubeWidget& );
   ~ScaledCubeWidget();

   virtual void execute();
   virtual void geom_moved(int, double, const Vector&, void*);
};


#endif
