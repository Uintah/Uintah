
/*
 *  FrameWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Frame_Widget_h
#define SCI_project_Frame_Widget_h 1

#include <Widgets/BaseWidget.h>


// Variable indexs
enum { FrameW_PointUL, FrameW_PointUR, FrameW_PointDR, FrameW_PointDL,
       FrameW_Dist1, FrameW_Dist2, FrameW_Hypo };
// Material indexs
enum { FrameW_PointMatl, FrameW_EdgeMatl, FrameW_HighMatl };


class FrameWidget : public BaseWidget {
public:
   FrameWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   FrameWidget( const FrameWidget& );
   ~FrameWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, void*);
};


#endif
