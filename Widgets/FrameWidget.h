
/*
 *  FrameWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Frame_Widget_h
#define SCI_project_Frame_Widget_h 1

#include <Widgets/BaseWidget.h>

enum { FrameW_PointUL, FrameW_PointUR, FrameW_PointDR, FrameW_PointDL,
       FrameW_Dist, FrameW_Hypo };

class FrameWidget : public BaseWidget {
public:
   FrameWidget( Module* module );
   FrameWidget( const FrameWidget& );
   ~FrameWidget();

   virtual void execute();
   virtual void geom_moved(int, double, const Vector&, void*);
};


#endif
