
/*
 *  ViewWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_View_Widget_h
#define SCI_project_View_Widget_h 1

#include <Widgets/BaseWidget.h>
#include <Geom/View.h>


class ViewWidget : public BaseWidget {
public:
   ViewWidget( Module* module, CrowdMonitor* lock, double widget_scale,
	       const Real AspectRatio=1.3333);
   ViewWidget( const ViewWidget& );
   ~ViewWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, int);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   View& GetView();
   Vector& GetUpVector();
   Real GetFOV() const;

   void SetAspectRatio( const Real aspect );
   Real GetAspectRatio() const;
   
   Vector GetEyeAxis();
   Vector GetUpAxis();
   Point GetFrontUL();
   Point GetFrontUR();
   Point GetFrontDR();
   Point GetFrontDL();
   Point GetBackUL();
   Point GetBackUR();
   Point GetBackDR();
   Point GetBackDL();

   // Variable indexs
   enum { EyeVar, ForeVar, LookAtVar, UpVar, UpDistVar, EyeDistVar, FOVVar };
   
private:
   Real ratio;
   Vector oldaxis1;
   Vector oldaxis2;
};


#endif
