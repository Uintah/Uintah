
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
   virtual ~ViewWidget();

   virtual void widget_execute();
   virtual void geom_moved(int, double, const Vector&, int, const BState&);

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   View GetView();
   Vector GetUpVector();
   Real GetFOV() const;

   void SetView( const View& view );

   Real GetAspectRatio() const;
   void SetAspectRatio( const Real aspect );
   
   const Vector& GetEyeAxis();
   const Vector& GetUpAxis();
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

   // Material indexs
   enum { EyesMatl, ResizeMatl, ShaftMatl, FrustrumMatl };
   
protected:
   virtual clString GetMaterialName( const Index mindex ) const;   
   
private:
   Real ratio;
   Vector oldaxis1;
   Vector oldaxis2;
};


#endif
