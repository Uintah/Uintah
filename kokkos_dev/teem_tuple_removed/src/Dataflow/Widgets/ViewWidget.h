/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Dataflow/Widgets/BaseWidget.h>
#include <Core/Geom/View.h>


namespace SCIRun {


class ViewWidget : public BaseWidget {
public:
  ViewWidget( Module* module, CrowdMonitor* lock, double widget_scale,
	      double AspectRatio = 1.3333);
  ViewWidget( const ViewWidget& );
  virtual ~ViewWidget();

  virtual void redraw();
  virtual void geom_moved(GeomPickHandle, int, double, const Vector&, int,
			  const BState&, const Vector &pick_offset);

  virtual void MoveDelta( const Vector& delta );
  virtual Point ReferencePoint() const;

  View GetView();
  Vector GetUpVector();
  double GetFOV() const;

  void SetView( const View& view );

  double GetAspectRatio() const;
  void SetAspectRatio( const double aspect );
   
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
  virtual string GetMaterialName( const Index mindex ) const;   
   
private:
  double ratio;
  Vector oldaxis1;
  Vector oldaxis2;
};


} // End namespace SCIRun

#endif
