/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  LightWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Light_Widget_h
#define SCI_project_Light_Widget_h 1

#include <Dataflow/Widgets/BaseWidget.h>


namespace SCIRun {

enum LightType {
  DirectionalLight=0, PointLight=1, SpotLight=2, AreaLight=3, NumLightTypes=4
};

class FrameWidget;

class LightWidget : public BaseWidget {
public:
   LightWidget( Module* module, CrowdMonitor* lock, double widget_scale );
   LightWidget( Module* module, CrowdMonitor* lock, 
		double widget_scale, Point source,
		Point direct, Point cone, double rad, double rat );
   LightWidget( const LightWidget& );
   
   ~LightWidget();
   void init( Module* module);
   virtual void redraw();
   virtual void geom_moved(GeomPickHandle, int, double, const Vector&, int,
			   const BState&, const Vector &pick_offset);

   virtual void NextMode();

   virtual void MoveDelta( const Vector& delta );
   virtual Point ReferencePoint() const;

   void SetLightType( const LightType lighttype );
   LightType GetLightType() const;
   
  const Vector& GetAxis();
  Point GetSource() const;
  Point GetPointAt() const;
  Point GetCone() const;
  void SetPointAt( const Point& p );
  void SetCone( const Point& p );
  double GetRadius() const;
  void SetRadius( double r );
  double GetRatio() const;
  void SetRatio( double r );

   // Variable indexs
   enum { SourceVar, DirectVar, ConeVar, DistVar, RadiusVar, RatioVar };

   // Variable Indexs
   enum { SourceMatl, ArrowMatl, PointMatl, ConeMatl };

protected:
   virtual string GetMaterialName( const Index mindex ) const;   
   
private:
   LightType ltype;

   FrameWidget* arealight;

   Vector oldaxis;
};


} // End namespace SCIRun

#endif
