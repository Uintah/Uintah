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
 *  PathWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Path_Widget_h
#define SCI_project_Path_Widget_h 1

#include <Dataflow/Widgets/BaseWidget.h>


namespace SCIRun {

class PathPoint;

class PathWidget : public BaseWidget {
  friend class PathPoint;
public:
  PathWidget( Module* module, CrowdMonitor* lock, double widget_scale,
	      Index num_points=10 );
  PathWidget( const PathWidget& );
  virtual ~PathWidget();

  virtual void redraw();
  virtual void geom_moved(GeomPickHandle, int, double, const Vector&, int,
			  const BState&, const Vector &pick_offset);

  virtual void MoveDelta( const Vector& delta );
  virtual Point ReferencePoint() const;

  Index GetNumPoints() const;
   
protected:
  virtual string GetMaterialName( const Index mindex ) const;   
   
private:
  RealVariable* dist;
  RealVariable* hypo;
   
  GeomGroup* pointgroup;
  GeomGroup* tangentgroup;
  GeomGroup* orientgroup;
  GeomGroup* upgroup;
  GeomGroup* splinegroup;

  vector<PathPoint*> points;

  void GenerateSpline();
};


} // End namespace SCIRun

#endif
