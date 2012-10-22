/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef GLANIMATEDSTREAMS_H
#define GLANIMATEDSTREAMS_H

#include <sci_defs/ogl_defs.h>

#include <Core/Thread/Mutex.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/GeomObj.h>

#include <sci_gl.h>

namespace Uintah {

using SCIRun::Point;
using SCIRun::Ray;
using SCIRun::Vector;
using SCIRun::BBox;
using SCIRun::Transform;
using SCIRun::Field;
using SCIRun::FieldHandle;
using SCIRun::ColorMap;
using SCIRun::ColorMapHandle;
using SCIRun::Material;
using SCIRun::GeomObj;
using SCIRun::DrawInfoOpenGL;
using SCIRun::GeomSave;
using SCIRun::Mutex;
using SCIRun::Piostream;
using SCIRun::PersistentTypeID;

struct streamerNode {
				// to keep track of length
  int counter;
  double length;
  Point position;
  Vector normal;
  Material color;
  Vector tangent;
  streamerNode* next;
  
};


#define STREAM_LIGHT_WIRE 0
#define STREAM_LIGHT_CURVE 1
  
class GLAnimatedStreams : public GeomObj
{
public:

  GLAnimatedStreams();

  GLAnimatedStreams(FieldHandle tex, ColorMapHandle map);


  void SetVectorField( FieldHandle vfh );
  
  // tries to keep going with a new vector field.  If the new field is
  // incompatable with the old one then the effective call is to
  // SetVectorField().
  void ChangeVectorField( FieldHandle new_vfh );
  void SetColorMap( ColorMapHandle map);

  void ResetStreams();
  
  void Pause( bool p){ _pause = p; }
  void Normals( bool n) {_normalsOn = n; }
  void Lighting( bool l) {_lighting = l; }
  void UseDeltaT(bool use_deltat) { _use_dt = use_deltat; }
  void SetLineWidth( int w){ _linewidth = w; }
  void SetStepSize( double step){ _stepsize = step; }
  void SetWidgetLocation(Point p){ widgetLocation = p; }
  void IncrementFlow();
  void SetNormalMethod( int method );
  void SetDeltaT(double dt) { _delta_T = dt; }
  //void UseWidget(bool b){ _usesWidget = b; }
  GLAnimatedStreams(const GLAnimatedStreams&);
  ~GLAnimatedStreams();

  virtual void draw(DrawInfoOpenGL*, Material*, double time);
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox& bb){
    bb = vfh_->mesh()->get_bounding_box();
  }
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const std::string& format, GeomSave*);

  void setup();
  void preDraw();
  void draw();
  void postDraw();
  void cleanup();

  void drawWireFrame();

protected:
  bool interpolate( FieldHandle f, const Point& p, Vector& v);
private:

  Mutex mutex;
  FieldHandle vfh_;
  ColorMapHandle _cmapH;

  bool _pause;
  bool _normalsOn;
  bool _lighting;
  bool _use_dt;
  double _stepsize;

  Point* fx;

  streamerNode** tail;		// array of pointers to tail node in
				// each solution
  streamerNode** head;		// array of pointers to head node in
				// each solution

  int _numStreams;
  int _linewidth;
  int flow;
  double _delta_T;
  int _normal_method;
  bool _usesWidget;
  Point widgetLocation;

  double slice_alpha;

  bool cmapHasChanged;

  

  void newStreamer(int whichStreamer);
  void RungeKutta(Point& x, double h);
  void init();
  void initColorMap();
  void DecrementFlow();
  void AdvanceStreams(int start, int end);
  static const double FADE;
  static const int MAXN;


  double maxwidth;
  double maxspeed;
  double minspeed;


  // opengl states
  bool gl_lighting_disabled;
};
 
} // namespace Uintah


#endif
