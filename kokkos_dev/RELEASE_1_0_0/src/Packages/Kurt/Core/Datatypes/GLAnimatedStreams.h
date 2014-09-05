#ifndef GLANIMATEDSTREAMS_H
#define GLANIMATEDSTREAMS_H

#include <Core/Thread/Mutex.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/Datatypes/VectorField.h>
#include <Core/Datatypes/ColorMap.h>
#include <Core/Geom/GeomObj.h>
#include <GL/gl.h>
#include <GL/glu.h>

namespace Kurt {

using namespace SCIRun;

struct streamerNode {
				// to keep track of length
  int counter;
  Point position;
  Vector normal;
  Material color;
  Vector tangent;
  streamerNode* next;
  
};

class GLAnimatedStreams : public GeomObj
{
public:

  GLAnimatedStreams(int id);

  GLAnimatedStreams(int id, VectorFieldHandle tex,
		   ColorMapHandle map);


  void SetVectorField( VectorFieldHandle vfh ){ 
    mutex.lock(); this->_vfH = vfh; init(); mutex.unlock();}
  void SetColorMap( ColorMapHandle map){
    mutex.lock(); this->_cmapH = map; cmapHasChanged = true;
    mutex.unlock(); }

  void Pause( bool p){ _pause = p; }
  void Normals( bool n) {_normalsOn = n; }
  void SetLineWidth( int w){ _linewidth = w; }
  void SetStepSize( double step){ _stepsize = step; }
  void SetWidgetLocation(Point p){ widgetLocation = p; }
  //void UseWidget(bool b){ _usesWidget = b; }
  GLAnimatedStreams(const GLAnimatedStreams&);
  ~GLAnimatedStreams();


#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  
  virtual GeomObj* clone();
  virtual void get_bounds(BBox& bb){
    Point min, max;
    _vfH->get_bounds(min, max );
    bb.extend(min); bb.extend(max);
  }
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);

  void setup();
  void preDraw();
  void draw();
  void postDraw();
  void cleanup();

  void drawWireFrame();

protected:


private:

  Mutex mutex;
  ColorMapHandle _cmapH;
  VectorFieldHandle _vfH;


  double slice_alpha;

  bool cmapHasChanged;
  bool _usesWidget;
  Point widgetLocation;

  bool _pause;
  bool _normalsOn;
  double _stepsize;
  int _linewidth;

  streamerNode** head;		// array of pointers to head node in
				// each solution
  streamerNode** tail;		// array of pointers to tail node in
				// each solution

  void newStreamer(int whichStreamer);
  void RungeKutta(Point& x, double h);
  void init();
  static const double FADE;
  static const int MAXN;

  int _numStreams;
  Point* fx;

  double maxwidth;
  double maxspeed;
  double minspeed;
};

} // End namespace Kurt

#endif
