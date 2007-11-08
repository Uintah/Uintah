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


/*----------------------------------------------------------------------
CLASS
    EditPath

    Interactive tool for editing and playback of camera path.

GENERAL INFORMATION

    Created by:
    J. Davison de St. Germain
    Department of Computer Science
    University of Utah
    July 2000
    
    Copyright (C) 2007 SCI Institute - University of Utah

KEYWORDS

DESCRIPTION
   EditPath module provides a set of interactive tools for creation and manipulation of
   camera path in Viewer. The basic features of the modules are as follows:
    
   - Interactive adding/inserting/deleting of key frames of current camera position in edited path
   - Navigation through existing key frames</para>
   - Automatic generation of circle path based on current camera position and position of reference widget
   - Looped and reversed path modes

PATTERNS

WARNING    
    
POSSIBLE REVISIONS
    Adding additional interpolation modes (subject to experiment)
    Adding support of PathWidget
----------------------------------------------------------------------*/


#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Dataflow/Network/Scheduler.h>
#include <Dataflow/Widgets/CrosshairWidget.h>
#include <Dataflow/Widgets/ArrowWidget.h>

#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/View.h>
#include <Core/Geometry/Plane.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Quaternion.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/CatmullRomSpline.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Thread/Time.h>

#include <iostream>
#include <vector>
#include <map>

using namespace std;

namespace SCIRun {

class EditPath : public Module
{
public:

  EditPath( GuiContext * ctx );

  virtual ~EditPath();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);

private:

  enum PathState_E { STOPPED, RUNNING };

  PathState_E              state_;

  GuiInt                   numKeyFramesTcl_;
  GuiInt                   uiInitializedTcl_;
  GuiString                pathFilenameTcl_;
  GuiDouble                currentFrameTcl_;

  GuiDouble                delayTcl_;

  GuiInt                   numberSubKeyFramesTcl_;

  GuiInt                   updateOnExecuteTcl_;
  GuiInt                   loopTcl_;
  GuiInt                   reverseTcl_;

  GuiInt                   showCircleWidgetTcl_;
  GuiInt                   showPathWidgetsTcl_;

  GuiInt                   circleNumPointsTcl_;

  bool                     pathLoaded_;

  double                   currentTime_; // Range: [0 - numKeyFrames]

  CatmullRomSpline<Point>  eyepointKeyFrames_;
  CatmullRomSpline<Point>  lookatKeyFrames_;
  CatmullRomSpline<Vector> upVectorKeyFrames_;
  vector<double>           fovKeyFrames_;
  vector<int>              numberSubKeyFrames_;

  GeometryOPort       * ogeom_;

  CrowdMonitor          widgetLock_;

  CrosshairWidget     * crossWidget_;
  GeomID                crossId_;
  map<ArrowWidget*,int> pathPointsMap_;
  GeomID                pathPointsId_;

  MaterialHandle        redMtl_;
  GeomHandle            pathGroup_;

  void makeCircle();

  void removeView();
  void savePathToFile();
  void loadPathFromFile();
  bool reset();
  void start();
  void stop();

  void highlight();

  void sendView();
  void updateView( bool forward );
  void updateTime();

  void   updateNumberSubKeyFrames();
  void   updateAllNumberSubKeyFrames();
  int    validateNumberSubKeyFrames( int frames );

  void toggleCircleWidget();
  void togglePathPoints( bool updateGeometry = false, // if 'updateGeometry', then re-build the path
                         bool updateView = true );    // if 'updateView', then move the eyepoint

  virtual void widget_moved( bool last, BaseWidget * widget );

  // Inherited from Module.  I need this to setup the callback
  // function for the scheduler.
  virtual void set_context(Network* network);
  static bool network_finished( void * module );

};

DECLARE_MAKER(EditPath)

EditPath::EditPath(GuiContext* ctx) :
  Module("EditPath", ctx, Filter, "Render", "SCIRun"),
  numKeyFramesTcl_( get_ctx()->subVar("numKeyFrames") ),
  uiInitializedTcl_( get_ctx()->subVar("uiInitialized") ),
  pathFilenameTcl_( get_ctx()->subVar("pathFilename") ),
  currentFrameTcl_( get_ctx()->subVar("currentFrame") ),
  delayTcl_( get_ctx()->subVar("delay") ),
  numberSubKeyFramesTcl_( get_ctx()->subVar("numSubKeyFrames") ),
  updateOnExecuteTcl_( get_ctx()->subVar("updateOnExecute") ),
  loopTcl_( get_ctx()->subVar("loop") ),
  reverseTcl_( get_ctx()->subVar("reverse") ),
  showCircleWidgetTcl_( get_ctx()->subVar("showCircleWidget") ),
  showPathWidgetsTcl_( get_ctx()->subVar("showPathWidgets") ),
  circleNumPointsTcl_( get_ctx()->subVar("circleNumPoints") ),
  pathLoaded_( false ),
  currentTime_( 0.0 ),
  eyepointKeyFrames_(0), lookatKeyFrames_(0), upVectorKeyFrames_(0),
  widgetLock_("EditPath module widget lock")
{
  state_ = STOPPED;

  ogeom_ = (GeometryOPort *)get_oport("Geometry");

  crossWidget_ = scinew CrosshairWidget(this, &widgetLock_, 0.01);
  crossWidget_->Connect( ogeom_ );
  crossWidget_->SetState(0);
  crossWidget_->SetPosition(Point(0, 0, 0));
  crossId_ = -1;

  pathPointsId_ = -1;

  need_execute_ = 1; // This allows the data file (if specified) to load.

  redMtl_    = scinew Material( Color(1.0, 0.0, 0.0) );
  pathGroup_ = scinew GeomGroup();
}

EditPath::~EditPath()
{
  sched_->remove_callback( network_finished, this );
}

///////////////////////////////////////////////////////////////////

void
EditPath::execute()
{
  if( !pathLoaded_ && pathFilenameTcl_.get() != "" ) {
    // The first time this module is execute it will load data from a file
    // if a file is specified.
    loadPathFromFile();
    pathLoaded_ = true;
    if( uiInitializedTcl_.get() ){
      get_gui()->execute( get_id() + " updateButtons" );
    }
  }

  if( state_ == RUNNING ) {

    sendView();
    Time::waitFor( delayTcl_.get() );
    updateTime();
    want_to_execute();    

  } else if( !updateOnExecuteTcl_.get() ) {

    if( numKeyFramesTcl_.get() > 0 ) {
      // If not running, then default to update one frame on each execute.
      updateTime();
      sendView();
    }
  }

} // end execute()

void
EditPath::sendView()
{
  //cout << "sendView: currentTime: " << currentTime_ << "\n";

  Point   eye         = eyepointKeyFrames_( currentTime_ );
  Point   lookatPoint = lookatKeyFrames_(currentTime_);

  showPathWidgetsTcl_.reset();  // Sync with TCL side.
  showCircleWidgetTcl_.reset(); // Sync with TCL side.

  if( showCircleWidgetTcl_.get() ) {
    crossWidget_->SetPosition( lookatKeyFrames_( currentTime_ ) );
  }

  if( showPathWidgetsTcl_.get() ) {
    Vector lookat = lookatPoint - eye;
    eye -= lookat; // Pull the eye back away from path point widget 
                   // so path point widget doesn't obscure view (as much).
  }

  View view( eye,
             lookatPoint,
             upVectorKeyFrames_(currentTime_),
             fovKeyFrames_[currentFrameTcl_.get()] );

  ogeom_->setView( 0, 0, view );
  currentFrameTcl_.set( currentTime_ );
  numberSubKeyFramesTcl_.set( numberSubKeyFrames_[ (int)currentTime_ ] );
}

void
EditPath::tcl_command(GuiArgs& args, void* userdata)
{   
  if (args[1] == "add_vp") {

    currentFrameTcl_.reset();       // Sync with TCL side.
    numberSubKeyFramesTcl_.reset(); // Sync with TCL side.

    int  numberSubKeyFrames = validateNumberSubKeyFrames( numberSubKeyFramesTcl_.get() );
    int  insertAtFrame = currentFrameTcl_.get() + 0.9999;

    printf("insert at %d\n", insertAtFrame);

    GeometryData * data = ogeom_->getData(0, 0, 1);
    if( data && data->view ) {
      eyepointKeyFrames_.insertData( insertAtFrame, data->view->eyep() );
      lookatKeyFrames_.insertData(   insertAtFrame, data->view->lookat() );
      upVectorKeyFrames_.insertData( insertAtFrame, data->view->up() );

      vector<double>::iterator iterFov = fovKeyFrames_.begin();
      vector<int>::iterator    iterSKF = numberSubKeyFrames_.begin();

      fovKeyFrames_.insert(       iterFov + insertAtFrame, data->view->fov() );
      numberSubKeyFrames_.insert( iterSKF + insertAtFrame, numberSubKeyFrames );
      numKeyFramesTcl_.set(       fovKeyFrames_.size() );

      togglePathPoints(); // redisplay the path points (if necessary)
    }
  }  
  else if (args[1] == "remove_vp") {
    removeView();
  }
  else if (args[1] == "rpl_vp") {
    printf( "warning, rpl_vp is not implemented\n" );
  }
  else if (args[1] == "delete_path") { 
    reset();
  }
  else if (args[1] == "run") {
    start();
  }
  else if (args[1] == "stop") {
    stop();
  }
  else if (args[1] == "toggleCircleWidget") {
    toggleCircleWidget();
  } 
  else if (args[1] == "togglePathPoints") {
    togglePathPoints();
  } 
  else if (args[1] == "doSavePath") {
    savePathToFile();
  } 
  else if (args[1] == "doLoadPath") {
    loadPathFromFile();
  } 
  else if (args[1] == "next_view") {
    updateView( true );
  }
  else if (args[1] == "prev_view") {
    updateView( false );
  }
  else if (args[1] == "make_circle") {
    makeCircle();
  }
  else if (args[1] == "update_num_key_frames") {
    updateNumberSubKeyFrames();
  }
  else if (args[1] == "update_all_num_key_frames") {
    updateAllNumberSubKeyFrames();
  }
  else{
    Module::tcl_command(args, userdata);
  }
}

// Makes sure the number of Sub key frames is in the range of 0 to 30.
// Updates TCL side if it is not.  Returns a valid speed.
int
EditPath::validateNumberSubKeyFrames( int frames )
{
  if( frames < 0 ) {
    frames = 0;
    numberSubKeyFramesTcl_.set( frames );
  }
  else if( frames > 30 ) {
    frames = 30;
    numberSubKeyFramesTcl_.set( frames );
  }
  return frames;
}

void
EditPath::updateNumberSubKeyFrames()
{
  numberSubKeyFramesTcl_.reset(); // Sync with TCL side.
  currentFrameTcl_.reset();       // Sync with TCL side.

  int subFrames = validateNumberSubKeyFrames( numberSubKeyFramesTcl_.get() );

  int currentKeyFrame = currentFrameTcl_.get();
  numberSubKeyFrames_[ currentKeyFrame ] = subFrames;

  togglePathPoints( true, false );
}

void
EditPath::updateAllNumberSubKeyFrames()
{
  numKeyFramesTcl_.reset();       // Sync with TCL side.
  numberSubKeyFramesTcl_.reset(); // Sync with TCL side.

  int  numKeyFrames = numKeyFramesTcl_.get();
  int  subFrames = validateNumberSubKeyFrames( numberSubKeyFramesTcl_.get() );

  for( int pos = 0; pos < numKeyFrames; pos++ ) {
    numberSubKeyFrames_[ pos ] = subFrames;
  }

  togglePathPoints( true, false );
}

void
EditPath::togglePathPoints( bool updateGeometry /* = false */, bool updateView /* = true */ )
{
  numKeyFramesTcl_.reset();    // Sync with TCL side.
  showPathWidgetsTcl_.reset(); // Sync with TCL side.

  int numKeyFrames = numKeyFramesTcl_.get();

  if( numKeyFrames == 0 ) return;

  if( ( pathPointsMap_.size() != numKeyFrames ) || updateGeometry ) {

    // (Re-)Build path widgets, lines, markers.

    pathPointsMap_.clear(); // WARNING! Do I need to delete each of the arrows individually?
    ((GeomGroup*)pathGroup_.get_rep())->remove_all();

    double        arrowWidgetRadius = 0.1;

    GeomLines   * lines   = scinew GeomLines();
    GeomSpheres * spheres = scinew GeomSpheres( arrowWidgetRadius / 2.0 ); // 1/2 the size of the arrow widgets

    lines->setLineWidth( 1.0 );

    // Create the visual representation path points...
    for( int idx = 0; idx < numKeyFrames; idx++ ) {

      ArrowWidget * arrowWidget = scinew ArrowWidget( this, &widgetLock_, arrowWidgetRadius );

      arrowWidget->SetState( 1 );
      arrowWidget->SetPosition( eyepointKeyFrames_[ idx ] );
      arrowWidget->SetDirection( lookatKeyFrames_[ idx ] - eyepointKeyFrames_[ idx ] );
      arrowWidget->Connect( ogeom_ );

      // Draw a mark at the location of each interpolated keyframe:
      int subFrames = numberSubKeyFrames_[ idx ];
      double speed  = 1.0 / (subFrames + 1);

      for( double step = 0; step < 1; step += speed ) {
        
        double time = idx + step;
        Point eyepCur  = eyepointKeyFrames_( time );

        time += speed;
        if( time > (idx + 1.0) ) { time = idx + 1.0; }
        Point eyepNext = eyepointKeyFrames_( time );

        spheres->add( eyepCur, redMtl_ );
        lines->add( eyepCur, redMtl_, eyepNext, redMtl_ );
      }

      ((GeomGroup*)pathGroup_.get_rep())->add( arrowWidget->GetWidget() );

      pathPointsMap_[ arrowWidget ] = idx;
    }
    ((GeomGroup*)pathGroup_.get_rep())->add( lines );
    ((GeomGroup*)pathGroup_.get_rep())->add( spheres );
  }

  if( pathPointsId_ != -1 ) {
    ogeom_->delObj( pathPointsId_ );
    pathPointsId_ = -1;
  }

  if( showPathWidgetsTcl_.get() ) {
    pathPointsId_ = ogeom_->addObj( pathGroup_, string("EditPath Path"), &widgetLock_ );
  }

  if( updateView ) {
    sendView();
  }
  ogeom_->flushViews();
}

void
EditPath::toggleCircleWidget()
{
  if( crossId_ < 0 ) {
    crossId_ = ogeom_->addObj( crossWidget_->GetWidget(), string("EditPath Crosshair"), &widgetLock_ );
  }

  showCircleWidgetTcl_.reset(); // Sync with TCL side.

  crossWidget_->SetState( showCircleWidgetTcl_.get() );
  ogeom_->flushViews();
}

void
EditPath::makeCircle()
{
  GeometryData * data = ogeom_->getData(0, 0, 1);
  if( !data || !data->view ) {
    printf( "Warning: view not valid... circle creation aborted\n" );
    return;
  }

  const Point  eye     = data->view->eyep();
  const Point  lookp   = crossWidget_->GetPosition(); //const Point  lookp = data->view->lookat();
  const Vector upV     = data->view->up();
  //const Vector lookdir = lookp - eye;

#if 0
  const double proj = Dot( lookdir.normal(), rdir.normal() ) * rdir.length();
  const double radius = (proj>0) ? proj : -proj;

  if( rdir.length() < 10e-7 || radius < 10e-5) {
    printf( "Warning: Bad rdir or radius... aborting circle creation.\n" );
    return;
  }
#endif
 
  circleNumPointsTcl_.reset(); // Sync with TCL side.
  int npts = circleNumPointsTcl_.get();

  if( npts < 10 ) {
    npts = 10;
    circleNumPointsTcl_.set( 10 );
  } else if( npts > 100 ) {
    npts = 100;
    circleNumPointsTcl_.set( 100 );
  }

  //  Point center = eye + (lookdir.normal() * proj);
  Plane plane( eye, upV );

  // WARNING... if the lookp is in the plane, this crashes... FIX ME
  Point center = plane.project( lookp );

  vector<Point> pts(npts);
       
  double fov     = data->view->fov();

  for( int i = 0; i < npts-1; i++ ) {

    double     angle = ( i * M_PI ) / (npts-1);
    Quaternion rot(cos(angle), upV*sin(angle));
    Vector     dir = rot.rotate(eye-center);

    pts[i] = center + dir;
    eyepointKeyFrames_.add( pts[i] );
    lookatKeyFrames_.add( lookp );

    Vector u;

    const Vector lookdir = lookp - (center + dir);

    if( lookdir == -dir ) {
      u = upV;
    } else {
      Vector tang;
      tang = Cross( lookdir, dir );
      u    = -Cross( tang, lookdir );
    }
    u.normalize();

    upVectorKeyFrames_.add( u );
    fovKeyFrames_.push_back( fov );
    numberSubKeyFrames_.push_back( 10 );
  }
  numKeyFramesTcl_.set( fovKeyFrames_.size() );

  togglePathPoints(); // redisplay the path points (if necessary)

} // end makeCircle()

void
EditPath::removeView()
{
  currentFrameTcl_.reset(); // Sync with TCL side.

  int    deleteFrame = currentFrameTcl_.get() + 0.5;

  printf("deleteframe %d\n", deleteFrame);

  eyepointKeyFrames_.removeData( deleteFrame );
  lookatKeyFrames_.removeData(   deleteFrame );
  upVectorKeyFrames_.removeData( deleteFrame );

  vector<double>::iterator iterFov = fovKeyFrames_.begin();
  vector<int>::iterator    iterSKF = numberSubKeyFrames_.begin();

  fovKeyFrames_.erase(   iterFov + deleteFrame );
  numberSubKeyFrames_.erase( iterSKF + deleteFrame );
  numKeyFramesTcl_.set( fovKeyFrames_.size() );

  togglePathPoints( true, false ); // redisplay the path points (if necessary)
}

void
EditPath::updateTime()
{
  reverseTcl_.reset();      // Sync with TCL side.
  currentFrameTcl_.reset();

  int  currentKeyFrame = currentFrameTcl_.get();
  int  numSubKeyFrames = numberSubKeyFrames_[ currentKeyFrame ];

  double speed = 1.0 / (numSubKeyFrames + 1);

  if( reverseTcl_.get() ) { speed *= -1; }

  //cout << "Current Time Was: " << currentTime_ << "\n";

  currentTime_ += speed;

  //cout << "CURRENT TIME IS : " << currentTime_ << "\n";

  if( currentTime_ >= (numKeyFramesTcl_.get()) ) {
    //cout << "in here\n";
    if( !loopTcl_.get() ) {
      currentTime_ = numKeyFramesTcl_.get()-1;
      stop();
    } else {
      currentTime_ = speed;
    }
  }
  else if( currentTime_ < 0 ) {
    //cout << "IN HERE\n";
    if( !loopTcl_.get() ) {
      stop();
    } else {
      currentTime_ = numKeyFramesTcl_.get() + speed; // speed is negative so add it.
    }
  }
  currentFrameTcl_.set( currentTime_ );
  numberSubKeyFramesTcl_.set( numberSubKeyFrames_[ (int)currentTime_ ] );

  //cout << "current time: " << currentTime_ << "\n";
}

void
EditPath::updateView( bool forward )
{
  if( numKeyFramesTcl_.get() == 0 ) {
    return;
  }

  int currentKeyFrame = currentFrameTcl_.get();

  if( forward ) {
    currentKeyFrame++;
    if( currentKeyFrame > (numKeyFramesTcl_.get()-1) ) {
      currentKeyFrame = 0;
    }
  } else {
    currentKeyFrame--;
    if( currentKeyFrame < 0 ) {
      currentKeyFrame = numKeyFramesTcl_.get()-1;
    }
  }

  currentTime_ = currentKeyFrame;

  //cout << "UPDATING CURRENT TIME TO: " << currentTime_ << "\n";

  sendView();
}

void
EditPath::savePathToFile()
{
  numKeyFramesTcl_.reset(); // Sync with tcl side.
  pathFilenameTcl_.reset(); // Sync with tcl side.

  int numKeyFrames = numKeyFramesTcl_.get();

  if( numKeyFrames == 0 ) {
    return;
  }

  FILE * fp = fopen( pathFilenameTcl_.get().c_str(), "w" );

  if( fp == NULL ) {
    // TODO: Popup GUI message.
    printf( "ERROR, could not open file '%s' (for path writing).\n", pathFilenameTcl_.get().c_str() );
    error(  string( "Could not open file '") + pathFilenameTcl_.get() + "' (for path writing).\n" );
    return;
  }

  fprintf( fp, "%d\n", numKeyFrames );
  for( int pos = 0; pos < numKeyFrames; pos++ ) {
    fprintf( fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
             eyepointKeyFrames_[ pos ].x(), eyepointKeyFrames_[ pos ].y(), eyepointKeyFrames_[ pos ].z(),
             lookatKeyFrames_[ pos ].x(),   lookatKeyFrames_[ pos ].y(),   lookatKeyFrames_[ pos ].z(),
             upVectorKeyFrames_[ pos ].x(), upVectorKeyFrames_[ pos ].y(), upVectorKeyFrames_[ pos ].z(),
             fovKeyFrames_[ pos ], numberSubKeyFrames_[ pos ] );
  }
  fclose( fp );
}

void
EditPath::loadPathFromFile()
{
  pathFilenameTcl_.reset(); // Sync with tcl side.

  FILE * fp = fopen( pathFilenameTcl_.get().c_str(), "r" );

  if( fp == NULL ) {
    // TODO: Popup GUI message.
    error(  string( "Could not open file '") + pathFilenameTcl_.get() + "' (for path reading).\n" );
    printf( "ERROR, could not open file '%s' (for path reading).\n", pathFilenameTcl_.get().c_str() );
    return;
  }

  // TODO: FIXME: this is broken for reading in two files... need to either clear all
  // the points and start fresh, or just add the points to the existing ones.

  int numKeyFrames;
  fscanf( fp, "%d\n", &numKeyFrames );

  numKeyFramesTcl_.set( numKeyFrames );

  for( int pos = 0; pos < numKeyFrames; pos++ ) {
    double x, y, z, val;
    fscanf( fp, "%lf %lf %lf", &x, &y, &z );
    eyepointKeyFrames_.add( Point( x, y, z ) );
    fscanf( fp, "%lf %lf %lf", &x, &y, &z );
    lookatKeyFrames_.add( Point( x, y, z ) );
    fscanf( fp, "%lf %lf %lf", &x, &y, &z );
    upVectorKeyFrames_.add( Vector( x, y, z ) );
    fscanf( fp, "%lf", &val );
    fovKeyFrames_.push_back( val );
    fscanf( fp, "%lf", &val );
    numberSubKeyFrames_.push_back( val );
    //cout << "loaded lookat of: " << lookatKeyFrames_[ pos ] << "\n";
  }
  fclose( fp );
}

bool
EditPath::reset()
{
  state_ = STOPPED;
  eyepointKeyFrames_.clear();
  lookatKeyFrames_.clear();
  upVectorKeyFrames_.clear();
  fovKeyFrames_.clear();
  numberSubKeyFrames_.clear();
  numKeyFramesTcl_.set( 0 );

  if( pathPointsId_ != -1 ) {
    showPathWidgetsTcl_.set( 0 );
    pathPointsMap_.clear(); // WARNING! Do I need to delete each of the arrows individually?
    ogeom_->delObj( pathPointsId_ ); // Clean up previous points.
    ogeom_->flush();
    pathPointsId_ = -1;
  }
}

void
EditPath::start()
{
  int numKeyFrames = numKeyFramesTcl_.get();

  if( numKeyFrames > 0 ) {
    if( currentTime_ >= numKeyFrames-1 ) {
      currentTime_ = 0;
    }
    state_ = RUNNING;
    want_to_execute();
  }
}

void
EditPath::stop()
{
  state_ = STOPPED;
  if( uiInitializedTcl_.get() ){
    get_gui()->lock();
    get_gui()->execute( get_id() + " stop" );
    get_gui()->unlock();
  }
}

// 'Blinks' the module on the Network Editor... this is used to indicate that 
// the path has been updated.
void
EditPath::highlight()
{
  if( uiInitializedTcl_.get() ){
    get_gui()->lock();
    get_gui()->execute( get_id() + " highlight" );
    get_gui()->unlock();
  }
}

// This is a callback made by the scheduler when the network finishes.
bool
EditPath::network_finished( void * module )
{
  EditPath * epModule = (EditPath*)module;

  printf( "EditPath: network_finished: %2.2lf\n", epModule->currentTime_ );

  epModule->updateOnExecuteTcl_.reset(); // Sync with TCL side.

  if( epModule->updateOnExecuteTcl_.get() )
    {
      epModule->highlight();
      epModule->updateTime();
      epModule->sendView();
    }
  return true;
}

void
EditPath::set_context( Network* network )
{
  Module::set_context( network );
  // Set up a callback to call after the network has finished executing.
  sched_->add_callback( network_finished, this );
}

void
EditPath::widget_moved( bool last, BaseWidget * widget )
{
  if( last ) {
    ArrowWidget * arrow = dynamic_cast<ArrowWidget*>(widget);
    if( arrow == NULL ) return; // it was the center widget...
    int index = pathPointsMap_[ arrow ];

    const Point & curPos = eyepointKeyFrames_[ index ];
    const Point & newPos = arrow->GetPosition();

    const Vector & newDir = arrow->GetDirection();
    Vector curDir = lookatKeyFrames_[ index ] - curPos;

    // curDir.normalize(); // For comparison later, must normalize... however, the comparison fails anyway.
                           // Also, need this non-normalized to get its lenght.

    if( curPos != newPos ) { // Position changed... update:
      //cout << "Position changed\n";
      eyepointKeyFrames_[ index ] = newPos;
      arrow->SetDirection( lookatKeyFrames_[ index ] - newPos );

      togglePathPoints( true, false );
    } 
    else if( newDir != curDir ) { // Direction of arrow changed...

      // The above comparison isn't accurate... FIXME/TODO

      //cout << "Direction changed\n";
      //cout << "new: " << newDir << "\n";
      //cout << "old: " << curDir << "\n";

      // APPROXIMATION FOR NOW:... just create a new lookat point...
      lookatKeyFrames_[ index ] = curPos + ( arrow->GetDirection() * curDir.length() );
      crossWidget_->SetPosition( lookatKeyFrames_[ index ] );
    }
    ogeom_->flushViews();
  }
}

} // End namespace SCIRun

