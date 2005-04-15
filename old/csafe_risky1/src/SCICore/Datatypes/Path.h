/*----------------------------------------------------------------------
CLASS
    Path

    Datatype for storing and manipulating camera path. Used in EditPath module

GENERAL INFORMATION

    Created by:
    Alexei Samsonov
    Department of Computer Science
    University of Utah
    July 2000
    
    Copyright (C) 2000 SCI Group

KEYWORDS
    Quaternion

DESCRIPTION
    Provides much of functionality of EditPath module. 

PATTERNS
   
WARNING    
    

POSSIBLE REVISIONS
    Adding additional interpolation modes (subject to experiment)
----------------------------------------------------------------------*/

#ifndef SCI_SCICore_Datatypes_Path_h
#define SCI_SCICore_Datatypes_Path_h 1

#include <SCICore/share/share.h>
#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/Handle.h>

#include <SCICore/Containers/String.h>
#include <SCICore/Math/LinearPWI.h>
#include <SCICore/Math/CubicPWI.h>

#include <SCICore/Geom/View.h>
#include <SCICore/Geometry/Quaternion.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore {
namespace Datatypes {

using namespace SCICore::Math;
using namespace SCICore::Geometry;

using SCICore::Containers::LockingHandle;
using SCICore::Containers::Handle;
using SCICore::Containers::Array1;
using SCICore::GeomSpace::View;
using SCICore::Containers::clString;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class Path;
typedef LockingHandle<Path> PathHandle;

//typedef Handle<PiecewiseInterp<Point> > PointPWI;
//typedef Handle<PiecewiseInterp<Vector> > VectorPWI;
//typedef Handle<PiecewiseInterp<double> > SimplePWI;


enum {KEYFRAMED=0, LINEAR, CUBIC};
enum {NO_ACC=0, SMOOTH, USERMODE};

class SCICORESHARE Path : public Datatype {
 
  double    pathP;                 // current value of distance from the start point
  int       keyF;                  // keyframe nearest to the current path point
  double    path_prm;              // current value of corresponding parameter
                                   // (changes from 0 to 1)

  int       is_loop, is_built, is_back, is_acc, is_run;
  
  int       path_t;
  int       acc_t;
  
  int       ns;                    // discretization of path
  double    step_size;              // average distance between neighboor path points              
  
  // input data for path calculation

  Array1<View>   keyViews;         // keyframes
  Array1<double> param;            // and corresponding parameter value(based on linear distance)
  double         path_dist;        // total distance of the current path

  Array1<Quaternion> rotFrames;    // quaternion representation of camera frame orientation
				   // (direction of view and up vector)
  Array1<double> q_param;          // quaternion parametrization (by angle)
  				   // every interval of param linearly maps to q_param
  

  Array1<double> speedVal;         // speed values at each key frame
  Array1<int>    accPatt;
  
  Array1<double> dist_prm;         // arc-length parametrization

  PiecewiseInterp<Point>* eyeP;
  PiecewiseInterp<Point>* lookatP;
  PiecewiseInterp<Vector>* upV;
  PiecewiseInterp<double>* fov;
  PiecewiseInterp<double>* speed;

  // smooth start/end mode support
  Array1<double> sint;
  int count;
  double sin_series;
  
  int calc_view(double, View&);    // return number of current interval and hence number of current view
  bool resample_path();
  double get_delta();

  // syncronization between arc-length parameter and initial parametrization
  inline void set_param();
  inline void set_arc_param();

public:
   
    void   reset();
    void   del_keyF(int);
    bool   ins_keyF(int, const View&, double speed=1, int acc_patt=NO_ACC);
    bool   add_keyF(const View&, double speed=1, int acc_patt=NO_ACC);
    int    get_acc_t (int n=0) const;
    bool   set_acc_t(int);
    double get_speed_val(int) const;

    bool get_keyF(int, View&, double&);
    int  get_num_views() const;

    // path points manipulation
    bool   set_step(double);
    double get_step() const;
       
    bool     set_path_t(int t);
    int      get_path_t() const;
  
    bool build_path();
    
    inline bool is_looped() const;
    inline bool is_backed() const;
    inline bool is_started() const;
    
    bool set_loop(bool);
    bool set_back(bool);
    void seek_start();
    void seek_end();
    void stop();

    // return next path view, the nearest view behind, current speed and acceleration
    bool get_nextPP(View&, int&, double&, double&);
    bool get_nextKF(View&, int&, double&, double&); 
  
    Path();
    virtual Path* clone();
    virtual ~Path();

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

inline bool Path::is_looped() const{
  return is_loop;
}

inline bool Path::is_backed() const{
  return is_back;
}

inline bool Path::is_started() const{
  return is_run;
}

inline void Path::set_param(){
  int ind=(int)((pathP/path_dist)*(ns-1));
      
  if (ind == ns-1)
    path_prm=dist_prm[ind];
  else {
    double delta=path_dist/(ns-1);
    double w=(pathP-ind*delta)/delta;
    path_prm=dist_prm[ind+1]*w+(1-w)*dist_prm[ind];
  }
}

inline void Path::set_arc_param(){
  int sz=dist_prm.size();
  ASSERT(sz>=ns && ns > 0 && path_prm<=dist_prm[sz-1]);
  int j=1;
  while ((path_prm-dist_prm[j])>10e-13) j++;
  
  double  w=(path_prm-dist_prm[j-1])/(dist_prm[j]-dist_prm[j-1]);
  double delta=path_dist/(ns-1);
  pathP=(j*w+(1-w)*(j-1))*delta;    
}


} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.3.2.1  2000/10/19 05:17:23  sparker
// Merge changes from main branch into csafe_risky1
//
// Revision 1.4  2000/09/29 08:42:34  samsonov
// Added camera speed support
//
// Revision 1.3  2000/09/28 05:57:56  samsonov
// minor fix
//
// Revision 1.2  2000/08/09 07:15:55  samsonov
// final version and Cocoon comments
//
// Revision 1.1  2000/07/19 06:40:05  samsonov
// Path datatype moved form DaveW
//
// Revision 1.2  2000/07/14 23:36:32  samsonov
// Rewriting to be used as Camera Path datatype in EditPath module
//
// Revision 1.1  1999/12/02 21:57:29  dmw
// new camera path datatypes and modules
//
//

#endif



