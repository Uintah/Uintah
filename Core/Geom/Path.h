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

#ifndef SCI_Core_Datatypes_Path_h
#define SCI_Core_Datatypes_Path_h 1

#include <Core/share/share.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Handle.h>

#include <Core/Math/LinearPWI.h>
#include <Core/Math/CubicPWI.h>

#include <Core/Geom/View.h>
#include <Core/Geometry/Quaternion.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {



class Path;
typedef LockingHandle<Path> PathHandle;

//typedef Handle<PiecewiseInterp<Point> > PointPWI;
//typedef Handle<PiecewiseInterp<Vector> > VectorPWI;
//typedef Handle<PiecewiseInterp<double> > SimplePWI;

enum {KEYFRAMED=0, LINEAR, CUBIC};
enum {NO_ACCEL=0, SMOOTH, USERMODE};

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
    bool   ins_keyF(int, const View&, double speed=1, int acc_patt=NO_ACCEL);
    bool   add_keyF(const View&, double speed=1, int acc_patt=NO_ACCEL);
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
  USE_IF_ASSERTS_ON(int sz=dist_prm.size();)
  ASSERT(sz>=ns && ns > 0 && path_prm<=dist_prm[sz-1]);
  int j=1;
  while ((path_prm-dist_prm[j])>10e-13) j++;
  
  double  w=(path_prm-dist_prm[j-1])/(dist_prm[j]-dist_prm[j-1]);
  double delta=path_dist/(ns-1);
  pathP=(j*w+(1-w)*(j-1))*delta;    
}


} // End namespace SCIRun

#endif



