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

#include <Core/Datatypes/PropertyManager.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Handle.h>

#include <Core/Math/LinearPWI.h>
#include <Core/Math/CubicPWI.h>

#include <Core/Geom/View.h>
#include <Core/Geometry/Quaternion.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geom/share.h>
#include <cstdio>

namespace SCIRun {

class Path;

typedef LockingHandle<Path> PathHandle;

class SCISHARE Path : public PropertyManager
{
public:
   
  enum Path_E { KEYFRAMED=0, LINEAR, CUBIC };
  enum Acc_E  { NO_ACCEL=0,  SMOOTH, USERMODE };

  void   reset();
  void   del_keyF(int);
  bool   ins_keyF(int, const View&, double speed = 1, Acc_E acc_patt = NO_ACCEL);
  bool   add_keyF(const View&, double speed = 1, Acc_E acc_patt = NO_ACCEL);

  Acc_E  get_acc_t() const;
  bool   set_acc_t( Acc_E value );

  double get_speed_val( int keyFrame ) const;
  bool   set_speed_val( int keyFrame, double speed ); // returns false if keyFrame does not exist.

  bool get_keyF(int, View&, double&);
  int  get_num_views() const;

  // path points manipulation
  bool   set_step(double);
  double get_step() const;
       
  bool     set_path_t( Path_E t );
  Path_E   get_path_t() const;
  
  bool build_path();
    
  inline bool is_looped() const;
  inline bool is_reversed() const;
  inline bool is_started() const;
    
  bool set_loop(bool);
  bool set_back(bool);
  void seek_start();
  void seek_end();
  void stop();

  // return next path view, the nearest view behind, current speed and acceleration
  bool get_nextPP( View & view, int & curr_view, double & curr_speed, double & curr_acc );
  bool get_nextKF( View & view, int & keyFramNumber, double & keyFrameSpeed, double & keyFrameAccPatt );
  
  Path();
  virtual Path* clone();
  virtual ~Path();

  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

private:
  double    pathP;                 // current value of distance from the start point
  int       keyF;                  // keyframe nearest to the current path point
  double    path_prm;              // current value of corresponding parameter
  // (changes from 0 to 1)

  bool      is_loop, is_built, is_reverse, is_acc, is_run;
  
  Path_E    path_t;
  Acc_E     acc_t;
  
  int       ns;                    // discretization of path
  double    step_size;             // average distance between neighbor path points
  
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
  Array1<double> sin_values;
  int count;
  double sin_series;
  
  int calc_view(double, View&);    // return number of current interval and hence number of current view
  bool resample_path();
  double get_delta();

  // syncronization between arc-length parameter and initial parametrization
  inline void set_param();
  inline void set_arc_param();

};

inline bool Path::is_looped() const{
  return is_loop;
}

inline bool Path::is_reversed() const{
  return is_reverse;
}

inline bool Path::is_started() const{
  return is_run;
}

inline void Path::set_param(){

  printf( "pathP:     %lf\n",pathP );
  printf( "path_dist: %lf\n", path_dist );
  printf( "ns:        %d\n", ns );

  int ind = (int)( (pathP/path_dist)*(ns-1) );

  printf( "ind: %d\n", ind );
      
  if( ind == ns-1 ) {
    path_prm = dist_prm[ind];
  }
  else {
    double delta = path_dist / (ns-1);
    double w     = (pathP-(ind*delta)) / delta;

    printf("del,w: %lf, %lf\n", delta, w);

    path_prm     = dist_prm[ind+1] * w + (1-w)*dist_prm[ind];
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

