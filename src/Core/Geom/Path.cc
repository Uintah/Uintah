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
    Implementation of class Path

GENERAL INFORMATION

    Created by:
    Alexei Samsonov
    Department of Computer Science
    University of Utah
    July 2000
    
    Copyright (C) 2000 SCI Group

KEYWORDS
    Quaternion

POSSIBLE REVISIONS
    Adding additional interpolation modes (subject to experiment)
----------------------------------------------------------------------*/

#include <Core/Geom/Path.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Quaternion.h>
#include <Core/Util/Assert.h>
#include <Core/Math/MiscMath.h>
#include <cmath>
#include <cstdio>
#include <iostream>

namespace SCIRun {

static Persistent* make_Path()
{
    return scinew Path;
}

PersistentTypeID Path::type_id("Path", "Datatype", make_Path);

//-----------------------------------------------------------------------------
// Constructor/Destructor

#define NUM_SIN_VALUES 30

Path::Path() :
  sin_values( NUM_SIN_VALUES )
{
  upV     = scinew Cubic3DPWI<Vector>();
  lookatP = scinew Cubic3DPWI<Point>();
  fov     = scinew CubicPWI();
  speed   = scinew CubicPWI();
  eyeP    = NULL;
  
  reset();

  set_path_t( CUBIC );

  acc_t      = SMOOTH;
  sin_series = 0;

  // Lookup table for first quadrant of sin(x).
  for( int i=0; i < NUM_SIN_VALUES; i++ ) {
    sin_values[i] = sin((i+1)*M_PI/60);
  }
}

Path::~Path()
{
  delete eyeP;
  delete lookatP;
  delete upV;
  delete fov;
  delete speed;
}

//-----------------------------------------------------------------------------

void
Path::reset()
{
  keyViews.remove_all();
  speedVal.remove_all();
  accPatt.remove_all();
  is_built   = 0;
  is_reverse = 0;
  is_loop    = 0;
  step_size  = 0.01;
  keyF       = 0;
  pathP      = 0;
  path_dist  = 0;
  is_acc     = false;
  is_run     = false;
}

Path*
Path::clone()
{
  return scinew Path(*this);
}

//-----------------------------------------------------------------------------

#define Path_VERSION 4

void
Path::io(Piostream& stream)
{
    Path_E oldpath = path_t;

    int version=stream.begin_class("Path", Path_VERSION);

    if (version > 3)
    {
      PropertyManager::io(stream);
    }

    Pio(stream, is_loop);
    Pio(stream, is_reverse);
    
    Pio(stream, (int&)path_t);
    Pio(stream, (int&)acc_t);
    
    Pio(stream, step_size);
    
    Pio(stream, keyViews);
    Pio(stream, speedVal);
    Pio(stream, accPatt);

    Pio(stream, sin_values);
    Pio(stream, count);
    Pio(stream, sin_series);    

    if (stream.reading()){
      is_built       = false;
      count          = 100;
      Path_E newpath = path_t;
      path_t         = oldpath;
      set_path_t( newpath );  // to initialize interpolation classes to new type
      if (version<3){
	for (int i=0; i<speedVal.size();i++)
	  speedVal[i]=1;
      }
    }
    
    stream.end_class();
}

//-----------------------------------------------------------------------------

void
Path::del_keyF(int i)
{
  ASSERT(i>=0 && i<keyViews.size());
  keyViews.remove(i);
  speedVal.remove(i);
  accPatt.remove(i);
  is_built=0;
}

// helper function to see if points are the same in respect to numerical zero
bool
cmp_pts(const Point& p1, const Point& p2)
{
  return Abs(((Vector)(p1-p2)).length()) < 10e-6;
}

bool
Path::ins_keyF(int i, const View& v, double speed, Acc_E acc_patt)
{
  int sz=keyViews.size();

  if(sz>0 && i<sz){
    int l=(i==0)?i:(i-1), r=i;
   
    View tmp1=keyViews[l], tmp2=keyViews[r];
    if (cmp_pts(tmp1.eyep(), v.eyep()) || cmp_pts(tmp2.eyep(), v.eyep())){
      return false;;
    }
    keyViews.insert(i, v);
    speedVal.insert(i, speed);
    accPatt.insert(i, acc_patt);
  }
  else
    if (sz==0){
      return add_keyF(v, speed, acc_patt);
    }
    else {
      return false;
    }
  
  is_built=0;
  return true;
}

bool
Path::add_keyF( const View & view, double speed, Acc_E acc_patt )
{
  int sz = keyViews.size();
  if( sz > 0 ) {
    View tmp=keyViews[ sz-1 ];
    if (cmp_pts(tmp.eyep(), view.eyep())){
      // the same point
      return false;
    }
  }
  keyViews.add( view );
  speedVal.add( speed );
  accPatt.add( acc_patt );
  is_built = 0;
  return true;
}

//-----------------------------------------------------------------------------

int
Path::get_num_views() const
{
  return keyViews.size();
}

void
Path::seek_start()
{
  pathP=0;
  keyF=0;
  is_acc = false;
  is_run = false;
}

void
Path::seek_end()
{
  pathP  = path_dist;
  keyF   = keyViews.size()-1;
  is_acc = false;
  is_run = false;
}

void
Path::stop()
{
  is_acc = false;
  is_run = false;
}

bool
Path::set_back(bool bck)
{
  if (is_reverse ^ bck){
    is_reverse = bck;
    return true;
  }
  else {
    return false;
  }
}

bool
Path::set_loop(bool st)
{
  if (is_loop ^ st){
    is_loop=st;
    return true;
  }
  else {
    return false;
  }   
}

bool
Path::set_acc_t( Acc_E value )
{
  if( acc_t != value ) {
    acc_t = value;
    return true;
  }
  return false;
}

Path::Acc_E
Path::get_acc_t() const
{
  return acc_t;
}


//-----------------------------------------------------------------------------

double
Path::get_speed_val( int keyFrame ) const
{
  return( keyFrame < speedVal.size() && keyFrame >= 0 ) ? speedVal[ keyFrame ] : -1;
}

bool
Path::set_speed_val( int keyFrame, double speed )
{
  if( keyFrame < speedVal.size() && keyFrame >= 0 ) {
    speedVal[keyFrame] = speed;
    return true;
  }
  else {
    return false;
  }
}
   
bool
Path::get_keyF(int i, View& v, double& sp)
{
  if (i>=0 && i<keyViews.size()){
    v    = keyViews[i];
    keyF = i;
    sp   = speedVal[i];
    return true;
  }
  return false;
}

//-----------------------------------------------------------------------------

bool
Path::build_path()
{
  if( !is_built ) {

    if( path_t == KEYFRAMED ) {
      printf("keyframed\n");
      is_built = 1;
      return true;
    }

    param.remove_all();
   
    double sum=0;
    param.add(sum);
    Vector dist;
    
    for (int i=1; i<keyViews.size(); i++) {
      dist=keyViews[i].eyep()-keyViews[i-1].eyep();
      sum+=sqrt(dist.length());                      // parametrization is still under question  
      //sum+=dist.length(); 
      param.add(sum);
    }
    
    Array1<Point> epts;
    Array1<Point> lpts;
    Array1<Vector> vects;
    Array1<double> fovs;
    Array1<double> sps;

    for (int i=0; i<keyViews.size(); i++) {
      lpts.add(keyViews[i].lookat());
      epts.add(keyViews[i].eyep());
      vects.add(keyViews[i].up());
      fovs.add(keyViews[i].fov());
      sps.add(speedVal[i]);
    }

    //***************************************************************
    // building quaternion representation of rotation
    rotFrames.remove_all();
    for (int i=0; i<keyViews.size(); i++) {
      Vector dir(keyViews[i].lookat()-keyViews[i].eyep());
      Quaternion tmp(dir, keyViews[i].up());
      rotFrames.add(tmp);
    }

    // rotational parametrization by angles; no normalization
    sum=0;
    q_param.remove_all();
    q_param.add(sum);
    double tmp=0;
    for (int i=1; i<rotFrames.size(); i++){
      tmp=Dot(rotFrames[i], rotFrames[i-1]);    // unit quaternions !!!
      sum+=(tmp>0)?tmp:-tmp;						
      q_param.add(sum);
    }
    
    if (path_t == CUBIC && param.size() < 2 )
    {
      is_built = 0;
      return false;
    }
    
    if (eyeP->set_data(param, epts) && lookatP->set_data(param, lpts) && upV->set_data(param, vects) 
	&& fov->set_data(param, fovs) && speed->set_data(param, sps) && resample_path()){
	 is_built=1;
    }
    else {
      eyeP->reset();
      lookatP->reset();
      upV->reset();
      fov->reset();
      speed->reset();
      //msg ("Path not built");
      is_built=0;
    }
  }
  return is_built;
}

bool Path::resample_path(){

  ns=1001;
  const double delta=1/(double)(ns-1);
  
  Array1<double> pathPoints;
  pathPoints.resize(ns);
  
  Vector dist;
  double w=0;
  Point tmp1, tmp2;
  eyeP->get_value(w, tmp1);
                 
  // sampling arc_length
  pathPoints[0]=0;
  double paramd=param[param.size()-1];
  for (int i=1; i<ns; i++){    
    w=paramd*((double)i)/(ns-1);
    eyeP->get_value(w, tmp2);
    dist=tmp2-tmp1;
    pathPoints[i]=dist.length()+pathPoints[i-1];
    tmp1=tmp2;
  }
  
  path_dist = pathPoints[ns-1];
  
  dist_prm.remove_all();
  dist_prm.resize(ns);
  double s=0;
  int j=1;
  
  // getting arc-length parametrization
  dist_prm[0]=0;
  for (int i=1; i<ns; i++){
    s=delta*i*path_dist;
    while ((s-pathPoints[j])>10e-13) j++;
    w=(s-pathPoints[j-1])/(pathPoints[j]-pathPoints[j-1]);
    dist_prm[i]=(j*w+(1-w)*(j-1))*delta*paramd;    
  }

  return 1;
}

//-----------------------------------------------------------------------------

// return keyframe number right behind the view
int
Path::calc_view( double w, View & view )
{
#undef QUATERNIONS

  int p = eyeP->get_interval( w );
  
  ASSERT( p >= 0 && p < q_param.size()-1 );

  Point ep;
  double fv;
  if( !eyeP->get_value(w, ep) || !fov->get_value(w, fv) ) {
    printf( "Warning, get_value failed for eyeP or fov...\n" );
  }

#ifdef QUATERNIONS 

  Vector dir, up, tang;
  // mapping w to quaternion parameter
  double qw=(w-param[p])/(param[p+1]-param[p]);

  Quaternion q=Slerp(rotFrames[p], rotFrames[p+1], qw);
#endif

#ifndef QUATERNIONS
  Point lp;
  Vector uv;
 
  lookatP->get_value(w, lp);
  upV->get_value(w, uv);
  
  view = View(ep, lp, uv,  fv);
#endif

  return p;
}  

//-----------------------------------------------------------------------------

bool
Path::set_step( double newStep )
{
  if( newStep != step_size ) {
    step_size = newStep;
    return true;
  }
  else {
    return false;
  }
}
    
double
Path::get_step() const
{
  return step_size;
}

bool
Path::set_path_t( Path_E t )
{
  if( t != path_t || !eyeP ) {

    path_t = t;
    is_run   = false;
    is_built = false;

    if (path_t != KEYFRAMED) {
      delete eyeP;                          // safe on NULL pointers
      //      delete lookatP;

      switch (path_t){
      case CUBIC:
	eyeP=scinew Cubic3DPWI<Point>();
	//lookatP=scinew Cubic3DPWI<Point>();
	return true;
	
      case LINEAR:
	eyeP=scinew Linear3DPWI<Point>();
	//lookatP=scinew Linear3DPWI<Point>();
	return true;
      default:
	break;
      }
    }
  }
  return false;
}

Path::Path_E
Path::get_path_t() const
{
  return path_t;
}

bool
Path::get_nextKF(View& v, int& kf_num, double& kf_speed, double& kf_acc_patt)
{
  keyF+=(is_reverse)?-1:1;
  bool is_valid=true;
  int sz=keyViews.size();
  if (sz==0)
    return false;
  if (keyF>=sz || keyF<0){
    keyF=(is_reverse)?(sz-1):0;
    if (is_loop){
      is_valid=true;
    }
    else {
      is_valid=false;
    }
  }

  v=keyViews[keyF];
  kf_num=keyF;
  kf_speed=speedVal[keyF];
  kf_acc_patt=0;

  return is_valid;
}

//-----------------------------------------------------------------------------

// function providing acceleration patterns support
double
Path::get_delta()
{
  double step;
  speed->get_value( path_prm, step );

  double s = step_size * path_dist;

  step = step * s;

  switch (acc_t){
  case SMOOTH: {
    double ps=s*60/M_PI;
    int d=1;
    if (ps>path_dist/2){  // ensuring the path point not in first and last acc zones at a time
      ps=path_dist/2;
      d=(int)(2*ps*M_PI/s);
    }
    
    bool is_fzone=(is_reverse)?(pathP>(path_dist-ps)):(pathP<ps);
    bool is_lzone=(!is_reverse)?(pathP>(path_dist-ps)):(pathP<ps);	     
    is_fzone=((!is_run || is_acc)&& is_fzone)?true:false;
    is_lzone=(!is_loop && is_lzone)?true:false;
      
    // trigerring acceleration
    if (!is_acc && (is_fzone || is_lzone)){
      count=(is_fzone)?0:29;
      is_acc=1;
    }
    
    if( count>=0 && count<30 ) {
      step = s * sin_values[count];
      if ((pathP>ps && !is_reverse) || (is_reverse && pathP<ps)) 
	count-=d;
      else
	count+=d;
    }
    
    if(!(is_fzone || is_lzone)){
      is_acc=0;
      count=100;
    }
  }
  break;
  
  case USERMODE:
    break;
    
  case NO_ACCEL:
  default:
    step=s;
  }
  
  is_run = true;
  return step;
}

bool
Path::get_nextPP(View& view, int& curr_view, double& curr_speed, double& curr_acc)
{
  if (!is_built && !build_path()) {
    return false;
  }
  else {
    if( path_t == KEYFRAMED ) {
      printf("keyframed\n");
      is_run = 1;
      return get_nextKF(view, curr_view, curr_speed, curr_acc);
    }
    else {
      if( !is_run ){
        printf("!is_run\n");
	// setting path point to the current key frame
	path_prm = param[keyF];
	set_arc_param();
	is_acc = 0;
      }

      double step = get_delta();
      
      pathP += (is_reverse)?-step:step;
      bool is_valid = true;
      
      if( pathP > path_dist || pathP < 0 ) {
	pathP = (is_reverse)?(pathP+path_dist):(pathP-path_dist);
	if( !is_loop ) {
	  is_valid = false;
	}
      }

      printf("step, path: %lf, %lf,    %d\n", step, pathP, is_valid);

      if( is_valid ) {

	// syncronzing pathP with actual parameter

        printf("was prm: %lf\n", path_prm);
	set_param();
        printf("is  PRM: %lf\n", path_prm);

	keyF = calc_view( path_prm, view );

        printf("keyF:  %d\n",keyF);

        curr_view = keyF;

	curr_speed = step / (step_size*path_dist);
	curr_acc = path_prm;      // not used
	return true;
      }
      else {
        printf("not valid\n");
	curr_speed=0;
	curr_acc=0;
        curr_view = (is_reverse) ? 0 : keyViews.size()-1;
	keyF = curr_view;
	count=0;
	if (keyViews.size()) {
	  view = keyViews[curr_view];
        }
	return false;
      }
    }
  }
}

} // End namespace SCIRun

