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
#include <math.h>
#include <iostream>

namespace SCIRun {

#define PI 3.14159265358979323846
#define MSG(m) std::cout << m << std::endl; 


static Persistent* make_Path()
{
    return scinew Path;
}

PersistentTypeID Path::type_id("Path", "Datatype", make_Path);

//-----------------------------------------------------------------------------
// Constructor/Destructor

Path::Path(): sint(30)
{
  upV=scinew Cubic3DPWI<Vector>();
  lookatP=scinew Cubic3DPWI<Point>();
  fov=scinew CubicPWI();
  speed=scinew CubicPWI();
  eyeP=NULL;
  
  reset();

  set_path_t(CUBIC);
  acc_t=SMOOTH;
  sin_series=0;

  // lookup table for first quadrant of sin(x)
  for (int i=0; i<30; i++)
    sint[i]=sin((i+1)*PI/60);
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

void Path::reset(){
 
  keyViews.remove_all();
  speedVal.remove_all();
  accPatt.remove_all();
  is_built=is_back=is_loop=0;
  step_size=0.01;
  keyF=0;
  pathP=0;
  path_dist=0;
  is_acc=is_run=false;
}

Path* Path::clone()
{
    return scinew Path(*this);
}

//-----------------------------------------------------------------------------

#define Path_VERSION 3

void Path::io(Piostream& stream)
{
    
    int oldpath=path_t;

    int version=stream.begin_class("Path", Path_VERSION);

    Pio(stream, is_loop);
    Pio(stream, is_back);
    
    Pio(stream, path_t);
    Pio(stream, acc_t);
    
    Pio(stream, step_size);
    
    Pio(stream, keyViews);
    Pio(stream, speedVal);
    Pio(stream, accPatt);

    Pio(stream, sint);
    Pio(stream, count);
    Pio(stream, sin_series);    

    if (stream.reading()){
      is_built=false;
      count=100;
      int newpath=path_t;
      path_t=oldpath;
      set_path_t(newpath);  // to initialize interpolation classes to new type
      if (version<3){
	for (int i=0; i<speedVal.size();i++)
	  speedVal[i]=1;
      }
    }
    
    stream.end_class();
}

//-----------------------------------------------------------------------------

void Path::del_keyF(int i){
  ASSERT(i>=0 && i<keyViews.size());
  keyViews.remove(i);
  speedVal.remove(i);
  accPatt.remove(i);
  is_built=0;
}

// helper function to see if points are the same in respect to numerical zero
bool cmp_pts(const Point& p1, const Point& p2){
  return Abs(((Vector)(p1-p2)).length())<10e-6;
}

bool Path::ins_keyF(int i, const View& v, double speed, int acc_patt){
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

bool Path::add_keyF(const View& v, double speed, int acc_patt){
  int sz=keyViews.size();
  if (sz>0){
    View tmp=keyViews[sz-1];
    if (cmp_pts(tmp.eyep(), v.eyep())){
      // the same point
      return false;
    }
  }
  keyViews.add(v);
  speedVal.add(speed);
  accPatt.add(acc_patt);
  is_built=0;
  return true;
}

//-----------------------------------------------------------------------------

int  Path::get_num_views() const{
  return keyViews.size();
}

void Path::seek_start() {
  pathP=0;
  keyF=0;
  is_acc=is_run=0;
}

void Path::seek_end() {
  pathP=path_dist;
  keyF=keyViews.size()-1;
  is_acc=is_run=0;
}

void Path::stop(){
  is_acc=is_run=0;
}

bool Path::set_back(bool bck){
  if (is_back ^ bck){
    is_back=bck;
    return true;
  }
  else {
    return false;
  }
}

bool Path::set_loop(bool st){
  if (is_loop ^ st){
    is_loop=st;
    return true;
  }
  else {
    return false;
  }   
}
bool Path::set_acc_t(int t){
  if (acc_t!=t){
    acc_t=t;
    return true;
  }
  return false;
}

int Path::get_acc_t(int) const{
  return acc_t;
}


//-----------------------------------------------------------------------------

double Path::get_speed_val(int n) const {
  return (n<speedVal.size() && n>=0)? speedVal[n]:-1;
}
   
bool Path::get_keyF(int i, View& v, double& sp){
  if (i>=0 && i<keyViews.size()){
    v=keyViews[i];
    keyF=i;
    sp=speedVal[i];
    return true;
  }
  return false;
}

//-----------------------------------------------------------------------------

bool Path::build_path(){
  if (!is_built){

    if(path_t==KEYFRAMED)
    {
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
    
    if (path_t==CUBIC && param.size()<2)
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
      msg ("Path not built");
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
  
  path_dist=pathPoints[ns-1];
  
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
int Path::calc_view(double w, View& v){

#undef QUATERNIONS

  int p=eyeP->get_interval(w);
  
  ASSERT(p>=0 && p<q_param.size()-1);

  Point ep;
  double fv;
  eyeP->get_value(w, ep);
  fov->get_value(w, fv);

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
  
  v=View(ep, lp, uv,  fv);
#endif

  return p;
}  

//-----------------------------------------------------------------------------

bool Path::set_step(double s){ 
  if (s!=step_size) {
    step_size=s;
    return true;
  }
  else {
    return false;
  }
}
    
double Path::get_step() const { 
  return step_size;
}

bool Path::set_path_t(int t){
  if (t!=path_t || !eyeP ){
    path_t=t;
    is_run=is_built=0;
    if (path_t!=KEYFRAMED){
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

int Path::get_path_t() const{
  return path_t;
}

bool Path::get_nextKF(View& v, int& kf_num, double& kf_speed, double& kf_acc_patt){
  keyF+=(is_back)?-1:1;
  bool is_valid=true;
  int sz=keyViews.size();
  if (sz==0)
    return false;
  if (keyF>=sz || keyF<0){
    keyF=(is_back)?(sz-1):0;
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
double Path::get_delta(){
 
  double step;
  speed->get_value(path_prm, step);
  double s=step_size*path_dist;
  step=step*s;

  switch (acc_t){
  case SMOOTH: {
    double ps=s*60/PI;
    int d=1;
    if (ps>path_dist/2){  // ensuring the path point not in first and last acc zones at a time
      ps=path_dist/2;
      d=(int)(2*ps*PI/s);
    }
    
    bool is_fzone=(is_back)?(pathP>(path_dist-ps)):(pathP<ps);
    bool is_lzone=(!is_back)?(pathP>(path_dist-ps)):(pathP<ps);	     
    is_fzone=((!is_run || is_acc)&& is_fzone)?true:false;
    is_lzone=(!is_loop && is_lzone)?true:false;
      
    // trigerring acceleration
    if (!is_acc && (is_fzone || is_lzone)){
      count=(is_fzone)?0:29;
      is_acc=1;
    }
    
    if (count>=0 && count<30){
      step=s*sint[count];
      if ((pathP>ps && !is_back) || (is_back && pathP<ps)) 
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
  
  is_run=1;
  return step;
}

bool Path::get_nextPP(View& v, int& curr_view, double& curr_speed, double& curr_acc){
  if (!is_built && !build_path()){
    return false;
  }
  else {
    if (path_t==KEYFRAMED){
      is_run=1;
      return get_nextKF(v, curr_view, curr_speed, curr_acc);
    }
    else {
      if (!is_run){
	// setting path point to the current key frame
	path_prm=param[keyF];
	set_arc_param();
	is_acc=0;
      }

      double step=get_delta();
      
      pathP+=(is_back)?-step:step;
      bool is_valid=true;
      
      if (pathP>path_dist || pathP<0){
	pathP=(is_back)?(pathP+path_dist):(pathP-path_dist);
	if (is_loop){
	  is_valid=true;
	}
	else {
	  is_valid=false;
	}
      }
      
      if (is_valid){
	// syncronzing pathP with actual parameter
	set_param();
	keyF=curr_view=calc_view(path_prm, v);
	curr_speed=step/(step_size*path_dist);
	curr_acc=path_prm;      // not used
	return true;
      }
      else {
	curr_speed=0;
	curr_acc=0;
	keyF=curr_view=(is_back)?0:keyViews.size()-1;
	count=0;
	if (keyViews.size())
	  v=keyViews[curr_view];
	return false;
      }
    }
  }
}

} // End namespace SCIRun

