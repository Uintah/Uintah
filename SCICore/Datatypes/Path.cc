//static char *id="@(#) $Id$";
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

#include <SCICore/Datatypes/Path.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Quaternion.h>
#include <SCICore/Util/Assert.h>
#include <SCICore/Math/MiscMath.h>
#include <math.h>
#include <iostream>

namespace SCICore {
namespace Datatypes {

#define PI 3.14159265358979323846
#define MSG(m) std::cout << m << std::endl; 

using namespace SCICore::Geometry;
using namespace SCICore::GeomSpace;

static Persistent* make_Path()
{
    return scinew Path;
}

PersistentTypeID Path::type_id("Path", "Datatype", make_Path);

Path::Path(): sint(30)
{
  upV=scinew Linear3DPWI<Vector>();
  fov=scinew LinearPWI();

  eyeP=NULL;
  lookatP=NULL;
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
}

void Path::reset(){
 
  keyViews.remove_all();
  speedVal.remove_all();
  accPatt.remove_all();
  is_built=is_sampled=is_back=is_loop=0;
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

#define Path_VERSION 2

void Path::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;
    using SCICore::Geometry::Pio;
    using SCICore::GeomSpace::Pio;
    
    int oldpath=path_t;

    stream.begin_class("Path", Path_VERSION);

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
    }
    stream.end_class();
}

void Path::del_keyF(int i){
  ASSERT(i>=0 && i<keyViews.size());
  keyViews.remove(i);
  speedVal.remove(i);
  accPatt.remove(i);
  is_built=is_sampled=0;
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
  
  is_built=is_sampled=0;
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
  is_built=is_sampled=0;
  return true;
}

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

int Path::get_acc_t(int n) const{
  return acc_t;
}

double Path::get_speed_val(int n) const {
  return (n<speedVal.size() && n>=0)? speedVal[n]:-1;
}
   
bool Path::get_keyF(int i, View& v){
  if (i>=0 && i<keyViews.size()){
    v=keyViews[i];
    keyF=i;
    return true;
  }
  return false;
}

bool Path::build_path(){
  if (!is_built){
    is_sampled=0;
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
    
    //std::cout << "Linear Path Length: " << sum << std::endl;
    //std::cout << "Parameters of the path: " << param << std::endl;
    
    Array1<Point> epts;
    Array1<Point> lpts;
    Array1<Vector> vects;
    Array1<double> fovs;

    for (int i=0; i<keyViews.size(); i++) {
      lpts.add(keyViews[i].lookat());
      epts.add(keyViews[i].eyep());
      vects.add(keyViews[i].up());
      fovs.add(keyViews[i].fov());
    }

    //***************************************************************
    // building quaternion representation of rotation
    rotFrames.remove_all();
    for (int i=0; i<keyViews.size(); i++) {
      Vector dir(keyViews[i].lookat()-keyViews[i].eyep());
      Quaternion tmp(dir, keyViews[i].up());
      rotFrames.add(tmp);
    }

    //std::cout << "Quats are built. Rotation frames:" <<  rotFrames << std::endl;

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
    
    //std::cout << "Parametrization is done. Total angle:" << sum  << std::endl;
    //std::cout << "parametrization array:" << std::endl << q_param  << std::endl;
 
    // testing quaternions:
    // test_quat();
    
    //*****************************************************************
    /*  //Attempt to optimize splines by supplying tangents to interpolation
        //objects correlated with tangent of space curve
    Array1<Vector> drvs;
    drvs.resize(keyViews.size());
    Vector k;
    double dx, dy, dz, d, delta;
    for (int i=0; i<drvs.size(); i++){
      k=Cross(Vector(keyViews[i].lookat()-keyViews[i].eyep()), keyViews[i].up());
      if (i==0 || i==drvs.size()-1){
	int fi=i, si;
	si=(i==0)?i+1:i-1;
	delta=param[si]-param[fi];
	dx=(epts[si].x()-epts[fi].x())/delta;
	dy=(epts[si].y()-epts[fi].y())/delta;
	dz=(epts[si].z()-epts[fi].z())/delta;
      }
      else {
	delta=param[i+1]-param[i-1];
	dx=(epts[i+1].x()-epts[i-1].x())/delta;
	dy=(epts[i+1].y()-epts[i-1].y())/delta;
	dz=(epts[i+1].z()-epts[i-1].z())/delta;
      }
      d=sqrt(dx*dx+dy*dy+dz*dz);
      k.normalize();
      drvs[i]=k*(-d);
    }
    */

    if (eyeP->set_data(param, epts) && lookatP->set_data(param, lpts) && upV->set_data(param, vects) 
	&& fov->set_data(param, fovs) && resample_path()){
	 is_built=1;
    }
    else {
      eyeP->reset();
      lookatP->reset();
      upV->reset();
      fov->reset();
      msg ("Path not built");
      is_built=is_sampled=0;
    }
  }
  return is_built;
}

bool Path::resample_path(){
  is_sampled=0;
  ns=1001;
  const double delta=1/(double)(ns-1);
  
  Array1<double> pathPoints;
  pathPoints.resize(ns);
  
  is_sampled=1;
  
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
  
  // std::cout << "Last w on the sampling: " << w << std::endl;
  // std::cout << "path length got from arc_length parametrization: " << path_dist << std::endl;
  
  dist_prm.remove_all();
  dist_prm.resize(ns);
  double s=0;
  int j=1;
  
  // getting arc-length parametrization
  dist_prm[0]=0;
  for (int i=1; i<ns; i++){
    s=delta*i*path_dist;
    while (pathPoints[j]<s) j++;
    w=(s-pathPoints[j-1])/(pathPoints[j]-pathPoints[j-1]);
    // if (!(i%50)) std::cout << "w[" << i << "]=" << w  << ", j[" << i << "]=" << j << ", s[" << i << "]=" << s << ", delta=" << delta << ", path_dist" << path_dist << std::endl;
    dist_prm[i]=(j*w+(1-w)*(j-1))*delta*paramd;    
  }

  //  for (int i=0; i< dist_prm.size(); i+=50)
  //  std::cout << "dist_rm[" << i << "]=" << dist_prm[i]  << std::endl;

  return is_sampled;
}

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
  MSG(" INTERPOLATED QUATERNION WITH qw=");
  MSG(qw);
  MSG(q);
  MSG("First quat:");
  MSG(rotFrames[0]);
  q.get_frame(dir, up, tang);
  MSG("Frame got:");
  MSG(dir);
  MSG(up);
  MSG(tang);
  v=View(ep, ep+dir, up, fv);
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
  if (t!=path_t || !eyeP || !lookatP){
    path_t=t;
    is_run=is_built=0;
    if (path_t!=KEYFRAMED){
      delete eyeP;                          // safe on NULL pointers
      delete lookatP;

      switch (path_t){
      case CUBIC:
	eyeP=scinew Cubic3DPWI<Point>();
	lookatP=scinew Cubic3DPWI<Point>();
	return true;
	
      case LINEAR:
	eyeP=scinew Linear3DPWI<Point>();
	lookatP=scinew Linear3DPWI<Point>();
	return true;
      default:
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
  kf_speed=0;             // to be changed in full, "accelerated" version
  kf_acc_patt=0;

  return is_valid;
}

// function providing acceleration patterns support
double Path::get_delta(){
  double step=step_size;

  if (acc_t==SMOOTH){
    double ps=step_size*60/PI;
    int d=1;
    if (ps>path_dist/2){  // ensuring the path point not in first and last acc zones at a time
      ps=path_dist/2;
      d=2*ps*PI/step_size;
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
      step=step_size*sint[count];
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
      
      // syncronzing pathP with actual parameter
      set_param();
   
      if (is_valid){
	keyF=curr_view=calc_view(path_prm, v);
	curr_speed=step;
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

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
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
