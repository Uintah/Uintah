//static char *id="@(#) $Id$";

/*
 *  Path.cc: Camera path datatype implementation 
 *
 *  Written by:
 *   David Weinstein & Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   November 1999, July 2000

 *
 *  Copyright (C) 1999 SCI Group
 */

#include <DaveW/Datatypes/General/Path.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>

// #include <as_debug.h>

namespace DaveW {
namespace Datatypes {

using namespace SCICore::Geometry;
using namespace SCICore::GeomSpace;

static Persistent* make_Path()
{
    return scinew Path;
}

PersistentTypeID Path::type_id("Path", "Datatype", make_Path);

Path::Path()
{
  reset();
}

Path::~Path()
{
}

void Path::reset(){
  keyViews.remove_all();
  speedVal.remove_all();
  accPatt.remove_all();
  pathPoints.remove_all();
  default_view=View(Point(2, 0, 0), Point(0, 0, 0), Vector(0, 0, 1), 90);
  is_built=false;
  stepSize=0.001;
  path_t=linear;
  is_back=false;
  is_loop=false;
  keyF=0;
  pathP=0;
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
    
    is_built=false; // reducing size of persistent object (no interpolation data members)
    
    int version=stream.begin_class("Path", Path_VERSION);
    
    // hack-way; didn't find Pio function for bool type
    int p=int(path_t), l=int(is_loop), b=int(is_built), bck=int(is_back);

      Pio(stream, p);
      Pio(stream, l);
      Pio(stream, b);
      Pio(stream, bck);
      
      path_t=PathType(p);
      is_loop=bool(l);
      is_built=bool(b);
      is_back=bool(bck);
      
      Pio(stream, keyF); 
      Pio(stream, pathP);      
      Pio(stream, stepSize);
      Pio(stream, default_view);
      Pio(stream, param);
//       Pio(stream, eyeP_x);
//       Pio(stream, eyeP_y);
//       Pio(stream, eyeP_z);
//       Pio(stream, lookatP_x);
//       Pio(stream, lookatP_y);
//       Pio(stream, lookatP_z);
//       Pio(stream, upV_x);
//       Pio(stream, upV_y);
//       Pio(stream, upV_z);
//       Pio(stream, fov);
      Pio(stream, keyViews);
      Pio(stream, speedVal);
      Pio(stream, accPatt);
      Pio(stream, pathPoints);
//       MSG ("Inside Path::io()");
    stream.end_class();
}

void Path::set_default_view(const View& v){
  default_view=v;
}

void Path::del_keyF(int i){
  keyViews.remove(i);
  speedVal.remove(i);
  accPatt.remove(i);
  is_built=false;
}

void Path::ins_keyF(int i, const View& v, double speed, int acc_patt){
  keyViews.insert(i, v);
  speedVal.insert(i, speed);
  accPatt.insert(i, acc_patt);
  is_built=false;
}

void Path::add_keyF(const View& v, double speed, int acc_patt){
  keyViews.add(v);
  speedVal.add(speed);
  accPatt.add(acc_patt);
  is_built=false;
}
    
bool Path::get_keyF(int i, View& v){
  if (i>=0 && i<keyViews.size()){
    v=keyViews[i];
    return true;
  }
  else
    return false;
}

int  Path::get_num_views() const{
  return keyViews.size();
}

bool Path::is_looped() const{
  return is_loop;
}

bool Path::is_backed() const{
  return is_back;
}

bool Path::built_path(){
  if (!is_built){
    param.remove_all();

    double sum=0;
    param.add(sum);
    Vector dist;
    
    for (int i=1; i<keyViews.size(); i++) {
      dist=keyViews[i].eyep()-keyViews[i-1].eyep();
      sum+=dist.length(); 
      param.add(sum);
    }
    
 //    MSG ("Path::built_path - param[] before norm.");
//     MSG (param);
    
    // normalizing
    for (int i=0; i<param.size(); i++) {
      param[i]/=sum;
    }
    
 //    MSG ("Path::built_path - param[] after norm.");
//     MSG (param);
   
    try {
       Array1<double> buffx;
       Array1<double> buffy;
       Array1<double> buffz;
       Point p;
       Vector v;
     
       for (int i=0; i<keyViews.size(); i++) {
	 p=keyViews[i].eyep();
	 buffx.add(p.x());
	 buffy.add(p.y());
	 buffz.add(p.z());
       }

 //       MSG ("Path::built_path - eyeP_x array");
//        MSG (buffx);
//        MSG ("Path::built_path - eyeP_y array");
//        MSG (buffy);
//        MSG ("Path::built_path - eyeP_z array");
//        MSG (buffz);

      if (!eyeP_x.set_data(param, buffx) || ! eyeP_y.set_data(param, buffy) || !eyeP_z.set_data(param, buffz))
	throw 3;
      
      for (int i=0; i<keyViews.size(); i++) {
        p=keyViews[i].lookat();
        buffx[i]=p.x();
        buffy[i]=p.y();
        buffz[i]=p.z();
      } 

//        MSG ("Path::built_path - lookatP_x array");
//        MSG (buffx);
//        MSG ("Path::built_path - lookatP_y array");
//        MSG (buffy);
//        MSG ("Path::built_path - lookatP_z array");
//        MSG (buffz);

      if (!lookatP_x.set_data(param, buffx) || !lookatP_y.set_data(param, buffy) || !lookatP_z.set_data(param, buffz)){
	throw 3;
      }
      
      for (int i=0; i<keyViews.size(); i++) {
        v=keyViews[i].up();
	p=keyViews[i].eyep()+v;
        buffx[i]=p.x();
        buffy[i]=p.y();
        buffz[i]=p.z();
      }
//        MSG ("Path::built_path - upV_x array");
//        MSG (buffx);
//        MSG ("Path::built_path - upV_y array");
//        MSG (buffy);
//        MSG ("Path::built_path - upV_z array");
//        MSG (buffz);
      
       if (!upV_x.set_data(param, buffx) || !upV_y.set_data(param, buffy) || !upV_z.set_data(param, buffz)){
	 throw 3;
       }

      double temp;

      for (int i=0; i<keyViews.size(); i++) {
        temp=keyViews[i].fov();
        buffx[i]=temp;
      }

//       MSG ("Path::built_path - fov array");
//       MSG (buffx);

      if (!fov.set_data(param, buffx)){
	throw 3;
      }
      is_built=true;
    }
    catch (int) {
      param.remove_all();
      eyeP_x.reset();
      eyeP_y.reset();
      eyeP_z.reset();
      lookatP_x.reset();
      lookatP_y.reset();
      lookatP_z.reset();
      upV_x.reset();
      upV_y.reset();
      upV_z.reset();
      fov.reset();
      is_built=false;
      //      cout << "Wrong interpolation" << std::endl;
    }
  }

  return is_built;
}

bool Path::calc_view(double w, View& v){
  if (!is_built && !built_path() || w > 1 || w <0) {
    return false;
  }
  else {
    Point eyep(eyeP_x.get_value(w), eyeP_y.get_value(w), eyeP_z.get_value(w));
    Point lookatp(lookatP_x.get_value(w), lookatP_y.get_value(w), lookatP_z.get_value(w));
    Vector upV(upV_x.get_value(w)-eyep.x(), upV_y.get_value(w)-eyep.y(), upV_z.get_value(w)-eyep.z());
    double f=fov.get_value(w);
    v=View(eyep, lookatp, upV, f);
    return true;
  }
}  

void Path::set_step(double s){ 
  if (s>0) 
    stepSize=s;
  // is_built=false;
}
    
double Path::get_step() const { 
  return stepSize; 
}

void Path::set_path_type(PathType t){
  path_t=t;
  is_built=false;
}

PathType Path::get_path_type() const{
  return path_t;
}

bool Path::get_nextKF(View& v, int& kf_num, double& kf_speed, int& kf_acc_patt){
    keyF+=(is_back)?-1:1;

  if (keyF>=keyViews.size() || keyF<0  && !is_loop){
    kf_speed=0;            // need to be changed in future to get key frame speed/acc. pattern
    kf_acc_patt=0;
    kf_num=(is_back)?0:keyViews.size()-1;
    if (kf_num>=0)
      v=keyViews[kf_num];
    return false;
  }
  else
    if (is_loop){
      keyF=(is_back)?(keyViews.size()-1):0;
    }

  v=keyViews[keyF];
  kf_num=keyF;
  kf_speed=0;             // to be changed in full, "accelerated" version
  kf_acc_patt=0;
  return true;  
}

bool Path::get_nextPP(View& v, int& curr_view, double& curr_speed, double& curr_acc){
  pathP+=(is_back)?-stepSize:stepSize;
  
  bool is_valid=true;

  if (pathP>1 || pathP<0){
    pathP=(is_back)?1:0;
    if (is_loop){
      is_valid=true;
    }
    else {
      is_valid=false;
    }
  }
    if (is_valid && calc_view(pathP, v)){
      keyF=curr_view=eyeP_x.get_interval(pathP);
      curr_speed=1;                                   // to be fixed
      curr_acc=0;
      return true;
    }
    else {
      curr_speed=0;
      curr_acc=0;
      keyF=curr_view=(is_back)?0:keyViews.size()-1;
      if (curr_view>=0)
	v=keyViews[curr_view];
      return false;
    }
}


void Path::seek_start() {
  pathP=0;
  keyF=0;
}

void Path::seek_end() {
  pathP=1;
  keyF=keyViews.size()-1;
}

void Path::set_back(bool bck){
  is_back=bck;
}

void Path::set_loop(bool st){
  is_loop=st;    
}

int Path::get_acc_patt(int n){
  return (n<accPatt.size() && n>=0)? accPatt[n]:-1;
}

double Path::get_speed_val(int n){
  return (n<speedVal.size() && n>=0)? speedVal[n]:-1;
}


} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.2  2000/07/14 23:36:32  samsonov
// Rewriting to be used as Camera Path datatype in EditPath module
//
// Revision 1.1  1999/12/02 21:57:29  dmw
// new camera path datatypes and modules
//
//
