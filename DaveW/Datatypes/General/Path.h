/*
 *  Path.h: Camera path
 *
 *  Written by:
 *   David Weinstein & Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   December 1999, July 2000
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_DaveW_Datatypes_Path_h
#define SCI_DaveW_Datatypes_Path_h 1

#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Math/LinearPWI.h>
#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Geom/View.h>


namespace DaveW {
namespace Datatypes {

using namespace SCICore::Math;
using SCICore::Containers::LockingHandle;
using SCICore::Containers::Array1;
using SCICore::GeomSpace::View;
using SCICore::Containers::clString;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

using namespace SCICore::Datatypes;

class Path;
typedef LockingHandle<Path> PathHandle;

enum PathType { linear, other };

class Path : public Datatype {
  enum { Default, Linear, Other };
  PathType  path_t;
  int       keyF;
  double    pathP;
  bool      is_loop;
  bool      is_built;
  bool      is_back;
  
  double    stepSize;

  View      default_view;

  Array1<float> param;

  LinearPWI eyeP_x, eyeP_y, eyeP_z, lookatP_x, lookatP_y, lookatP_z, upV_x, upV_y, upV_z, fov;
  bool      calc_view(double, View&);
  
public:
    // input data for path calculation
    Array1<View>   keyViews;
    Array1<double> speedVal;
    Array1<int>    accPatt;
   
    // resulting path points 
    Array1<double> pathPoints;
    
    void reset();

    // base view points manipulation
    void set_default_view(const View&);
    void del_keyF(int);
    void ins_keyF(int, const View&, double speed=Default, int acc_patt=Default);
    void add_keyF(const View&, double speed=1, int acc_patt=Default);
    int  get_acc_patt(int);
    double get_speed_val(int);

    bool get_keyF(int, View&);
    int  get_num_views() const;

    // path points manipulation
    void set_step(double);
    double get_step() const;
       
    void set_path_type(PathType t);
    PathType get_path_type() const;
    
    bool is_looped() const;
    bool is_backed() const;
    void set_loop(bool);
    void set_back(bool);
    
    // return next path view, the nearest view behind, current speed and acceleration
    bool get_nextPP(View&, int&, double&, double&);
    bool get_nextKF(View&, int&, double&, int&); 
    void seek_start();
    void seek_end();

    Path();
//    Path(const Path&);
    virtual Path* clone();
    virtual ~Path();

    bool      built_path();

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

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

#endif
