#ifndef SCALARPARTICLES_H
#define SCALARPARTICLES_H

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <Uintah/Interface/DataArchive.h>
//#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <iostream>
#include <vector>
using std::vector;

namespace Uintah {
namespace Datatypes {

using namespace Uintah;
using SCICore::Datatypes::Datatype;
using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;

/**************************************

CLASS
   ScalarParticles
   
   Simple ScalarParticles Class.

GENERAL INFORMATION

   ScalarParticles.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Texture

DESCRIPTION
   ScalarParticles class.
  
WARNING
  
****************************************/
class ScalarParticles;
typedef LockingHandle<ScalarParticles> ScalarParticlesHandle;

class ScalarParticles : public Datatype {

public:
  // GROUP: Constructors:
  //////////
  // Constructor
  ScalarParticles();
  //////////
  // Constructor
  ScalarParticles(const vector <ParticleVariable<Point> >& positions,
		 const vector <ParticleVariable<double> >& scalars,
		 void* callbackClass);

  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~ScalarParticles();
 
  // GROUP: Access
  //////////
  // return the Points
  vector<ParticleVariable<Point> >&  getPositions(){ return positions;}

  //////////
  // return the Scalars
  vector<ParticleVariable<double> >& get(){ return scalars; }

  //////////
  // return the callback
  void* getCallbackClass(){ return cbClass; }
  // GROUP: Modify
  //////////  
  // Set the Scalars
  void Set(vector <ParticleVariable<double> >& s){ scalars = s; }
  //////////
  // Set the particle Positions
  void SetPositions(vector <ParticleVariable<Point> >& p){ positions = p; }
  // Set the particle ids
  //////////
  // Set callback class
  void SetCallbackClass( void* cbc){ cbClass = cbc; }

  void AddVar( const ParticleVariable<Point> locs,
	       const ParticleVariable<double> parts,
	       const Patch* p);

  void SetGrid( GridP g ){ _grid = g; }
  void SetLevel( LevelP l){ _level = l; }
  void SetName( string vname ) { _varname = vname; }
  void SetMaterial( int index) { _matIndex = index; }
	       

  // Persistant representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  void get_minmax(double& v0, double& v1);
  void get_bounds(Point& p0, Point& p1);
protected:
  bool have_bounds;
  Point bmin;
  Point bmax;
  Vector diagonal;
  void compute_bounds();

  bool have_minmax;
  double data_min;
  double data_max;
  void compute_minmax();


private:


  void* cbClass;

  GridP _grid;
  LevelP _level;
  string _varname;
  int _matIndex;

  vector<ParticleVariable<Point> >  positions;
  vector<ParticleVariable<double> >  scalars;

};

} // end namespace Datatypes
} // end namespace Kurt
#endif
