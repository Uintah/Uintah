#ifndef VECTORPARTICLES_H
#define VECTORPARTICLES_H

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <Uintah/Interface/DataArchive.h>
//#include <Uintah/Components/MPM/Util/Matrix3D.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <iostream>

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
   VectorParticles
   
   Simple VectorParticles Class.

GENERAL INFORMATION

   VectorParticles.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Texture

DESCRIPTION
   VectorParticles class.
  
WARNING
  
****************************************/
class VectorParticles;
typedef LockingHandle<VectorParticles> VectorParticlesHandle;

class VectorParticles : public Datatype {

public:
  // GROUP: Constructors:
  //////////
  // Constructor
  VectorParticles();
  //////////
  // Constructor
  VectorParticles(const vector <ParticleVariable<Point> >& positions,
		 const vector <ParticleVariable<Vector> >& scalars,
		 void* callbackClass);

  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~VectorParticles();
 
  // GROUP: Access
  //////////
  // return the Points
   vector<ParticleVariable<Point> >& getPositions(){ return positions;}
  //////////
  // return the Vectors
  vector<ParticleVariable<Vector> >& get(){ return vectors; }
  //////////
  // return the callback
  void* getCallbackClass(){ return cbClass; }
  // GROUP: Modify
  //////////  
  // Set the Vectors
  void Set(const vector<ParticleVariable<Vector> >& s){ vectors = s; }
  //////////
  // Set the particle Positions
  void SetPositions(const vector<ParticleVariable<Point> >& p){ positions = p;}
  //////////
  // Set callback class
  void SetCallbackClass( void* cbc){ cbClass = cbc; }

  void AddVar( const ParticleVariable<Point> locs,
	       const ParticleVariable<Vector> parts,
	       const Patch* p);

  void SetGrid( GridP g ){ _grid = g; }
  void SetLevel( LevelP l){ _level = l; }
  void SetName( string vname ) { _varname = vname; }
  void SetMaterial( int index) { _matIndex = index; }
  // Persistant representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  // returns the min & max vectory length
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
  GridP  _grid;
  LevelP _level;
  string _varname;
  int _matIndex;

  vector<ParticleVariable<Point> > positions;
  vector<ParticleVariable<Vector> >vectors;
};

} // end namespace Datatypes
} // end namespace Kurt
#endif
