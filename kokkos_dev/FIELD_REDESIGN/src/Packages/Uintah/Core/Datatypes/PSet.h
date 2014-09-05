#ifndef  UINTAH_DATATYPES_PSet_H
#define  UINTAH_DATATYPES_PSet_H

#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Datatypes/Datatype.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Geometry/Point.h>
#include <iostream>
#include <vector>
using std::vector;

namespace Uintah {
namespace Datatypes{

using SCICore::Containers::LockingHandle;
using SCICore::Datatypes::Datatype;
using SCICore::Geometry::Point;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
/**************************************

CLASS
   PSet
   
   This class is a container for particle ids and positions. 

GENERAL INFORMATION

   PSet.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   PSet, Level

DESCRIPTION
   This class is simply a container for particle ids and positions
   and can be shared by ScalarParticles, VectorParticles and Tensor
   particle classes.

  
WARNING
  
****************************************/
class PSet;
typedef LockingHandle<PSet> PSetHandle;

class PSet : public Datatype {
 public:

  // GROUP: Constructors:
  //////////
  // Constructor
  PSet();
  //////////
  // Constructor
  PSet( const vector <ParticleVariable<Point> >& positions,
	const vector <ParticleVariable<long> >& ids,
	const vector <const Patch *> patches,
	void* callbackClass);
  // GROUP: Destructors
  //////////
  // Destructor
  virtual  ~PSet();
  // GROUP: Access
  //////////
  // return the locations
  vector<ParticleVariable<Point> >&  getPositions(){ return positions;}
  //////////
  // return the ids
  vector<ParticleVariable<long> >&  getIDs(){ return particle_ids;}
  //////////
  // return the patches
  vector<const Patch*>&  getPatches(){ return patches; }
  //////////
  // return the grid level
  LevelP getLevel(){ return _level; }
  //////////
  // return the grid level
  void get_bounds(Point& p0, Point& p1);
  /////////
  // return the call back class
  void* getCallBackClass(){ return cbClass; }

  // GROUP: Modify
  //////////  
  // add a particle
  void AddParticles( const ParticleVariable<Point> locs,
		     const ParticleVariable<long> ids,
		     const Patch* p);
  //////////  
  // associate a grid
  void SetGrid( GridP g ){ _grid = g; }
  //////////
  // set the grid level
  void SetLevel( LevelP l){ _level = l; }
  //////////
  // Set callback class
  void SetCallbackClass( void* cbc){ cbClass = cbc; }
 
  // Persistant representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

 protected:
  bool have_bounds;
  Point bmin;
  Point bmax;
  Vector diagonal;
  void compute_bounds();
  
 private:

  void* cbClass;

  GridP _grid;
  LevelP _level;

  vector< ParticleVariable<Point> >  positions;
  vector< ParticleVariable<long> >  particle_ids;
  vector< const Patch* >  patches;
};
} // end namespace Datatypes
} // end namespace Uintah

  
#endif
