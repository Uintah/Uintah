#ifndef  UINTAH_DATATYPES_PSet_H
#define  UINTAH_DATATYPES_PSet_H

#include <Packages/Uintah/Core/Grid/ShareAssignParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Geometry/Point.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::vector;
using namespace SCIRun;
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
  PSet( const vector <ShareAssignParticleVariable<Point> >& positions,
	const vector <ShareAssignParticleVariable<long64> >& ids,
	const vector <const Patch *> patches,
	void* callbackClass);
  // GROUP: Destructors
  //////////
  // Destructor
  virtual  ~PSet();
  // GROUP: Access
  //////////
  // return the locations
  vector<ShareAssignParticleVariable<Point> >&  getPositions()
  { return positions;}
  //////////
  // return the ids
  vector<ShareAssignParticleVariable<long64> >&  getIDs()
  { return particle_ids;}
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
  void AddParticles( const ParticleVariable<Point>& locs,
		     const ParticleVariable<long64>& ids,
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

  vector< ShareAssignParticleVariable<Point> >  positions;
  vector< ShareAssignParticleVariable<long64> >  particle_ids;
  vector< const Patch* >  patches;
};

} // End namespace Uintah

#endif
