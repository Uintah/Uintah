/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#ifndef  UINTAH_DATATYPES_PSet_H
#define  UINTAH_DATATYPES_PSet_H

#include <Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>

#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Geometry/Point.h>

#include <vector>

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
  vector<ShareAssignParticleVariable<Point> >&  get_positions()
  { return positions;}
  //////////
  // return the ids
  vector<ShareAssignParticleVariable<long64> >&  get_ids()
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
