#ifndef VECTORPARTICLES_H
#define VECTORPARTICLES_H

#include <Packages/Uintah/Core/Datatypes/PSet.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
//#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/ShareAssignParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  using std::vector;
  using namespace SCIRun;

/**************************************

CLASS
   VectorParticles
   
   Simple VectorParticles Class.

GENERAL INFORMATION

   VectorParticles.h

   Packages/Kurt Zimmerman
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
  VectorParticles(const vector<ShareAssignParticleVariable<Vector> >& vectors,
		  PSet* pset);

  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~VectorParticles();
 
  // GROUP: Access
  //////////
  // return the Vectors
  vector<ShareAssignParticleVariable<Vector> >& get(){ return vectors; }
  PSet* getParticleSet(){ return psetH.get_rep(); }

  // GROUP: Modify
  //////////  
  // Set the Particle Set Handle
  void Set(PSetHandle psh){ psetH = psh;}
  //////////  
  // Set the Vectors
  void Set(vector<ShareAssignParticleVariable<Vector> >& s){ vectors = s; }

  void AddVar( const ParticleVariable<Vector>& parts );


  void SetName( string vname ) { _varname = vname; }
  void SetMaterial( int index) { _matIndex = index; }
	       

  // Persistant representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  void get_minmax(double& v0, double& v1);
  void get_bounds(Point& p0, Point& p1){ psetH->get_bounds(p0,p1);}

protected:
  bool have_minmax;
  double data_min;
  double data_max;
  void compute_minmax();

private:
  PSetHandle psetH;
  string _varname;
  int _matIndex;
  vector<ShareAssignParticleVariable<Vector> >  vectors;
};

} // End namespace Uintah

#endif
