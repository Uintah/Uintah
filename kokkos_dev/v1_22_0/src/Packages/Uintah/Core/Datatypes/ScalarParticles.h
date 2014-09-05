#ifndef SCALARPARTICLES_H
#define SCALARPARTICLES_H

#include <Packages/Uintah/Core/Datatypes/PSet.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
//#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/ShareAssignParticleVariable.h>
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
   ScalarParticles
   
   Simple ScalarParticles Class.

GENERAL INFORMATION

   ScalarParticles.h

   Packages/Kurt Zimmerman
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
  ScalarParticles(const vector<ShareAssignParticleVariable<double> >& scalars,
		  PSet* pset);

  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~ScalarParticles();
 
  // GROUP: Access
  //////////
  // return the Scalars
  vector<ShareAssignParticleVariable<double> >& get(){ return scalars; }
  PSet* getParticleSet(){ return psetH.get_rep(); }

  // GROUP: Modify
  //////////  
  // Set the Particle Set Handle
  void Set(PSetHandle psh){ psetH = psh;}
  //////////  
  // Set the Scalars
  void Set(vector<ShareAssignParticleVariable<double> >& s){ scalars = s; }
  //////////
  // Are these particles from a new archive?

  void AddVar( const ParticleVariable<double>& parts );

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
  vector<ShareAssignParticleVariable<double> >  scalars;

};

} // End namespace Uintah

#endif
