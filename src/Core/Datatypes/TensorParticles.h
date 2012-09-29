/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef TENSORPARTICLES_H
#define TENSORPARTICLES_H

#include <Core/Math/Matrix3.h>
#include <Core/Datatypes/PSet.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include   <vector>

namespace Uintah {
using std::vector;
using namespace SCIRun;

/**************************************

CLASS
   TensorParticles
   
   Simple TensorParticles Class.

GENERAL INFORMATION

   TensorParticles.h

   Packages/Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Texture

DESCRIPTION
   TensorParticles class.
  
WARNING
  
****************************************/
class TensorParticles;
typedef LockingHandle<TensorParticles> TensorParticlesHandle;

class TensorParticles : public Datatype {

public:
  // GROUP: Constructors:
  //////////
  // Constructor
  TensorParticles();
  //////////
  // Constructor
  TensorParticles(const vector<ShareAssignParticleVariable<Matrix3> >& tensors,
		  PSet* pset);

  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~TensorParticles();
 
  // GROUP: Access
  //////////
  // return the Tensors
  vector<ShareAssignParticleVariable<Matrix3> >& get(){ return tensors; }
  PSet* getParticleSet(){ return psetH.get_rep(); }

  // GROUP: Modify
  //////////  
  // Set the Particle Set Handle
  void Set(PSetHandle psh){ psetH = psh;}
  //////////  
  // Set the Tensors
  void Set(vector<ShareAssignParticleVariable<Matrix3> >& s){ tensors = s; }

  void AddVar( const ParticleVariable<Matrix3>& parts );


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
  LevelP level;
  string _varname;
  int _matIndex;
  vector<ShareAssignParticleVariable<Matrix3> >  tensors;
};

} // End namespace Uintah

#endif
