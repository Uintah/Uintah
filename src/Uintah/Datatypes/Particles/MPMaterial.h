#ifndef SCI_Datatypes_MPMaterial_h
#define SCI_Datatypes_MPMaterial_h 1



/*----------------------------------------------------------------------
CLASS
    MPMaterial

    A container class for data.

OVERVIEW TEXT
    MPMaterial contains scalar fields, vector fields and particle sets that
    from Mechanical Engineering simulations.


KEYWORDS


AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 12, 1999
----------------------------------------------------------------------*/

#include <map.h>

#include <SCICore/CoreDatatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/CoreDatatypes/ScalarField.h>
#include <SCICore/CoreDatatypes/VectorField.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/String.h>

#include <Uintah/Datatypes/Particles/ParticleSet.h>

namespace Uintah {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

  
class MPMaterial : public Datatype {
public:
  MPMaterial();
  virtual ~MPMaterial();
  MPMaterial(const MPMaterial&);
  virtual MPMaterial* clone() const;

  void AddVectorField(const clString& name, VectorFieldHandle vfh);
  void AddScalarField(const clString& name, ScalarFieldHandle sfh);
  void AddParticleSet(ParticleSetHandle psh);

  ParticleSetHandle getParticleSet();
  VectorFieldHandle getVectorField( clString name );
  ScalarFieldHandle getScalarField( clString name );
  
  void getScalarNames( Array1< clString>&);
  void getVectorNames( Array1< clString>&);
  

  ////////// Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;


private:

  struct ltstr
  {
    bool operator()(clString s1, clString s2) const
      {
	return (s1 < s2);
      }
  };

  map< clString, ScalarFieldHandle, ltstr > smap;
  map< clString, VectorFieldHandle, ltstr > vmap;
  ParticleSetHandle ps;

};

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.2  1999/08/17 06:40:06  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:58  mcq
// Initial commit
//
// Revision 1.1  1999/06/09 23:21:33  kuzimmer
// reformed the material/particle classes and removed the particleSetExtensions.  Now MPVizParticleSet inherits from cfdlibParticleSet-->use the new stl routines to dynamically check the particleSet type
//
// Revision 1.2  1999/04/27 23:18:39  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//

#endif
