#ifndef SCI_Datatypes_ParticleGridReader_h
#define SCI_Datatypes_ParticleGridReader_h 1

/*----------------------------------------------------------------------
CLASS
    ParticleGridReader

    A class for reading files containing both particle and gridded data.

OVERVIEW TEXT
    This is an abstract class.  Readers for particular file types should
    inherit from this class



KEYWORDS

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 6, 1999
----------------------------------------------------------------------*/

#include <SCICore/CoreDatatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Geometry/Vector.h>

#include <Uintah/Datatypes/Particles/MPMaterial.h>

namespace Uintah {
namespace Datatypes {

class ParticleGridReader;
typedef LockingHandle<ParticleGridReader> ParticleGridReaderHandle;

class ParticleGridReader : public Datatype {
public:
    ParticleGridReader();
    virtual ~ParticleGridReader();
    ParticleGridReader(const ParticleGridReader&);
    virtual ParticleGridReader* clone() const=0;

  //////////
  // SetFile expects a filename include full path
    virtual void SetFile(const clString& filename)=0;
  //////////
    virtual clString GetFile() = 0;
    virtual int GetNTimesteps()=0;
    virtual int GetStartTime()=0;
    virtual int GetEndTime() = 0;
    virtual int GetIncrement() = 0;
  //////////
  // Material and particle data is broken into materials
    virtual MPMaterial* getMaterial(int i)=0;
    virtual int getNMaterials()=0;
  //////////
  // GetGraphData will fill and array of length nTimesteps with
  // values corresponding to single variable of a particular particle
  // overtime.
    virtual void GetParticleData(int particleId,
			      int variableId,
			      int materialId,
			      bool isVector,
			      Array1<float>& values)=0;
  //////////
  // If the filename is set and it hasn't been read, read it and put
  // it into SCIRun data structures.
    virtual int readfile()=0;

  ////////// Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

};

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.2  1999/08/17 06:40:08  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:59  mcq
// Initial commit
//
// Revision 1.3  1999/06/09 23:21:33  kuzimmer
// reformed the material/particle classes and removed the particleSetExtensions.  Now MPVizParticleSet inherits from cfdlibParticleSet-->use the new stl routines to dynamically check the particleSet type
//
// Revision 1.2  1999/04/27 23:18:39  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//

#endif
