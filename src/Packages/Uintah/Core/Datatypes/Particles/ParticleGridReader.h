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

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Containers/String.h>
#include "ParticleSet.h"
#include "VizGrid.h" 

namespace Uintah {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::Containers::Array1;
using SCICore::Containers::clString;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Geometry::Vector;
using namespace SCICore::Datatypes;
  
class ParticleGridReader;
typedef LockingHandle<ParticleGridReader> ParticleGridReaderHandle;

class ParticleGridReader : public Datatype {
public:
    ParticleGridReader();
    virtual ~ParticleGridReader();
    ParticleGridReader(const ParticleGridReader&);

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
  // GetGraphData will fill and array of length nTimesteps with
  // values corresponding to single variable of a particular particle
  // overtime.
    virtual void GetParticleData(int particleId,
			      int variableId,
			      int materialId,
			      bool isVector,
			      Array1<float>& values)=0;
    virtual void GetParticleData(int particleId,
				 clString pSetName,
                                 clString varname,
                                 Array1<double>& values) = 0;
  //////////
  // If the filename is set and it hasn't been read, read it and put
  // it into SCIRun data structures.
    virtual int readfile()=0;

  //////////
  // Get Grid and Particle information
    virtual int GetNGrids() = 0;
    virtual int GetNParticleSets() = 0;
  
    virtual ParticleSetHandle GetParticleSet( clString name) = 0;
    virtual VizGridHandle  GetGrid(clString name) = 0;
    virtual ParticleSetHandle GetParticleSet( int i) = 0;
    virtual VizGridHandle  GetGrid(int i) = 0;
  ////////// Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

};

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.5  1999/12/28 21:09:08  kuzimmer
// modified file readers so that we can read multiple files for parallel output
//
// Revision 1.4  1999/09/21 16:08:30  kuzimmer
// modifications for binary file format
//
// Revision 1.3  1999/08/25 03:49:02  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
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
