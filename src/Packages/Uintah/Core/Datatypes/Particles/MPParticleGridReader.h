#ifndef _MPParticleGridReader_h_
#define _MPParticleGridReader_h_  1

/*----------------------------------------------------------------------
CLASS
    MPParticleGridReader

    A class for reading files containing both particle and gridded data.

OVERVIEW TEXT
    Reads and performs actions on tecplot files.


KEYWORDS
    ParticleGridReader

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 6, 1999
----------------------------------------------------------------------*/

#include "ParticleGridReader.h"
#include "MPMaterial.h"
#include "MPRead.h"
#include "MPVizGrid.h"
#include "MPVizParticleSet.h"

#include <SCICore/Util/NotFinished.h>
#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/Vector.h>

namespace Uintah {
namespace Datatypes {


class MPParticleGridReader : public ParticleGridReader {
public:
    MPParticleGridReader();
    virtual ~MPParticleGridReader();
    MPParticleGridReader(const MPParticleGridReader&);
    MPParticleGridReader(const clString& filename );
    MPParticleGridReader(const clString& filename, int start, int end, int incr);

  //////////
  // SetFile expects a filename include full path
    virtual void SetFile(const clString& filename);
    virtual clString GetFile();
    virtual int GetNTimesteps();
    virtual int GetStartTime(){ return startTime;}
    virtual int GetEndTime(){return endTime;}
    virtual int GetIncrement(){ return increment;}


  //////////
  // GetGraphData will fill and array of length nTimesteps with
  // values corresponding to single variable of a particular particle
  // overtime.
  virtual void GetParticleData(int particleId,
			      int variableId,
			      int materialId,
			      bool isVector,
			      Array1<float>& values);
  
  virtual void GetParticleData(int particleId,
			       clString pSetName,
			       clString varname,
			       Array1<double>& values);

  //////////
  // Get Grid and Particle information
  virtual int GetNGrids(){ return vgmap.size();}
  virtual int GetNParticleSets(){return psmap.size(); }
  
  virtual ParticleSetHandle GetParticleSet(clString);
  virtual VizGridHandle  GetGrid(clString);

  virtual ParticleSetHandle GetParticleSet( int i );
  virtual VizGridHandle GetGrid (int i );
  //////////
  // If the filename is set and it hasn't been read, read it and put
  // it into SCIRun data structures.
    virtual int readfile();

  ////////// Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

private:

  void readGrid( MPRead& reader);
  void readParticles(MPRead& reader);
  

  //////////  Animation vars
  int startTime;
  int endTime;
  int increment;

  // the number of materials in the dataset
  struct ltstr
  {
    bool operator()(clString s1, clString s2) const
      {
	return (s1 < s2);
      }
  };

  map< clString, ParticleSetHandle, ltstr > psmap;
  map< clString, VizGridHandle, ltstr > vgmap;

  clString filename;
  bool TwoD;
  double xmin, xmax, ymin, ymax, zmin, zmax; 
};

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.2  1999/12/28 21:09:08  kuzimmer
// modified file readers so that we can read multiple files for parallel output
//
// Revision 1.1  1999/09/21 16:28:31  kuzimmer
// forgot in previous update
//
// Revision 1.1  1999/07/27 17:08:58  mcq
// Initial commit
//
// Revision 1.3  1999/06/09 23:23:44  kuzimmer
// Modified the modules to work with the new Material/Particle classes.  When a module needs to determine the type of particleSet that is incoming, the new stl dynamic type testing is used.  Works good so far.
//
// Revision 1.2  1999/04/27 23:18:42  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//

#endif
