#ifndef _MFMPParticleGridReader_h_
#define _MFMPParticleGridReader_h_  1

/*----------------------------------------------------------------------
CLASS
    MFMPParticleGridReader

    A class for reading multiple files from parallel output containing
    both particle and gridded data. 

OVERVIEW TEXT
    Reads and performs actions on binary Material particle Grid files.


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

#include <vector>
using std::vector;
namespace Uintah {
namespace Datatypes {


class MFMPParticleGridReader : public ParticleGridReader {
public:
    MFMPParticleGridReader();
    virtual ~MFMPParticleGridReader();
    MFMPParticleGridReader(const MFMPParticleGridReader&);
    MFMPParticleGridReader(const clString& filenames );
    MFMPParticleGridReader(const clString& filenames, int start, int end, int incr);

  //////////
  // SetFile expects a string list of  filenames including full path
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

  void readGrid( vector<MPRead*>& reader);
  void readParticles(vector<MPRead*>& reader);
  void getindices(Point minPt, Point subminPt, Point dPt,
			      int& i, int& j, int& k);

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

  clString filenames;
  bool TwoD;
  double xmin, xmax, ymin, ymax, zmin, zmax; 
};

} // End namespace Datatypes
} // End namespace Uintah


#endif
