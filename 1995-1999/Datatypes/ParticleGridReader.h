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


#include <Datatypes/Datatype.h>
#include <Classlib/LockingHandle.h>
#include <Classlib/Array1.h>
#include <Datatypes/MEFluid.h>
#include <Geometry/Vector.h>

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
  // Material and particle data is broken into fluids
    virtual MEFluid* getFluid(int i)=0;
    virtual int getNFluids()=0;
  //////////
  // GetGraphData will fill and array of length nTimesteps with
  // values corresponding to single variable of a particular particle
  // overtime.
    virtual void GetParticleData(int particleId,
			      int variableId,
			      int fluidId,
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

#endif
