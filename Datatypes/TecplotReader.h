#ifndef SCI_Datatypes_TecplotReader_h
#define SCI_Datatypes_TecplotReader_h 1


/*----------------------------------------------------------------------
CLASS
    TecplotReader

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

#include <Datatypes/ParticleGridReader.h>
#include <Datatypes/Datatype.h>
#include <Datatypes/MEFluid.h>
#include <Classlib/LockingHandle.h>
#include <Classlib/Array1.h>
#include <Classlib/String.h>
#include <Geometry/Vector.h>


class TecplotReader : public ParticleGridReader {
public:
    TecplotReader();
    virtual ~TecplotReader();
    TecplotReader(const TecplotReader&);
    TecplotReader(const clString& filename );
    TecplotReader(const clString& filename, int start, int end, int incr);
    virtual ParticleGridReader* clone() const;

  //////////
  // SetFile expects a filename include full path
    virtual void SetFile(const clString& filename);
    virtual clString GetFile();
    virtual int GetNTimesteps();
    virtual int GetStartTime(){ return startTime;}
    virtual int GetEndTime(){return endTime;}
    virtual int GetIncrement(){ return increment;}


  MEFluid* getFluid(int i);   
  int getNFluids();  // return the number of fluids
  //////////
  // GetGraphData will fill and array of length nTimesteps with
  // values corresponding to single variable of a particular particle
  // overtime.
    virtual void GetParticleData(int particleId,
			      int variableId,
			      int fluidId,
			      Array1<float>& values);
  //////////
  // If the filename is set and it hasn't been read, read it and put
  // it into SCIRun data structures.
    virtual int readfile();

  ////////// Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

private:
  ////////// CONSTANTS
  // max line length
  static const int LINEMAX = 1000;
  // max length of a variable name
  static const int VARNAMELEN = 40;
  // the standard maximum number of variables;
  static const int VARSLEN = 40;
  // max allowable fluids
  static const int MAXFLUIDS = 10;


  //////////  Animation vars
  int startTime;
  int endTime;
  int increment;

  // the number of fluids in the dataset
  Array1< MEFluid* > fluids;
  
  void setFluidNum( clString str, int& nFluids);

  void stripVar(const clString &, clString&, int& index);


  void readVars(istream& is);
  void readZone(istream& is);

  void readBlock(int,int,int,istream&);
  void readParticles(int, int, istream&);
  void getBounds(double& min, double& max,
		 int ii, int jj, int kk,
		 istream& is);

  ScalarFieldHandle makeScalarField(int ii, int jj, int kk, istream& is);
  VectorFieldHandle makeVectorField(int ii, int jj, int kk, istream& is);

  int find( char c, char* buf);
  int isBlock(char*);
  int isVectorVar( const clString&);

  double xmin, xmax, ymin, ymax, zmin, zmax;  // bounds for block

  Array1<clString> variables;

  clString filename;
  int nTimesteps;

  bool TwoD;

};

#endif
