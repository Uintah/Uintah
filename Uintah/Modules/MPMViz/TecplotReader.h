#ifndef _TecplotReader_h_
#define _TecplotReader_h_  1

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

#include <SCICore/CoreDatatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Geometry/Vector.h>

#include <Uintah/Datatypes/Particles/ParticleGridReader.h>
#include <Uintah/Datatypes/Particles/MPMaterial.h>

namespace Uintah {
namespace Datatypes {

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


  MPMaterial* getMaterial(int i);   
  int getNMaterials();  // return the number of materials
  //////////
  // GetGraphData will fill and array of length nTimesteps with
  // values corresponding to single variable of a particular particle
  // overtime.
  virtual void GetParticleData(int particleId,
			      int variableId,
			      int materialId,
			      bool isVector,
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
  // max allowable materials
  static const int MAXMATERIALS = 10;


  //////////  Animation vars
  int startTime;
  int endTime;
  int increment;

  // the number of materials in the dataset
  Array1< MPMaterial *> materials;
  
  void setMaterialNum( clString str, int& nMaterialss);

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

  int ComputeAdjustedIndex(int variableId, int materialId, bool isVector);

  int find( char c, char* buf);
  int isBlock(char*);
  int isVectorVar( const clString&);

  double xmin, xmax, ymin, ymax, zmin, zmax;  // bounds for block

  Array1<clString> variables;

  clString filename;
  int nTimesteps;

  bool TwoD;

};

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.2  1999/08/17 06:40:12  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
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
