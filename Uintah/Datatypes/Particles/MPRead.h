//static char *id="@(#) $Id$";

/****************************************
CLASS
    MPRead

    A class for reeding material particle files.

OVERVIEW TEXT


KEYWORDS
    Material/Particle Method

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    June 1999

    Copyright (C) 1999 SCI Group

LOG
    Created June 28, 1999
****************************************/

#include <SCICore/Containers/String.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/VectorField.h>

#include <fstream> // Cannot be <fstream> due to seekg and tellg
#include <iostream>

using std::ifstream;
using std::cout;
using std::cerr;

namespace Uintah {
namespace Datatypes {  

using namespace SCICore::Containers;
using namespace SCICore::Datatypes;

class MPRead {
 public:

  enum GridType{ NC, NC_i, CC, CC_i, FC, FC_i, None };

  MPRead(clString fname);
  ~MPRead();
  int ReadHeader(clString& title,
		 clString& fileType,
		 clString& comments);

  int getComments(clString& comments);
  int ReadBlock( clString& datatype ); // either Grid or Particles 

  // if block is a Grid get scalar and vector fields in Grid
  int GetGridInfo( clString& name,  // get size and type data
		   clString& type,
		   int& x, int& y, int& z,
		   Array1<clString>& scalarVars,
		   Array1<clString>& vectorVars);

  int getScalarVars(Array1< clString >& sv);
  int getVectorVars(Array1< clString >& vv);

  // if GridType = NC, CC, FC  use the first function
  int GetGridPoints( double& o_x, double& o_y, double& o_z,
		       double& dx, double& dy, double& dz );
  // otherwise you have an irregularly spaced grid, use this.
  int GetGridPoints( Array1<double>& X,
		     Array1<double>& Y,
		     Array1<double>& z);

     // These functions will use dynamic type checking to return the 
     // correct fields based on the "type" variable.
     int GetScalarField( ScalarFieldHandle& sf );
     int GetVectorField( VectorFieldHandle& vf );
     int GetScalarField( double  *sf, int& length );
     int GetVectorField( Vector  *vf, int& length );
     int GetParticleInfo( clString& name,
			  int& N,
			  Array1<clString>& s,
			  Array1<clString>& v); // number of particles
     // this can only be called N times after
     int GetParticle( Point& p,
		      Array1< double >& scalars,
		      Array1< Vector >& vectors);

  int EndBlock();
  int ReadData();
  
  int virtual GetParticleVariableValue(int particleId,
			       clString varname,
			       double& value);
//   int GetParticleVariableValue(int particleId,
// 			       clString varname,
// 			       Vector& value);

 private:

  enum FileType { BIN, ASCII };
  enum State{ Open, Closed, Header, Grid, Particles };
  enum GridState{ Empty, Info, Points, Scalars, Vectors };

  void SetCurrentType( const clString& type );

  GridType currentType;
  FileType fileType;
  GridState gridState;
  State state;
    ifstream is;
  float version;
  
  clString filename;

  bool headerRead;  
  Array1< clString > sVars;
  Array1< clString > vVars;
  Array1< clString > psVars;
  Array1< clString > pvVars;
  
  Point minPt;
  Point maxPt;
  
  int x_size;
  int y_size;
  int z_size;

  int nParticles;
  int pCount;
  int svCount;
  int vvCount;

};


} // end namespace Modules
} // end namespace Uintah
