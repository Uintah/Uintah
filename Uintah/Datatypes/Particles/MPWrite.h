//static char *id="@(#) $Id$";

/****************************************
CLASS
    MPWrite

    A class for writing Material/Particle files.

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
    June 28, 1999
****************************************/

#include <iostream.h> 
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Geometry/Point.h>
namespace Uintah {
namespace Datatypes {  

using namespace SCICore::Containers;
using namespace SCICore::Datatypes;
using SCICore::Geometry::Point;
//using SCICore::Geometry::Vector;
  
class MPWrite {
 public:
  
  // GROUP: Constructors
  //////////
  // Create the MPWrite object, passing a filename and a file
  // writing mode ios::out or ios::app.  The file is opened
  // with the appropriate mode and the object state is set to Open.
  MPWrite(clString fname, int mode = ios::out);

  // GROUP: Destructors
  //////////
  ~MPWrite();

  // Group: Write Functions
  //////////
  // All functions return 0 upon failure. Need to implement Exceptions.
  
  //////////
  // This must be the first call to any opened file (unless you are
  // appending to the file).  An example is:
  // write.BeginHeader("Junk", "P T", "sigXYZ", "", "More Junk") 
  // where write is an object of type MPWrite, Junk is the title,
  // P & T are scalar variables, sigXYZ is a vector Var, and "More 
  // Junk" is a comment.  Since there is only one material, the 
  // material parameter is an empty string.  A more complex example:
  // write.BeginHeader("Lots o Junk", "P T_H20 E_H20 T_AL E_AL",
  //                   "MO_H20 MO_AL","H20 AL", "Boring stuff")
  // Here materials H20 and AL are specified.  Scalar and vector 
  // variables for a particular material must end with the
  // material name.  The variable P corresponds to all materials.
  // The object state is set to Header.
  int BeginHeader(clString title,
		  clString fileType, // BIN or ASCII
		  clString comment);

  //////////
  // Add more comments or information to the Header.  This function
  // can only be called between BeginHeader and EndHeader.
  int AddComment(clString comment); 

  //////////
  // Close the Header.  Set the Object state back to Open with the
  // headerWritten flag set to true.
  int EndHeader();


  /////////
  // Begin writing an irregularly spaced structured  grid (mesh).  
  // type must be "NC_i", "CC_i", "FC_i",  The X,Y,and Z values
  // are arrays that represent the locations of the node or cell
  // centers.  The size of the array is implicitly determined by
  // the size of the three arrays.  Object state is set to Grid.
  int BeginGrid( clString name,
		 clString type,
		 clString scalarVars,
		 clString vectorVars,
		 Array1< double > X,  
		 Array1< double > Y,
		 Array1< double > Z );


  /////////
  // Begin writing a regularly spaced structured  grid (mesh).  
  // type must be "NC", "CC", "FC".  x,y, and z represent the
  // size of the mesh in each direction.  dx,dy, and dz
  // represent the regular spacing. Object state is set to Grid.
  int BeginGrid( clString name,
		 clString type,
		 clString scalarVars,
		 clString vectorVars,
		 double sizeX, double sizeY, double sizeZ,
		 double minX, double minY, double minZ,
		 double dx, double dy, double dz);

  /////////
  // This function checks to see if name is a variable in the header.
  // If it is, then dynamic type checking is performed on the 
  // scalarfield to determin if its type and size match that of 
  // the BeginGrid header.  This function a can only be called
  // between BeginGrid and EndGrid.
  int AddSVarToGrid( clString name,
		    ScalarField* var );

  int AddSVarToGrid( clString name, double *sf, int length);
  /////////
  // The same as the previous function, but for vector fields.
  // A vector field cannot be added until all scalar fields associated
  // with the scalar variable list have been added.
  int AddVecVarToGrid( clString name,
		    VectorField* var );

  int AddVecVarToGrid( clString name, Vector* vf, int length);

  int AddVecVarToGrid( clString name, double *vf, int length);

  /////////
  // Close the Grid. Set the object state to Open
  int EndGrid();


  ////////
  // Begin writing particles.  material must be a material in the
  // header's material string or "" if it is empty.  N is the number
  // of particles.
  int BeginParticles( clString name,
		      int N,
		      clString scalarVars,
		      clString vectorVars);

  ////////
  // This will not allow you to add more than N Particles. p is the 
  // position of the particle.  The size of the scalars array must be 
  // equal to the number of scalar vars.  Ditto for the vector values.
  int AddParticle( Point p, // position
		   Array1< double >& scalars, 
		   Array1< Vector >& vectors);

  ////////
  // If you did not add N particles, this will pad the file with zero
  // values until it has N particles.
  int EndParticles(); 

  


  
  

 private:
  enum FileType { BIN, ASCII };
  enum GridType{ NoGrid, NC, CC, FC, NC_i, CC_i, FC_i };
  enum State{ Open, Closed, Header, Grid, Particles };
  State state;
  FileType fileType;
  GridType currentGrid;
  ofstream os;
  clString filename;
  bool headerWritten;
  Array1< clString > sVars;
  Array1< clString > vVars;

  Array1< clString > psVars; 
  Array1< clString > pvVars;
 

  void printCurrentState(ostream& out);
  int pN;
  int pCount;
  int svCount;
  int vvCount;


};

} // end namespace Modules
} // end namespace Uintah
