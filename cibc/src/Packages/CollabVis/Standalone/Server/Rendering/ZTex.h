/*
 *
 * ZTex: Provides ZTex creation.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: July 2001
 *
 */

#ifndef __ZTex_h_
#define __ZTex_h_

#include <Rendering/ZTex/Model.h>

namespace SemotusVisum {
namespace Rendering {

const double EPSILON = 1e-5; // Eyept to origin

/**************************************
 
CLASS
   ZTex
   
KEYWORDS
   ZTex
   
DESCRIPTION

   This class provides an infrastructure for ZTex creation.
   
****************************************/

class ZTex {
  
  
public:
  ////////////
  // Constructor
  ZTex();

  ////////////
  // Destructor
  ~ZTex();

  ////////////
  // Sets the OpenGL matrices for reprojecting the vertices.
  void setMatrices( const double * modelview,
		    const double * projection,
		    const int    * viewport );

  ////////////
  // Sets the eyepoint used to render the zbuffer; used to remove any
  // excess trianges from the mesh.
  void setEyepoint( const double * eyePoint );
  
  ////////////
  // Function to create a serial mesh from the given Z buffer.
  bool mkZTex( const unsigned int * zbuffer, const int zBufSize,
	       const int width, const int height,
	       int &meshSize );

  ////////////
  // Returns the serial mesh created with mkZTex(). Note - the user is
  // responsible for deleting the mesh after this function is called.
  inline char * getMesh() { deleteMesh = false; return mesh; }
  
protected:
  bool    deleteMesh;
  char   *mesh;
  double *modelmat;
  double *projmat;
  int    *viewport;
  double *eyept;

  bool    reprojectVertices( Model * Mo );
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:38  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:23:36  simpson
// Adding CollabVis files/dirs
//
// Revision 1.4  2001/10/10 15:18:57  luke
// Updates for IRIX
//
// Revision 1.3  2001/10/05 18:29:30  luke
// Fixed nonzero eyept bug in ZTex
//
// Revision 1.2  2001/08/23 19:56:02  luke
// ZTex and ZTex Driver work
//
// Revision 1.1  2001/08/22 16:24:07  luke
// Initial import of ZTex code
//
