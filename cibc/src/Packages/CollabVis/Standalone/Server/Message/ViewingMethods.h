/*
 *
 * ViewingMethods: Message helpers for various viewing methods.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: May 2001
 *
 */

#ifndef __VIEWING_METHODS_H_
#define __VIEWING_METHODS_H_

#include <iostream>
#include <values.h>

namespace SemotusVisum {
namespace Message {

using namespace std;

class ViewMethod {
public:

  //////////
  // Constructor. Sets all pointers to NULL.
  ViewMethod()/* : compression( NULL )*/ {
  }

  //////////
  // Destructor. Deletes all allocated memory.
  virtual ~ViewMethod() {
    /*delete compression;*/
  }
  /*
  //////////
  // Sets the new compression scheme. Allocates & copies the input
  inline void setCompression( const char * name ) {
    if ( name ) compression = strdup( name );
  }

  //////////
  // Fills in the param with the current compression scheme. Does no alloc.
  inline void getCompression( char * &name ) {
    name = this->compression;
  }
  */
  //////////
  // Operator to send params to a stream.
  //  friend inline std::ostream& operator<<(std::ostream &o,
  //					 const ViewMethod &vm ) {
  //   unsigned i = sizeof(vm);
    /*if ( vm.compression )
      o << "Compression: " << vm.compression << endl;
      else
      o << "Compression: None " << endl;*/
  //  return o;
  // }

protected:
  
  // Compression format
  //  char * compression;
};


/**************************************
 
CLASS
   VMImageStreaming
   
KEYWORDS
   Image Streaming, Message
   
DESCRIPTION

   This class provides read/write access to the parameters used by
   the SetViewingMethod message class. It supports all the parameters
   needed by Image Streaming.
   
****************************************/

class VMImageStreaming : public ViewMethod {
public:
  
  //////////
  // Constructor. Sets all pointers to NULL.
  VMImageStreaming() : resX(-1), resY(-1), reduction(-1),
		       subimage(-1),
    pictureFormat( NULL ), lighting(-1), fog(-1), shading( NULL ),
    eyeX( MINFLOAT ), eyeY( MINFLOAT ), eyeZ( MINFLOAT ),
    atX( MINFLOAT ), atY( MINFLOAT ), atZ( MINFLOAT ),
    upX( MINFLOAT ), upY( MINFLOAT ), upZ( MINFLOAT ),
    fov( MINFLOAT ), near( MINFLOAT ), far( MINFLOAT ) {
  }

  //////////
  // Copy constructor. Allocates memory for pictureFormat and shading.
  VMImageStreaming( const VMImageStreaming &vmi ) :
    resX(vmi.resX), resY(vmi.resY), reduction(vmi.reduction),
    subimage(-1),
    lighting(vmi.lighting), fog(vmi.fog), 
    eyeX( vmi.eyeX ), eyeY( vmi.eyeY ), eyeZ( vmi.eyeZ ),
    atX( vmi.atX ), atY( vmi.atY ), atZ( vmi.atZ ),
    upX( vmi.upX ), upY( vmi.upY ), upZ( vmi.upZ ),
    fov( vmi.fov ), near( vmi.near ), far( vmi.far ) {

    std::cerr << "VMI picture format: " << vmi.pictureFormat << endl;
    std::cerr << "VMI shading: " << vmi.shading << endl;
    if ( vmi.pictureFormat ) pictureFormat = strdup(vmi.pictureFormat);
    else pictureFormat = NULL;
    if ( vmi.shading ) shading = strdup(vmi.shading);
    else shading = NULL;
  }
    
  //////////
  // Destructor. Deletes all allocated memory.
  ~VMImageStreaming() {
    delete pictureFormat;
    delete shading;
  }

  //////////
  // Sets the resolution to the params given.
  inline void setResolution( int x, int y, int reduction ) {
    resX = x; resY = y; this->reduction = reduction;
  }

  //////////
  // Fills in the supplied params with the current resolution.
  inline void getResolution( int &x, int &y, int &reduction ) const {
    x = resX; y = resY; reduction = this->reduction;
  }

  inline void setSubimage( const bool on ) {
    subimage = on ? 1 : 0 ;
  }

  inline void getSubimage( int &set ) const {
    set = subimage;
  }
  
  
  //////////
  // Sets the current picture format. Allocates & copies the input.
  inline void setPictureFormat( const char * format ) {
    if ( format ) pictureFormat = strdup( format );
  }

  //////////
  // Fills in the param with the current format. Does no allocation.
  inline void getPictureFormat( char * &format ) {
    format = pictureFormat;
  }

  //////////
  // Sets the rendering parameters. Allocates & copies the shading param.
  inline void setRendering( int lighting, const char * shading, int fog ) {
    this->lighting = lighting; this->fog = fog;
    if ( shading ) this->shading = strdup( shading );
  }

  //////////
  // Fills in the supplied params with the current rendering params.
  // Does no allocation.
  inline void getRendering( int &lighting, char * &shading,
			    int &fog ) {
    lighting = this->lighting; shading = this->shading; fog = this->fog;
  }
  
  //////////
  // Sets the new eyepoint params to those supplied.
  inline void setEyePoint( double x, double y, double z ) {
    eyeX = x; eyeY = y; eyeZ = z;
  }

  //////////
  // Fills in the supplied params with the current eyepoint.
  inline void getEyePoint( double &x, double &y, double &z ) const {
    x = eyeX; y = eyeY; z = eyeZ;
  }

  //////////
  // Sets the new look at point params to those supplied.
  inline void setLookAtPoint( double x, double y, double z ) {
    atX = x; atY = y; atZ = z;
  }

  //////////
  // Fills in the supplied params with the current look at point.
  inline void getLookAtPoint( double &x, double &y, double &z ) const {
    x = atX; y = atY; z = atZ;
  }

  //////////
  // Sets the new up vector params to those supplied.
  inline void setUpVector( double x, double y, double z ) {
    upX = x; upY = y; upZ = z;
  }

  //////////
  // Fills in the supplied params with the current up vector.
  inline void getUpVector( double &x, double &y, double &z ) const {
    x = upX; y = upY; z = upZ;
  }

  //////////
  // Sets the new perspective params to those supplied.
  inline void setPerspective( double fov, double near, double far ) {
    this->fov = fov; this->near = near; this->far = far;
  }

  //////////
  // Fills in the supplied params with the current perspective.
  inline void getPerspective( double &fov, double &near, double &far ) const{
    fov=this->fov; near=this->near; far=this->far;
  }
  
  //////////
  // Operator to send params to a stream.
  friend inline std::ostream& operator<<(std::ostream &o,
					 const VMImageStreaming &vmi) {
    if ( vmi.resX != -1 || vmi.resY != -1 || vmi.reduction != -1 )
      o << "Resolution: (" << vmi.resX << ", " << vmi.resY
	<< ")\tReduction: " << vmi.reduction << endl;
    else
      o << "Resolution: Not set\tReduction: Not set" << endl;
    if ( vmi.subimage != -1 ) {
      if ( vmi.subimage == 0 )
	o << "Subimage: Disabled" << endl;
      else
	o << "Subimage: Enabled" << endl;
    }
    else 
      o << "Subimage: Not set" << endl;
    if ( vmi.pictureFormat )
      o << "Picture format: " << vmi.pictureFormat << endl;
    else
      o << "Picture format: None" << endl;

    o << "Rendering: \tLighting = ";
    if ( vmi.lighting == 1 )
      o << "on";
    else if (vmi.lighting == -1)
      o << "Not set";
    else o << "off";
    if ( vmi.fog == 1 )
      o << "\tFog = on";
    else if (vmi.fog == -1)
      o << "\tFog = Not set";
    else o << "\tFog = off";
    if ( vmi.shading )
      o << "\tShading = " << vmi.shading << endl;
    else
      o << "\tShading = None" << endl;
    /*    if ( vmi.compression )
      o << "Compression: " << vmi.compression << endl;
    else
    o << "Compression: None" << endl;*/
    if ( vmi.eyeX != MINFLOAT || vmi.eyeY != MINFLOAT ||
	 vmi.eyeZ != MINFLOAT )
      o << "Eye: (" << vmi.eyeX << ", " << vmi.eyeY
	<< ", " << vmi.eyeZ << ")" << endl;
    if ( vmi.atX != MINFLOAT || vmi.atY != MINFLOAT ||
	 vmi.atZ != MINFLOAT )
      o << "At: (" << vmi.atX << ", " << vmi.atY
	<< ", " << vmi.atZ << ")" << endl;
    if ( vmi.upX != MINFLOAT || vmi.upY != MINFLOAT ||
	 vmi.upZ != MINFLOAT )
      o << "Up: (" << vmi.upX << ", " << vmi.upY
	<< ", " << vmi.upZ << ")" << endl;
    if ( vmi.fov != MINFLOAT || vmi.near != MINFLOAT ||
	 vmi.far != MINFLOAT )
      o << "FOV: " << vmi.fov << " Near: " << vmi.near <<
	" Far: " << vmi.far << endl;
    return o;
  }
  
protected:
  // Resolution params
  int resX, resY, reduction;

  // Subimage params
  int subimage;
  
  // Picture format
  char * pictureFormat;

  // Rendering parameters
  int lighting, fog;
  char * shading;

  // Eyepoint
  double eyeX, eyeY, eyeZ;

  // Look at point
  double atX, atY, atZ;

  // Up vector
  double upX, upY, upZ;

  // Perspective
  double fov, near, far;
};

/**************************************
 
CLASS
   VMGeometry
   
KEYWORDS
   Geometry, Message
   
DESCRIPTION

   This class provides read/write access to the parameters used by
   the SetViewingMethod message class. It supports all the parameters
   needed by Geometry Transmission.
   
****************************************/
class VMGeometry : public ViewMethod {
public:

  //////////
  // Constructor. Sets all pointers to NULL.
  VMGeometry() : trianglesOnly(-1), dataFormat( NULL ) {}

  //////////
  // Destructors. Frees all allocated memory.
  ~VMGeometry() {
    delete dataFormat;
  }

  //////////
  // Sets the 'triangles only' parameter (ie, all the geometry is only
  // triangles) to the supplied param.
  inline void setTrianglesOnly( int only ) {
    trianglesOnly = only;
  }

  //////////
  // Fills in the supplied param with true if the geom is only triangles;
  // else false.
  inline void getTrianglesOnly( int &only ) const {
    only = trianglesOnly;
  }

  //////////
  // Sets the new data format. Allocates and copies param.
  inline void setDataFormat( const char * format ) {
    if ( format ) dataFormat = strdup( format );
  }

  //////////
  // Fills in the param with the current data format. Does no allocation.
  inline void getDataFormat( char * &format ) {
    format = dataFormat;
  }

  //////////
  // Operator to send params to a stream.
  friend inline std::ostream& operator<<(std::ostream &o,
					 const VMGeometry &vmg) {
    if ( vmg.trianglesOnly ) o << "Triangles only" << endl;
    else o << "Not only triangles" << endl;

    if ( vmg.dataFormat )
      o << "Data format: " << vmg.dataFormat << endl;
    else
      o << "Data format: None" << endl;
    return o;
  }
  
protected:
  // True if geometry will be supplied as only triangles.
  int trianglesOnly;

  // Data format (indexed vertices vs an array of vertices.
  char * dataFormat;
};

/**************************************
 
CLASS
   VMZTex
   
KEYWORDS
   ZTex, Geometry, Image Streaming, Message
   
DESCRIPTION

   This class provides read/write access to the parameters used by
   the SetViewingMethod message class. It supports all the parameters
   needed by the ZTex method.
   
****************************************/
class VMZTex : public VMImageStreaming, public VMGeometry {
public:
  
  //////////
  // Constructor. Relies on superclasses.
  VMZTex() {}

  //////////
  // Destructor. Relies on superclasses.
  ~VMZTex() {}

  //////////
  // Operator to send params to a stream.
  friend inline std::ostream& operator<<(std::ostream &o,
					 const VMZTex &vmz) {
    unsigned i = sizeof(vmz); // so compiler doesn't whine
    if ( i == 1 ) return o;
    //o << (VMImageStreaming)vmz << endl;
    //o << (VMGeometry)vmz << endl;
    return o;
  }
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:21  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:06  simpson
// Adding CollabVis files/dirs
//
// Revision 1.6  2001/08/01 21:40:50  luke
// Fixed a number of memory leaks
//
// Revision 1.5  2001/07/16 20:29:29  luke
// Updated messages...
//
// Revision 1.4  2001/06/05 17:44:56  luke
// Multicast basics working
//
// Revision 1.3  2001/05/29 03:43:12  luke
// Merged in changed to allow code to compile/run on IRIX. Note that we have a problem with network byte order in the networking code....
//
// Revision 1.2  2001/05/14 19:04:53  luke
// Documentation done
//
// Revision 1.1  2001/05/11 20:06:03  luke
// Initial coding of Message abstraction. Documentation not yet done.
//
