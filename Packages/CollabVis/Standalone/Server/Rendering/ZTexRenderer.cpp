/*
 *
 * ZTexRenderer: Provides ZTex creation and transmission capability.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: July 2001
 *
 */

#include <Rendering/ZTexRenderer.h>
#include <Rendering/ZTex.h>
#include <Network/dataItem.h>
#include <Network/NetConversion.h>

#include <GL/glu.h>
namespace SemotusVisum {
namespace Rendering {

const char * const
ZTexRenderer::name = "ZTex Transmission";

const char * const
ZTexRenderer::version = "$Revision$";

ZTexRenderer::ZTexRenderer() :
  GeometryRenderer( this, "ZTexRenderer",  "ZTexRendererMailbox" ),
  modelmat( NULL ), projmat( NULL ), viewport( NULL ), eyept( NULL ),
  width(-1), height(-1)
#ifdef PREPROCESS
  , dataMutex("ZtexDataMutex")
#endif
{
  cerr << "In ZTexRenderer::ZTexRenderer" << endl;
  cerr << "End of ZTexRenderer::ZTexRenderer" << endl;  
}

   
ZTexRenderer::~ZTexRenderer() {
  delete modelmat;
  delete projmat;
  delete viewport;
  delete eyept;
  std::cerr << "Bye" << endl;
}


void 
ZTexRenderer::setCallbacks() {
  cerr << "In ZTexRenderer::setCallbacks" << endl;
  // Set up our superclass's callbacks.
  setSuperCallbacks();
  cerr << "End of ZTexRenderer::setCallbacks" << endl;

}

void 
ZTexRenderer::removeCallbacks() {
  cerr << "In ZTexRenderer::removeCallbacks" << endl;
  // Remove our superclass's callbacks.
  removeSuperCallbacks();
  cerr << "End of ZTexRenderer::removeCallbacks" << endl;
}

void
ZTexRenderer::setMatrices( const double * modelview,
			   const double * projection,
			   const int    * viewport ) {
  cerr << "In ZTexRenderer::setMatrices" << endl;
  if ( modelview ) {
    if ( modelmat ) delete modelmat;
    modelmat = scinew double[ 16 ];
    memcpy( modelmat, modelview, 16 * sizeof( double ) );
  }

  if ( projection ) {
    if ( projmat ) delete projmat;
    projmat = scinew double[ 16 ];
    memcpy( projmat, projection, 16 * sizeof( double ) );
  }

  if ( viewport ) {
    if ( this->viewport ) delete this->viewport;
    this->viewport = scinew int[ 4 ];
    memcpy( this->viewport, viewport, 4 * sizeof( int ) );
  }
  cerr << "End of ZTexRenderer::setMatrices" << endl;
}

void
ZTexRenderer::setEyepointAtZero( bool atZero ) {
  cerr << "In ZTexRenderer::setEyepointAtZero" << endl;
  if ( modelmat == NULL || eyept == NULL ) return;

  if ( atZero ) {
    if ( eyept[0] != 0.0 || eyept[1] != 0.0 || eyept[2] != 0 ) {
      Log::log( Logging::ERROR,
		"ZTexRenderer - setting eyepoint at zero when eyepoint is nonzero!" );
      return;
    }
    eyept[0] = modelmat[13]; modelmat[13] = 0.0;
    eyept[1] = modelmat[14]; modelmat[14] = 0.0;
    eyept[2] = modelmat[15]; modelmat[15] = 0.0;
  }
  else {
    if ( eyept[0] == 0.0 && eyept[1] == 0.0 && eyept[2] == 0.0 ) {
      Log::log( Logging::ERROR,
		"ZTexRenderer - setting eyepoint at nonzero when eyepoint is zero!" );
      return;
    }
    modelmat[13] = eyept[0]; eyept[0] = 0.0;
    modelmat[14] = eyept[1]; eyept[1] = 0.0;
    modelmat[15] = eyept[2]; eyept[2] = 0.0;
  }
  cerr << "End of ZTexRenderer::setEyepointAtZero" << endl;
}

void
ZTexRenderer::setEyepoint( const double * eyePoint ) {
  cerr << "In ZTexRenderer::setEyepoint" << endl;
  if ( eyePoint ) {
    if ( eyept ) delete eyept;
    eyept = scinew double[3];
    eyept[ 0 ] = eyePoint[ 0 ];
    eyept[ 1 ] = eyePoint[ 1 ];
    eyept[ 2 ] = eyePoint[ 2 ];
  }
  cerr << "End of ZTexRenderer::setEyepoint" << endl;
}

void        
ZTexRenderer::sendRenderData( const char * image, const int imageSize,
			      const unsigned int * zbuffer, const int zBufSize,
			      bool copy ) {
  cerr << "In ZTexRenderer::sendRenderData" << endl;
  /* For right now, we are not so clever:
   * First, create a (serialized) mesh from the given zbuffer.
   * Concatenate the image data with the zbuffer data (ie, mesh then image)
   * Send the total package to the helper to be compressed and sent.

   All of these steps could be optimized...
   */

#ifdef PREPROCESS
  if ( !dataMutex.tryLock() ) {
    Log::log( Logging::ERROR,
	      "Unable to access internal data structures!" );
    // EJL - This made the damn thing not do anything! Fucker! return;
  }
  // send the data
  this->imageSize = imageSize;
  char * allData = scinew char[ imageSize + zBufSize ];
  memcpy( allData, image, imageSize );
  memcpy( allData+imageSize, zbuffer, zBufSize );
  helper->getMailbox().send( dataItem( allData,
				       zBufSize + imageSize,
				       false ) );
#else
  /* First check that we've already set the transform matrices and eyept.
     If we haven't, note that in the log and return - there's nothing we
     can do. */
  if ( !modelmat || !projmat || !viewport ) {
    Log::log( Logging::ERROR,
	      "Transform matrices not set in ZTexRenderer - skipping" );
    return;
  }
  if ( !eyept ) {
    Log::log( Logging::ERROR,
	      "Eyepoint not set in ZTexRenderer - skipping" );
    return;
  }
  /* Also, insure that we have a valid dimension */
  if ( width <= 0 || height <= 0 ) {
    Log::log( Logging::ERROR,
	      "Dimension of Z buffer not set - skipping" );
    return;
  }

  int zmeshSize = 0;
  ZTex *z = scinew ZTex();

  // Set the transform matrices and eyepoint.
  z->setMatrices( modelmat, projmat, viewport );
  z->setEyepoint( eyept );

  // Make the mesh
  if ( !z->mkZTex( zbuffer, zBufSize,
		   width, height, zmeshSize ) ) {
    Log::log( Logging::ERROR, "Error creating Z Mesh in ZTex creation!" );
    delete z;
    return;
  }

  char * zmesh = z->getMesh();
  vertexCount = zmeshSize / (3 * sizeof(float) );

  char * allData = scinew char[ zmeshSize + imageSize ];
  Log::log( Logging::DEBUG, "Converting...");
  
  // Convert the mesh to network byte order.
  //NetConversion::convertFloat( (float *)zmesh,
  //			       zmeshSize,
  //			       (float *)allData );
			       
  HomogenousConvertHostToNetwork( (void *)allData,
				  (void *)zmesh,
				  FLOAT_TYPE,
				  zmeshSize/sizeof(float) );
  
  // We cannot do our normal conversion (as our data is packed), so we
  // simply copy the image data.
  //memcpy( allData + zmeshSize, image, imageSize );
  NetConversion::convertRGB( (char *)image,
			     imageSize,
			     (char *)(allData + zmeshSize ) );
  
  Log::log( Logging::DEBUG, "Converted...");			  

  
  //memcpy( allData, zmesh, zmeshSize );
  //memcpy( allData + zmeshSize, image, imageSize );

  Log::log( Logging::DEBUG, "ZTex built - transmitting..." );
  
  // Set the data dimensions - one long string.
  //x = zmeshSize + imageSize; y = 1;
  x = width; y = height;
  helper->getMailbox().send( dataItem( allData,
				       zmeshSize + imageSize,
				       false ) );

  // Clean up
  delete zmesh;
  //delete allData; // Deleting this causes bad shit to happen...
  delete z;
#endif

  cerr << "End of ZTexRenderer::sendRenderData" << endl;
}


void        
ZTexRenderer::sendRenderData( const char * data, int numBytes, bool copy ) {
  cerr << "In ZTexRenderer::sendRenderData" << endl; 
  // Set the data dimensions - one long string.
  x = numBytes; y = 1;

  /* Send it to the helper */
  helper->getMailbox().send( dataItem( data, numBytes, copy ) );
  cerr << "End of ZTexRenderer::sendRenderData" << endl;
}

char *
ZTexRenderer::preprocess( const char * data, int &numBytes ) {
  cerr << "In ZTexRenderer::preprocess" << endl;
#ifdef PREPROCESS
  /* First check that we've already set the transform matrices and eyept.
     If we haven't, note that in the log and return - there's nothing we
     can do. */
  if ( !modelmat || !projmat || !viewport ) {
    Log::log( Logging::ERROR,
	      "Transform matrices not set in ZTexRenderer - skipping" );
    return NULL;
  }
  if ( !eyept ) {
    Log::log( Logging::ERROR,
	      "Eyepoint not set in ZTexRenderer - skipping" );
    dataMutex.unlock();
    return NULL;
  }
  /* Also, insure that we have a valid dimension */
  if ( width <= 0 || height <= 0 ) {
    Log::log( Logging::ERROR,
	      "Dimension of Z buffer not set - skipping" );
    dataMutex.unlock();
    return NULL;
  }
  
  // Get local copy of data
  cerr << "Total size is " << numBytes << " bytes. " << endl;
  cerr << "Image is " << imageSize << " bytes." << endl;
  char * image = scinew char[ imageSize ];
  memcpy( image, data, imageSize );
  int zBufSize = numBytes - imageSize;
  cerr << "ZBuf is " << zBufSize << " bytes " << endl;
  unsigned int * zbuffer =
    scinew unsigned int[ zBufSize/ sizeof(unsigned int) ];
  memcpy( zbuffer, data+imageSize, zBufSize );
  
  int zmeshSize = 0;
  ZTex *z = scinew ZTex();

  // Set the transform matrices and eyepoint.
  z->setMatrices( modelmat, projmat, viewport );
  z->setEyepoint( eyept );

  // Make the mesh
  if ( !z->mkZTex( zbuffer, zBufSize,
		   width, height, zmeshSize ) ) {
    Log::log( Logging::ERROR, "Error creating Z Mesh in ZTex creation!" );
    delete z;
    dataMutex.unlock();
    return NULL;
  }

  char * zmesh = z->getMesh();
  vertexCount = zmeshSize / (3 * sizeof(float) );

  char * allData = scinew char[ zmeshSize + imageSize ];
  Log::log( Logging::DEBUG, "Converting...");
  
  // Convert the mesh to network byte order.
  //NetConversion::convertFloat( (float *)zmesh,
  //			       zmeshSize,
  //			       (float *)allData );

  // TMP
  if ( 1 ) {
    FILE * f = fopen( "texcoords.out", "w" );
    float * mesh = (float *)zmesh;
    double x,y,z;
    for ( int i = 0; i < vertexCount*3; i+=3 ) {
      gluProject( mesh[i], mesh[i+1], mesh[i+2],
		  modelmat,
		  projmat,
		  viewport, &x, &y, &z );
      fprintf( f, "Vertex %d: (%0.3f, %0.3f, %0.3f) -> [%0.3f, %0.3f]\n",
	       (i/3)+1, mesh[i], mesh[i+1], mesh[i+2], x, y );
    }
      
    fclose(f);
  }
  HomogenousConvertHostToNetwork( (void *)allData,
				  (void *)zmesh,
				  FLOAT_TYPE,
				  zmeshSize/sizeof(float) );
  
  // We cannot do our normal conversion (as our data is packed), so we
  // simply copy the image data.
  //memcpy( allData + zmeshSize, image, imageSize );
  NetConversion::convertRGB( (char *)image,
			     imageSize,
			     (char *)(allData + zmeshSize ) );
  
  Log::log( Logging::DEBUG, "Converted...");			  

  // TMP
  if ( 1 ) {
    FILE * f = fopen( "image.out", "w" );
    fwrite( allData+zmeshSize, 1, imageSize, f );
    fclose(f);
    f = fopen( "tris.out", "w" );
    fwrite( allData, 1, zmeshSize, f );
    fclose( f );
  }
  // /TMP
  
  //memcpy( allData, zmesh, zmeshSize );
  //memcpy( allData + zmeshSize, image, imageSize );

  Log::log( Logging::DEBUG, "ZTex built - transmitting..." );
  
  // Set the data dimensions - one long string.
  //x = zmeshSize + imageSize; y = 1;
  x = width; y = height;
  
  // Clean up
  delete zmesh;
  //delete allData; // Deleting this causes bad shit to happen...
  delete z;

  numBytes = zmeshSize + imageSize;
  dataMutex.unlock();
  return allData;
#else
  return (char *)data;
#endif
  cerr << "End of ZTexRenderer::preprocess" << endl;
}

}
}
