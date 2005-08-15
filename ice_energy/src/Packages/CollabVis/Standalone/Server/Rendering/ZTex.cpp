/*
 *
 * ZTex: Provides ZTex creation.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: August 2001
 *
 */
#include <Malloc/Allocator.h>

#include <Rendering/ZTex.h>
#include <Rendering/ZTex/ZImage.h>
#include <Rendering/ZTex/HeightSimp.h>
#include <Rendering/ZTex/SimpMesh.h>

#include <Logging/Log.h>

#include <GL/glu.h>
#include <GL/gl.h>

#include <stdio.h>

namespace SemotusVisum {
namespace Rendering {

using namespace Logging;

ZTex::ZTex() : deleteMesh( true ), mesh( NULL ), modelmat( NULL ),
	       projmat( NULL ), viewport( NULL ), eyept( NULL )
{ }


ZTex::~ZTex() {
  if ( deleteMesh )
    delete mesh;

  delete modelmat;
  delete projmat;
  delete viewport;
  delete eyept;
}


void
ZTex::setMatrices( const double * modelview,
		   const double * projection,
		   const int    * viewport ) {
  
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
}

void
ZTex::setEyepoint( const double * eyePoint ) {
  if ( eyePoint ) {
    if ( eyept ) delete eyept;
    eyept = scinew double[3];
    eyept[ 0 ] = eyePoint[ 0 ];
    eyept[ 1 ] = eyePoint[ 1 ];
    eyept[ 2 ] = eyePoint[ 2 ];
  }
}


bool
ZTex::reprojectVertices( Model * Mo ) {
  double tmp[3];
  int numVertices = Mo->Objs[0].verts.size();
  Vector * vertices;

  // Use gluUnProject()
  for ( int j = 0; j < numVertices; j++ ) {
    vertices = &(Mo->Objs[0].verts[j]);
    
    if ( gluUnProject( vertices->x, vertices->y, vertices->z,
		       modelmat, projmat, viewport,
		       &tmp[0], &tmp[1], &tmp[2] ) == GL_FALSE ) {
      std::cerr << "GLU error: Vertex = " << j << endl;
      return false;
    }
    vertices->x = tmp[0];
    vertices->y = tmp[1];
    vertices->z = tmp[2];
  }

  return true;
}

bool
ZTex::mkZTex( const unsigned int * zbuffer, const int zBufSize,
	      const int width, const int height, int &meshSize ) {

  /* Get an internal representation of the Z buffer */
  ZImage ZBuf( width, height );
  memcpy( ZBuf.IPix(), zbuffer, zBufSize );

  /* Simplify the height field into a mesh. Note - I have no freakin
     idea what the last two parameters are for. Fuck undocumented code. */
  HeightSimp* HFSimplifier = scinew HeightSimp(ZBuf, 0x01000000, 0x01000000);
  if ( !HFSimplifier ) return false;
  
  SimpMesh *SM = HFSimplifier->HFSimp();
  if ( !SM ) return false;

  // NOT YET! SM->Simplify(100); 
  
  delete HFSimplifier;

  /* Create an organized model from the simplified mesh. */
  Model * Mo = scinew Model(SM->ExportObject());

  if ( !Mo ) return false;
  delete SM;

  std::cerr << "We have " << Mo->Objs.size() << " objects!" << endl;

  /* Make sure we only have triangles! */
  Mo->Objs[0].MakeTriangles();

  
  
  /* Reproject the vertices if we have the appropriate matrices */
  if ( modelmat &&  projmat && viewport ) {
    Log::log( Logging::DEBUG, "Reprojecting vertices in ZTex..." );
    if ( !reprojectVertices( Mo ) ) {
      delete Mo;
      return false;
    }
  }
  
  /* Remove triangles if we have the eyepoint set */
  if ( eyept ) {
    Log::log( Logging::DEBUG, "Removing triangles from ZTex" );
    int oldTri = Mo->Objs[0].verts.size();


    Vector pos;
    //if ( eyept[0] == 0.0 && eyept[1] == 0.0 && eyept[2] == 0.0 ) {
    if ( ABS( eyept[0] ) < EPSILON &&
	 ABS( eyept[1] ) < EPSILON &&
	 ABS( eyept[2] ) < EPSILON ) {
      std::cerr << "Eyept is at origin. Dealing with it..." << endl;
      // Now, we multiply the eyepoint by the inverse of the modelview
      // matrix...
      Matrix44 model( modelmat, true );
      Vector eye( eyept[0], eyept[1], eyept[2] );
      pos = model.UnProject( eye );
      //pos = Vector( modelmat[13], modelmat[14], -modelmat[15] );
      model.Invert();
      std::cerr << "Inverse matrix is " << model << endl;
      std::cerr << "Using eyept of " << pos.x << " " << pos.y << " " <<
	pos.z << endl;
    }
    else 
      pos = Vector( eyept[0], eyept[1], eyept[2] );

    std::cerr << "Using eyept of " << pos.x << " " << pos.y << " " <<
      pos.z << endl;
    
      // WTF? 0.2? Copied from previous use, but 0.2 not explained...
      Mo->RemoveTriangles( pos, 0.2 );
      
      std::cerr << "Before Removing Triangles: " << oldTri << "\tAfter: " <<
	Mo->Objs[0].verts.size() << endl;
      
  }
    
  
  /* Now, the model has a single object holding the triangles. */
  int numVertices = Mo->Objs[0].verts.size();
  
  /* Cast and copy the vertex data to float. */
  float * floatData = scinew float[ numVertices * 3 ];
  for ( int i = 0; i < numVertices; i++ ) {
    floatData[ i*3 + 0 ] = Mo->Objs[0].verts[ i ].x;
    floatData[ i*3 + 1 ] = Mo->Objs[0].verts[ i ].y;
    floatData[ i*3 + 2 ] = Mo->Objs[0].verts[ i ].z;
  }
  if ( mesh && deleteMesh )
    delete mesh;
  mesh = scinew char[ numVertices * 3 * sizeof( float ) ];
  memcpy( mesh, floatData, numVertices * 3 * sizeof( float ) );
  meshSize = numVertices * 3 * sizeof( float );
  
  /* Clean up */
  delete floatData;
  delete Mo;

  return true;
}



}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:38  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:23:36  simpson
// Adding CollabVis files/dirs
//
// Revision 1.7  2001/10/10 23:35:39  luke
// Removed file outputs from ZTex
//
// Revision 1.6  2001/10/05 18:29:30  luke
// Fixed nonzero eyept bug in ZTex
//
// Revision 1.5  2001/10/01 18:56:55  luke
// Scaling works to some degree on image renderer
//
// Revision 1.4  2001/09/26 17:05:05  luke
// ZTex works.
//
// Revision 1.3  2001/08/23 19:56:02  luke
// ZTex and ZTex Driver work
//
// Revision 1.2  2001/08/22 17:58:15  luke
// Fixed some small ZTex bugs
//
// Revision 1.1  2001/08/22 16:24:07  luke
// Initial import of ZTex code
//
