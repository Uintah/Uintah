/*
 *
 * GeometryRenderer: Provides geometry transmission capability.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <Rendering/GeometryRenderer.h>
#include <Network/dataItem.h>

namespace SemotusVisum {
namespace Rendering {

const char * const
GeometryRenderer::name = "Geometry Transmission";

const char * const
GeometryRenderer::version = "$Revision$";

GeometryRenderer::GeometryRenderer() :
  Renderer( this, "GeometryRenderer" ),
  mailbox( "GeometryRendererMailbox", 10 ),
  vertexCount( -1 ), indexCount( -1 ), polygonCount( -1 ),
  indexed( -1 ), replace( -1 ) {
  
}

   
GeometryRenderer::~GeometryRenderer() {
  
}


void 
GeometryRenderer::setCallbacks() {

  // Set up our superclass's callbacks.
  setSuperCallbacks();
}

void 
GeometryRenderer::removeCallbacks() {

  // Remove our superclass's callbacks.
  removeSuperCallbacks();
}


void        
GeometryRenderer::sendRenderData( const char * data, int numBytes,
				  bool copy ) {
  std::cerr << "Using geom send render data" << endl;
  //helper->parent = this;
  // Set the data dimensions - one long string.
  x = numBytes; y = 1;

  /* Send it to the helper */
  if ( copy ) {
    //std::cerr << "Copying data at " << (void *)data << "...";
    char * _data = scinew char[ numBytes ];
    memcpy( _data, data, numBytes );
    //std::cerr << "done copying" << endl;
    helper->getMailbox().send( dataItem( _data, numBytes, true ) );
  }
  else {
    //std::cerr << "Not copying data!" << endl;
    helper->getMailbox().send( dataItem( data, numBytes, false ) );
  }
}

char *
GeometryRenderer::preprocess( const char * data, int &numBytes ) {
  return (char *)data; // don't need to do anything!
}

}
}
