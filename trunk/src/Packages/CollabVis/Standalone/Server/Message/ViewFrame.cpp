#include <Message/ViewFrame.h>

namespace SemotusVisum {
namespace Message {

ViewFrame::ViewFrame() : size( 0 ), width(-1), height(-1),
			 offX(-1), offY(-1), fullX(-1), fullY(-1),
			 isIndexed(-1),
			 isReplace(-1),
			 numVertices(-1), numIndices(-1), numPolygons(-1),
			 textureSize( -1 ) {
}

ViewFrame::~ViewFrame() {
}
   
void
ViewFrame::finish() {
  char * _true = "true";
  char * _false = "false";
  
  /* As this is a lightweight object, we want to reuse it. Thus, we allow
     multiple setSize calls (and finish calls) on this object. */

  /* Create an XML getclientlist document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  // Create a 'viewFrame' document
  Attributes attributes;

  char  *buffer[14];
  int i;
  for ( i = 0; i < 14; i++ ) {
    buffer[i] = scinew char[ 257 ];
    memset( buffer[i], 0, 257 );
  }
  
  snprintf( buffer[0], 256, "%u", size );
	    
  attributes.setAttribute( "size", buffer[0] );

  if ( width > 0 ) {
    snprintf( buffer[5], 256, "%d", width );
    attributes.setAttribute( "width", buffer[5]);
  }
  if ( height > 0 ) {
    snprintf( buffer[6], 256, "%d", height );
    attributes.setAttribute( "height", buffer[6]);
  }
  if ( offX != -1 && offY != -1 && fullX != -1 && fullY != -1 ) {
    snprintf( buffer[7], 256, "%d", offX );
    attributes.setAttribute( "OffsetX", buffer[7] );
    
    snprintf( buffer[8], 256, "%d", offY );
    attributes.setAttribute( "OffsetY", buffer[8] );
    
    snprintf( buffer[9], 256, "%d", fullX );
    attributes.setAttribute( "FullX", buffer[9] );
    
    snprintf( buffer[10], 256, "%d", fullY );
    attributes.setAttribute( "FullY", buffer[10] );
    
    snprintf( buffer[11], 256, "%d", bkgd[0] );
    attributes.setAttribute( "BackgroundR", buffer[11] );
    
    snprintf( buffer[12], 256, "%d", bkgd[1] );
    attributes.setAttribute( "BackgroundG", buffer[12] );
    
    snprintf( buffer[13], 256, "%d", bkgd[2] );
    attributes.setAttribute( "BackgroundB", buffer[13] );
    
  }
  if ( isIndexed != -1 ) 
    attributes.setAttribute( "indexed", (isIndexed ? _true : _false ) );
  if ( isReplace != -1 ) 
    attributes.setAttribute( "replace", (isReplace ? _true : _false ) );
  if ( numVertices != -1 ) {
    snprintf( buffer[1], 256, "%d", numVertices );
    attributes.setAttribute( "vertices", buffer[1] );
  }
  if ( numIndices != -1 ) {
    snprintf( buffer[2], 256, "%d", numIndices );
    attributes.setAttribute( "indices", buffer[2] );
  }
  if ( numPolygons != -1 ) {
    snprintf( buffer[3], 256, "%d", numPolygons );
    attributes.setAttribute( "polygons", buffer[3] );
  }
  if ( textureSize != -1 ){
    snprintf( buffer[4], 256, "%d", textureSize );
    attributes.setAttribute( "texture", buffer[4] );
  }
  writer.addElement( "viewFrame", attributes, "Following" );
  //  output = writer.writeOutputData();
  mkOutput( writer.writeOutputData() );

  for ( i = 0; i < 14; i++)
    delete[] buffer[i];
  
  finished = true;
}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:21  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:06  simpson
// Adding CollabVis files/dirs
//
// Revision 1.5  2001/09/23 02:24:11  luke
// Added collaborate message
//
// Revision 1.4  2001/08/29 19:58:05  luke
// More work done on ZTex
//
// Revision 1.3  2001/07/16 20:29:29  luke
// Updated messages...
//
// Revision 1.2  2001/05/21 19:19:30  luke
// Added data marker to end of message text output
//
// Revision 1.1  2001/05/11 20:06:03  luke
// Initial coding of Message abstraction. Documentation not yet done.
//
