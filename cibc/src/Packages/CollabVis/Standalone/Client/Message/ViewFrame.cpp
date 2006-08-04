#include <Message/ViewFrame.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {

ViewFrame::ViewFrame() : frameSize(0), width(-1), height(-1),
			 offX(-1), offY(-1), fullX(-1), fullY(-1),
			 indexed(-1),
			 replace(-1), vertices(-1), indices(-1),
			 polygons(-1), textureDimension(-1) {
}

ViewFrame::~ViewFrame() {
}
   
void
ViewFrame::finish() {
}

ViewFrame * 
ViewFrame::mkViewFrame( void * data ) {
  ViewFrame * f = scinew ViewFrame( );
  
  // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building a set viewing message
  String element;

  element = reader.nextElement(); // header
  if ( element == NULL ){
    Log::log( ERROR, "BUG - no header in viewframe!" );
    return NULL;
  }

  /* Get and log client info */
  Attributes attributes;

  attributes = reader.getAttributes();
  string val = attributes.getAttribute( "size" );
  if ( val.empty() ) {
    Log::log( ERROR, "No frame size set in viewFrame header" );
    return NULL;
  }
  f->frameSize = atoi( val );

  val = attributes.getAttribute( "width" );
  if ( !val.empty() )
    f->width = atoi( val );
  val = attributes.getAttribute( "height" );
  if ( !val.empty() )
    f->height = atoi( val );
  
  val = attributes.getAttribute( "OffsetX" );
  if ( !val.empty() )
    f->offX = atoi( val );
  val = attributes.getAttribute( "OffsetY" );
  if ( !val.empty() )
    f->offY = atoi( val );

  
  val = attributes.getAttribute( "FullX" );
  if ( !val.empty() )
    f->fullX = atoi( val );
  val = attributes.getAttribute( "FullY" );
  if ( !val.empty() )
    f->fullY = atoi( val );

  int r=-1,g=-1,b=-1;
  val = attributes.getAttribute( "BackgroundR" );
  if ( !val.empty() ) r = atoi( val );
  val = attributes.getAttribute( "BackgroundG" );
  if ( !val.empty() ) g = atoi( val );
  val = attributes.getAttribute( "BackgroundB" );
  if ( !val.empty() ) b = atoi( val );
  f->bkgd = Color( r, g, b );
  
  string indexed = attributes.getAttribute( "indexed" );
  if ( !indexed.empty() )
    if ( !strcasecmp( indexed, "true" ) )
      f->indexed = 1;
    else 
      f->indexed = 0;
  
  string replace = attributes.getAttribute( "replace" );
  if ( !replace.empty() )
    if ( !strcasecmp( replace, "true" ) )
      f->replace = 1;
    else 
      f->replace = 0;
  
  string vertices = attributes.getAttribute( "vertices" );
  if ( !vertices.empty() ) {
    f->vertices = atoi( vertices );
  }
  
  string indices = attributes.getAttribute( "indices" );
  if ( !indices.empty() )
    f->indices = atoi( indices );
  
  string polygons = attributes.getAttribute( "polygons" );
  if ( !polygons.empty() )
    f->polygons = atoi( polygons );
  
  string texture = attributes.getAttribute( "texture" );
  if ( !texture.empty() )
    f->textureDimension = atoi( texture );

  return f;
}

}
//
// $Log$
// Revision 1.1  2003/07/22 20:59:29  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:13  simpson
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
