#include <Message/GetZTex.h>

namespace SemotusVisum {
namespace Message {

using namespace XML;

GetZTex::GetZTex() : eyeX(0), eyeY(0), eyeZ(0), atX(0), atY(0), atZ(0),
		     upX(0), upY(0), upZ(0), 
		     matrix( NULL ), matrixSet( false ), matrixOurs( true ) {
}

GetZTex::~GetZTex() {
  if ( matrixOurs )
    delete[] matrix;
}

void
GetZTex::finish() {
  /* We generate no XML yet - but maybe we should... */
}

void
GetZTex::setMatrix( const double * newmatrix ) {
  if ( matrix )
    delete[] matrix;

  if ( !newmatrix ) return;

  setMatrixSet( true ); // We have a matrix

  matrix = scinew double[ 16 ];
  
  memcpy( matrix, newmatrix, 16 * sizeof(double) );

}


GetZTex *
GetZTex::mkGetZTex( void * data ) {
  GetZTex * g = scinew GetZTex();
  char * name;
  double matrix[16];
  
  // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building a get ztex message
  String element;
  Attributes attributes;
  
  element = reader.nextElement(); // getztex
  if ( element == NULL )
    return NULL;

  while ( element != 0 ) {
    name = XMLI::getChar( element );

    if ( !strcasecmp( name, "eyePoint" ) ) {
      double x=-1, y=-1, z=-1;
      
      attributes = reader.getAttributes();
      char * data;
      
      data = attributes.getAttribute( "x" );
      if (data) x = atof( data );
      
      data = attributes.getAttribute( "y" );
      if (data) y = atof( data );
      
      data = attributes.getAttribute( "z" );
      if (data) z = atof( data );
      
      g->setEyePoint( x, y, z );
    }
    else if ( !strcasecmp( name, "lookAtPoint" ) ) {
      double x=-1, y=-1, z=-1;
      
      attributes = reader.getAttributes();
      char * data;
      
      data = attributes.getAttribute( "x" );
      if (data) x = atof( data );
      
      data = attributes.getAttribute( "y" );
      if (data) y = atof( data );
      
      data = attributes.getAttribute( "z" );
      if (data) z = atof( data );
      
      g->setLookAtPoint( x, y, z );
    }
    else if ( !strcasecmp( name, "upVector" ) ) {
      double x=-1, y=-1, z=-1;
      
      attributes = reader.getAttributes();
      char * data;
      
      data = attributes.getAttribute( "x" );
      if (data) x = atof( data );
      
      data = attributes.getAttribute( "y" );
      if (data) y = atof( data );
      
      data = attributes.getAttribute( "z" );
      if (data) z = atof( data );
      
      g->setUpVector( x, y, z );
    }
    else if ( !strcasecmp( name, "viewmatrix" ) ) {
      g->setMatrixSet( true );
    } 
    else if ( !strcasecmp( name, "row" ) ) {
      int row=0;

      attributes = reader.getAttributes();
      char * data;
      
      data = attributes.getAttribute( "row" );
      if ( !data ) continue;
      row = atoi( data );
      if ( row > 3 ) row = 3;
      if ( row < 0 ) row = 0;

      data = attributes.getAttribute( "x1" );
      if ( data ) matrix[ row*4 + 0 ] = atof( data );

      data = attributes.getAttribute( "x2" );
      if ( data ) matrix[ row*4 + 1 ] = atof( data );

      data = attributes.getAttribute( "x3" );
      if ( data ) matrix[ row*4 + 2 ] = atof( data );

      data = attributes.getAttribute( "x4" );
      if ( data ) matrix[ row*4 + 3 ] = atof( data );
    }
    
    element = reader.nextElement();
  }

  if ( g->isMatrixSet() )
    g->setMatrix( matrix );
  
  return g;
  
}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:18  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:02  simpson
// Adding CollabVis files/dirs
//
// Revision 1.3  2001/08/29 19:58:05  luke
// More work done on ZTex
//
// Revision 1.2  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.1  2001/05/11 20:06:03  luke
// Initial coding of Message abstraction. Documentation not yet done.
//
