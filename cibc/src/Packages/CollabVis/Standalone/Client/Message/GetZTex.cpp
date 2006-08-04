#include <Message/GetZTex.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {

GetZTex::GetZTex() : matrix( NULL ), matrixSet( false ), matrixOurs( true ) {
}

GetZTex::~GetZTex() {
  if ( matrixOurs )
    delete[] matrix;
}

void
GetZTex::finish() {
  if ( finished )
    return;
  
  /* Create an XML getclientlist document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  // Create a 'getztex' document
  Attributes attributes;

  writer.addElement("GetZTex" );
  if ( isMatrixSet() ) {
    writer.addElement("viewmatrix" );
    writer.push();

    for ( int i = 0 ; i < 4; i++ ) {
      attributes.clear();
      
      attributes.setAttribute("row", mkString( i ) );
      attributes.setAttribute("x1", mkString( matrix[ 0 + i*4 ] ) );
      attributes.setAttribute("x2", mkString( matrix[ 1 + i*4 ] ) );
      attributes.setAttribute("x3", mkString( matrix[ 2 + i*4 ] ) );
      attributes.setAttribute("x4", mkString( matrix[ 3 + i*4 ] ) );

      writer.addElement( "row", attributes, String(0) );
    }

    writer.pop();
  }
  // Eye point
  if ( eyeSet ) {
    attributes.clear();
    attributes.setAttribute( "x", mkString( eyeX ) );
    attributes.setAttribute( "y", mkString( eyeY ) );
    attributes.setAttribute( "z", mkString( eyeZ ) );
    
    writer.addElement( "eyePoint", attributes, String(0) );
  }
  
  // Look at point
  if ( atSet ) {
    attributes.clear();
    attributes.setAttribute( "x", mkString( atX ) );
    attributes.setAttribute( "y", mkString( atY ) );
    attributes.setAttribute( "z", mkString( atZ ) );
    
    writer.addElement( "lookAtPoint", attributes, String(0) );
  }
  
  // Up vector
  if ( upSet ) {
    attributes.clear();
    attributes.setAttribute( "x", mkString( upX ) );
    attributes.setAttribute( "y", mkString( upY ) );
    attributes.setAttribute( "z", mkString( upZ ) );
    
    writer.addElement( "upVector", attributes, String(0) );
  }
  
  mkOutput( writer.writeOutputData() );
  finished = true;
}

void
GetZTex::setTransform( const double * newmatrix ) {
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
  string name;
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
      string data;
      
      data = attributes.getAttribute( "x" );
      if (!data.empty()) x = atof( data.data() );
      
      data = attributes.getAttribute( "y" );
      if (!data.empty()) y = atof( data.data() );
      
      data = attributes.getAttribute( "z" );
      if (!data.empty()) z = atof( data.data() );
      
      g->setEyePoint( x, y, z );
    }
    else if ( !strcasecmp( name, "lookAtPoint" ) ) {
      double x=-1, y=-1, z=-1;
      
      attributes = reader.getAttributes();
      string data;
      
      data = attributes.getAttribute( "x" );
      if (!data.empty()) x = atof( data.data() );
      
      data = attributes.getAttribute( "y" );
      if (!data.empty()) y = atof( data.data() );
      
      data = attributes.getAttribute( "z" );
      if (!data.empty()) z = atof( data.data() );
      
      g->setAtPoint( x, y, z );
    }
    else if ( !strcasecmp( name, "upVector" ) ) {
      double x=-1, y=-1, z=-1;
      
      attributes = reader.getAttributes();
      string data;
      
      data = attributes.getAttribute( "x" );
      if (!data.empty()) x = atof( data.data() );
      
      data = attributes.getAttribute( "y" );
      if (!data.empty()) y = atof( data.data() );
      
      data = attributes.getAttribute( "z" );
      if (!data.empty()) z = atof( data.data() );
      
      g->setUpVector( x, y, z );
    }
    else if ( !strcasecmp( name, "viewmatrix" ) ) {
      g->setMatrixSet( true );
    } 
    else if ( !strcasecmp( name, "row" ) ) {
      int row=0;

      attributes = reader.getAttributes();
      string data;
      
      data = attributes.getAttribute( "row" );
      if ( data.empty() ) continue;
      row = atoi( data.data() );
      if ( row > 4 ) row = 4;
      if ( row < 1 ) row = 1;

      data = attributes.getAttribute( "x1" );
      if (! data.empty() ) matrix[ (row-1)*4 + 0 ] = atof( data.data() );

      data = attributes.getAttribute( "x2" );
      if ( !data.empty() ) matrix[ (row-1)*4 + 1 ] = atof( data.data() );

      data = attributes.getAttribute( "x3" );
      if ( !data.empty() ) matrix[ (row-1)*4 + 2 ] = atof( data.data() );

      data = attributes.getAttribute( "x4" );
      if ( !data.empty() ) matrix[ (row-1)*4 + 3 ] = atof( data.data() );
    }
    
    element = reader.nextElement();
  }

  if ( g->isMatrixSet() )
    g->setTransform( matrix );
  
  return g;
  
}

}
//
// $Log$
// Revision 1.1  2003/07/22 20:59:26  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:10  simpson
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
