#include <Message/Collaborate.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {

Collaborate::Collaborate( bool request ) : request(request) {

}


Collaborate::~Collaborate() {
}

bool 
Collaborate::exists( string ID ) {
  for ( int i = 0; i < (int)drawings.size(); i++ )
    if ( drawings[i].ID == ID )
      return true;
  return false;
}

/** Returns an int from a string in the attributes */
int  
Collaborate::getIntFromString( const string param, String element,
			       Attributes attributes ) {
  string tmp;
  tmp = attributes.getAttribute( param );
  if ( tmp.empty() ) {
    Log::log( ERROR, "No " + param + " specified in " +
	      XMLI::getChar(element) );
    return -1;
  }
  return atoi( tmp );
}

/** Returns a double from a string in the attributes */
double  
Collaborate::getDoubleFromString( const string param, String element,
				  Attributes attributes ) {
  string tmp;
  tmp = attributes.getAttribute( param );
  if ( tmp.empty() ) {
    Log::log( ERROR, "No " + param + " specified in " +
	      XMLI::getChar(element)  );
    return -1;
  }
  return atof( tmp );
}

void 
Collaborate::addPointer( string ID, bool erase, float x, float y, float z,
			 double theta, int width, Color color ) {
  PointerData *pd = scinew PointerData( ID, erase, x, y, z, theta,
					width, color );
  pointers.push_back( *pd );
}

void 
Collaborate::addText( string ID, bool erase, float x, float y, string _text, 
		      int size, Color color ) {
  TextData *td = scinew TextData( ID, erase, x, y, _text, size, color );
  text.push_back( *td ); 
}


void 
Collaborate::addDrawing( string ID, bool erase, int width,
			 Color color ) {
  if ( !exists( ID ) ) {
    DrawingData *dd = scinew DrawingData( ID, erase, width, color );
    drawings.push_back( *dd );
  }
}

void 
Collaborate::addDrawingSegment( string ID, float x, float y, float z ) {
  for ( unsigned i = 0; i < drawings.size(); i++ )
    if ( drawings[i].ID == ID )
      drawings[i].addSegment( x, y, z );
}

PointerData*
Collaborate::getPointer( int index ) {
  if ( index >= (int)pointers.size() ) return NULL;
  return &pointers[index];
}

TextData* 
Collaborate::getText( int index ) {
  if ( index >= (int)text.size() ) return NULL;
  return &text[index];
}

DrawingData*
Collaborate::getDrawing( int index ) {
  if ( index >= (int)drawings.size() ) return NULL;
  return &drawings[index];
}


void 
Collaborate::finish() {

  if ( finished )
    return;
  
  /* Create an XML XDisplay document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  Attributes attributes;

  writer.addElement( "collaborate", attributes, String(0) );
  
  // Add pointers
  Attributes *attrs;
  int i;
  PointerData *p = NULL;
  for ( i = 0; i < (int)pointers.size(); i++ ) {
    p = &pointers[i];
    attrs = scinew Attributes();
    attrs->clear();
    attrs->setAttribute("id", p->ID );
    if ( p->erase )
      attrs->setAttribute("erase", "true" );
    else 
      attrs->setAttribute("erase", "false" );
    attrs->setAttribute( "x", mkString(p->x) );
    attrs->setAttribute( "y", mkString(p->y) );
    attrs->setAttribute( "z", mkString(p->z) );
    attrs->setAttribute( "theta", mkString(p->theta) );
    attrs->setAttribute( "width", mkString(p->width) );
    attrs->setAttribute( "red", mkString(p->color.getRed()) );
    attrs->setAttribute( "green", 
			    mkString(p->color.getGreen()) );
    attrs->setAttribute( "blue", mkString(p->color.getBlue()) );
    
    writer.addElement( "pointer", *attrs, String(0) );
  }

  // Add text
  TextData *t = NULL;
  for ( i = 0; i < (int)text.size() ; i++ ) {
    t = &text[i];
    attrs = new Attributes();
    attrs->clear();
    attrs->setAttribute("id", t->ID );
    if ( t->erase )
      attrs->setAttribute("erase", "true" );
    else 
      attrs->setAttribute("erase", "false" );
    attrs->setAttribute( "x", mkString(t->x) );
    attrs->setAttribute( "y", mkString(t->y) );
    attrs->setAttribute( "size", mkString(t->size) );
    attrs->setAttribute( "red", mkString(t->color.getRed()) );
    attrs->setAttribute( "green", 
			    mkString(t->color.getGreen()) );
    attrs->setAttribute( "blue", mkString(t->color.getBlue()) );
    
    
    writer.addElement( "text", *attrs, t->_text );
  }
  
  // Add drawings
  DrawingData *d = NULL;
  for ( i = 0 ; i < (int)drawings.size(); i++ ) {
    d = &drawings[i];
    attrs = scinew Attributes();
    attrs->clear();
    attrs->setAttribute("id", d->ID );
    if ( d->erase )
      attrs->setAttribute("erase", "true" );
    else 
      attrs->setAttribute("erase", "false" );
    attrs->setAttribute( "width", mkString(d->width) );
    attrs->setAttribute( "red", mkString(d->color.getRed()) );
    attrs->setAttribute( "green", 
			    mkString(d->color.getGreen()) );
    attrs->setAttribute( "blue", mkString(d->color.getBlue()) );
    
    
    writer.addElement( "drawing", *attrs, String(0) );
    writer.push();
    for ( int j = 0; j < d->numSegments(); j++ ) {
      attrs = scinew Attributes();
      attrs->clear();
      Point3d p = d->getSegment( j );
      attrs->setAttribute( "x", mkString(p.x) );
      attrs->setAttribute( "y", mkString(p.y) );
      attrs->setAttribute( "z", mkString(p.z) );
      writer.addElement( "segment", *attrs, String(0) );
    } 
    writer.pop();
  }
  
  mkOutput( writer.writeOutputData() );
  finished = true;
}


Collaborate *  
Collaborate::mkCollaborate( void * data ) {

  Collaborate * c = NULL;

  // Make an XML Reader
  XMLReader reader( (char *)data );

  reader.parseInputData();

  // Start building a Collaborate message
  c = scinew Collaborate( true );

  String element;
  Attributes attributes;
  string ID;
  string tmp;
  bool   erase;
  float x=0, y=0, z=0; 
  int r=0, g=0, b=0;
  int size=0;
  double theta=0;
  string text;
  
  element = reader.nextElement(); // Skip the header  
  if ( element == NULL )
    return NULL;
  element = reader.nextElement();       // Now we're on our first element.
  if ( element == NULL )
    return NULL;

  while ( element != NULL ) {
    attributes = reader.getAttributes();
    
    // ID
    ID = attributes.getAttribute( "id" );
    if ( ID.empty() ) {
      Log::log( ERROR, "No ID specified in " + XMLI::getChar(element) );
      element = reader.nextElement();
      continue;
    }
    
    // Erase
    tmp = attributes.getAttribute( "erase" );
    if ( tmp.empty() ) {
      Log::log( WARNING,
		"No erase parameter specified; assuming an addition." );
      erase = false;
    }
    else {
      if ( tmp ==  "true" )
	erase = true;
      else if ( tmp == "false" )
	erase = false;
      else {
	Log::log( WARNING, "Bad parameter to erase. Assuming an addition." ); 
	erase = false;
      }
    }

    // Color
    r = getIntFromString("red", element, attributes );
    g = getIntFromString("green", element, attributes );
    b = getIntFromString("blue", element, attributes );

    // Now, we go by specific annotations
    if ( XMLI::getChar(element) == "pointer" ) {
      x = getDoubleFromString( "x", element, attributes );
      y = getDoubleFromString( "y", element, attributes );
      z = getDoubleFromString( "z", element, attributes );
      theta = getDoubleFromString( "theta", element, attributes );
      size = getIntFromString( "width", element, attributes );
	  
      c->addPointer( ID, erase, x, y, z, theta, size, Color(r,g,b) );
    }  
    else if ( XMLI::getChar(element) ==  "text" ) {
      x = getDoubleFromString( "x", element, attributes );
      y = getDoubleFromString( "y", element, attributes );
      size = getIntFromString( "size", element, attributes );
      text = XMLI::getChar(reader.getText());
      if ( text.empty() ) {
	Log::log( ERROR, "No text present in text annotation!" );
	element = reader.nextElement();
	continue;
      }
      c->addText( ID, erase, x, y, text, size, Color(r,g,b) );
    }
    else if ( XMLI::getChar(element) == "drawing" ) {
	  
      size = getIntFromString( "width", element, attributes );
      c->addDrawing( ID, erase, size, Color(r,g,b) );
      
      // Parse the segments
      element = reader.nextElement();
      while ( XMLI::getChar(element) == "segment" ) {
	attributes = reader.getAttributes();
	x = getDoubleFromString( "x", element, attributes );
	y = getDoubleFromString( "y", element, attributes );
	z = getDoubleFromString( "z", element, attributes );
	c->addDrawingSegment( ID, x, y, z );
	element = reader.nextElement();
	if ( element == NULL ) return c;
      }
    }
    else {
      Log::log( ERROR, "Unknown annotation " + XMLI::getChar(element) );
    }
    element = reader.nextElement();
  }
  return c;
}


}
//
// $Log$
// Revision 1.1  2003/07/22 20:59:25  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:08  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/09/25 14:44:58  luke
// Got collaboration working
//
// Revision 1.1  2001/09/23 02:24:11  luke
// Added collaborate message
//

