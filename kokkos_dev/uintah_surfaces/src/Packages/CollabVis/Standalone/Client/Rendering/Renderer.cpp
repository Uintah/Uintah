/*
 *
 * Renderer: Interface for various rendering classes
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: April 2001
 *
 */

#include <Rendering/Renderer.h>
#include <Network/NetInterface.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <UI/UserInterface.h>
#include <UI/uiHelper.h>

namespace SemotusVisum {

int
CollaborationItem::lastID = 0;

void
CollaborationItem::draw()  {
  cerr << "In CollaborationItem::draw\n";

  if ( !enabled ) {
    Log::log( DEBUG, "Not drawing annotation" );
    return;
  }
  
  Log::log( DEBUG, "Drawing annotation: " + toString() );
  
  // Set the color
  glColor3fv( getColor().array() );
    
  // Switch based on the type of the annotation
  switch ( getType() ) {
  case CollaborationItem::POINTER: {
    
    // Do lines
    glBegin( GL_LINES );
  
    // Draw the main part of the arrow using the coords and Z rotation
    float x1 = getX();
    float y1 = getY();
    float x2 = x1 + (pointerSize * cos( getZRotation() ) );
    float y2 = y1 + (pointerSize * sin( getZRotation() ) );
    glVertex3f( x1, y1, getZ() ); glVertex3f( x2, y2, getZ() );
    
    // Draw one part of the pointer
    x2 = x1 + (pointerSize/4.0 * cos( getZRotation() - M_PI_4 ) );
    y2 = y1 + (pointerSize/4.0 * sin( getZRotation() - M_PI_4 ) );
    glVertex3f( x1, y1, getZ() ); glVertex3f( x2, y2, getZ() );
    
    // Draw the other part
    x2 = x1 + (pointerSize/4.0 * cos( getZRotation() + M_PI_4 ) );
    y2 = y1 + (pointerSize/4.0 * sin( getZRotation() + M_PI_4 ) );
    glVertex3f( x1, y1, getZ() ); glVertex3f( x2, y2, getZ() );
    
    glEnd();
  }
    break;
  case CollaborationItem::TEXT:

    // Draw the string
    drawString( getText(), getX(), getY() );
    
    break;
  case CollaborationItem::DRAWING: {

    // If we have only one point, we can't draw a line!
    if ( segments.size() < 2 )
      return;

    // Do lines
    glBegin( GL_LINE_STRIP );

    // Do vertices
    for ( unsigned i = 0; i < segments.size(); i++ )
      glVertex2f( segments[i].x, segments[i].y );

    glEnd();
  }
    break;
  default:
    Log::log( ERROR, "Unknown type of collaboration item: " + toString() );
  }
  cerr << "End of CollaborationItem::draw\n";
}

/** Draws a string at location (X,Y) */
void
CollaborationItem::drawString( const string& s, const int x, const int y ) {
  cerr << "In CollaborationItem::drawString\n";
  glRasterPos2i( x, y );
  for ( unsigned i = 0; i < s.length(); i++ )
    glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, s[i] );
  glRasterPos2i( 0, 0 );
  cerr << "End of CollaborationItem::drawString\n";
}
 



const char * const
Renderer::shading[] = { "flat", 
			"gouraud", 
			"wireframe",
			0 };

const string
Renderer::name = "Renderer";
const string
Renderer::version = "1.0";

char *
Renderer::decompressFrame( char * data, const int offset, 
			   const int length, const int origSize ) {
  cerr << "In CollaborationItem::decompressFrame\n";
  char *outdata = NULL;
  int type;
  int outputBytes;
  
  /* If we need to decode the data (ie, it is compressed), do that */
  if ( compressionMode != NONE ) {
    Compressor * c = mkCompressor( compressionMode, type );
    if ( c == NULL ) {
      Log::log( ERROR, "Could not create a compressor! " );
      return NULL;
    }
    outdata = scinew char[ origSize ];
    Log::log( DEBUG, "Decompressing..." );
#if 0
    {
      char buffer[100];
      snprintf(buffer,100,"file%d.out",jj); jj++;
      FILE *f = fopen( buffer, "w");
      fwrite( data + offset, sizeof(char), length, f );
      fclose(f);
    }
#endif
    outputBytes = c->decompress( (DATA *)(data + offset),
				 length,
				 ( DATA ** )&outdata );
    Log::log( DEBUG, "Decompressed into " + mkString(outputBytes) + " bytes");
    if ( outputBytes != origSize ) {
      Log::log( ERROR, "Error decompressing data: Wanted " +
		mkString( origSize ) + " output bytes, got " +
		mkString( outputBytes ) );
      delete outdata;
      delete c;
      return NULL;
    }
    delete c;
  }
  else {
    outdata = scinew char[ length ];
    memcpy( data + offset, outdata, length );
  }
  cerr << "End of CollaborationItem::decompressFrame\n";
  return outdata;
}

/** 
   * Sets the new compression mode based on the given string.
   *
   * @param          mode           String name for compression.
   */
void  
Renderer::setCompressionMode( const string &mode ) {
  cerr << "In Renderer::setCompressionMode\n";
  Log::log( DEBUG, "Trying to set compression to mode " + mode );
  for ( int i = 0; i < NUM_COMPRESSORS; i++ )
    if ( !strcasecmp( mode, compressionMethods[ i ] ) ) {
      compressionMode = i;
      //ejlglutUI::getInstance().setCompression( mode );
      UserInterface::getHelper()->setCompression( mode );
      return;
    }
  
  // Special case - no compression.    
  if ( !strcasecmp( mode, "None" ) ) {
    //ejlglutUI::getInstance().setCompression( "" );
    UserInterface::getHelper()->setCompression( mode );
    compressionMode = NONE;
    return;
  }
  
  Log::log( ERROR, 
	    "Bug - setting compression to an unknown mode: " + mode );
  cerr << "End of Renderer::setCompressionMode\n";
}

/**
   * Adds a pointer at the given location, with the given rotations around
   * the two relevant axes.
   *
   * @param    point        Location of pointer.
   * @param    zRotation    Rotation in radians around the Z axis.
   *
   * @return                An ID for this pointer.
   */
string 
Renderer::addPointer( const Point3d point, const double zRotation ) {
    cerr << "In Renderer::addPointer\n";
    CollaborationItem ci( CollaborationItem::POINTER, 
			  point, zRotation );
    ci.setColor( pointerColor );
    
    collaborationItemList.push_back( ci );

    /* Send pointer to network */
    Collaborate *c = scinew Collaborate( true );
    c->addPointer( ci.getID(), false, point.x, point.y, point.z,
		   zRotation, lineWidth, pointerColor );
    NetInterface::getInstance().sendDataToServer( c );
    
    cerr << "End of Renderer::addPointer\n";
    return ci.getID();
  }

  /**
   * Adds a pointer at the given location, with the given rotations around
   * the two relevant axes, with the given ID.
   *
   * @param    x            X coordinate of tip of pointer.
   * @param    y            Y coordinate of tip of pointer.
   * @param    zRotation    Rotation in radians around the Z axis.
   * @param    ID           ID for this pointer.
   * @param    width        Line width for the pointer.
   * @param    color        Color for the pointer.
   *
   * @return                An ID for this pointer.
   */
string 
Renderer::addPointer( const Point3d point, const double zRotation,
		      string ID,
		      const int width, const Color color ) {
  cerr << "In Renderer::addPointer\n";
  string name = addPointer(point, zRotation );
  CollaborationItem *ci = getID(name);
  ci->setID( ID );
  ci->setLineWidth( width );
  ci->setColor( color );
  
  cerr << "End of Renderer::addPointer\n";
  return ID;
}

/**
   * Adds the text at the given coordinates.
   *
   * @param     point      Location of the text
   * @param     text       Text to add.
   *
   * @return               An ID for this text.
   */
string  
Renderer::addText( const Point3d point, const string &text ) {
  cerr << "In Renderer::addText\n";
  CollaborationItem ci( CollaborationItem::TEXT, point );
  
  ci.setText( text );
  ci.setColor( textColor );
  ci.setTextSize( textSize );
  collaborationItemList.push_back( ci );

  /* Send text to network */
  Collaborate *c = scinew Collaborate( true );
  c->addText( ci.getID(), false, point.x, point.y, text,
	      textSize, pointerColor );
  NetInterface::getInstance().sendDataToServer( c );
  
  cerr << "End of Renderer::addText\n";
  return ci.getID();
}
  
  /**
   * Adds the text at the given coordinates with the given ID.
   *
   * @param     x          X coordinate of midpoint of text.
   * @param     y          Y coordinate of midpoint of text.
   * @param     text       Text to add.
   * @param     ID         ID for this text.
   * @param     size       Text size.
   * @param     color      Text color.
   *
   * @return               An ID for this text.
   */
string  
Renderer::addText( const Point3d point, const string &text,
		   const string ID, const int size,
		   const Color color ) {
    cerr << "In Renderer::addText\n";
    string name = addText( point, text );
    CollaborationItem *ci = getID( name );
    ci->setID( ID );
    ci->setColor( color );
    ci->setTextSize( size );

    cerr << "End of Renderer::addText\n";
    return ID;
  }

/**
   * Creates a new drawing using the current drawing and line width
   * parameters. Returns an ID for the drawing.
   *
   * @return     An ID for the drawing.
   */
string  
Renderer::newDrawing() {
  cerr << "In Renderer::newDrawing\n";
  CollaborationItem ci( CollaborationItem::DRAWING );
  
  ci.setColor( drawingColor );
  ci.setLineWidth( lineWidth );
  drawingMode = true; 
  collaborationItemList.push_back( ci );
  cerr << "End of Renderer::newDrawing\n";
  return ci.getID();
}
  
/**
 * Creates a new drawing using the current drawing and line width
 * parameters. Returns an ID for the drawing.
 *
 * @param      ID        The ID for the drawing.
 *
 * @return     An ID for the drawing.
 */
string  
Renderer::newDrawing( const string ID, const int width,
		      const Color color ) {
  cerr << "In Renderer::newDrawing\n";
  string name = newDrawing();
  CollaborationItem *ci = getID(name);
  ci->setID(ID);
  ci->setColor(color);
  ci->setLineWidth(width);
  
  cerr << "End of Renderer::newDrawing\n";
  return ID;
}

/**
 * Starts/stops freehand drawing mode. 
 *
 * @param     start       True to start drawing; false to stop drawing.
 * 
 * @return                An ID for the current freehand drawing if we're
 *                        just starting to draw. Else, null.
 */
string  
Renderer::drawMode( const bool start ) {
  cerr << "In Renderer::drawMode\n";
  
  // If we've just started, create an item.
  if ( start && !drawingMode ) {
    return newDrawing();
  }
  // If we're drawing, and recieve a stop, note that.
  else if ( !start && drawingMode )
    drawingMode = false;
  
  // Otherwise, we ignore our input.

  cerr << "End of Renderer::drawMode\n";
  return "";
}

/**
 * Adds another segment to the drawing with the given ID.
 *
 * @param        point        Location of the next line segment.
 * @param        ID           Which drawing to add the segment to.
 */
void  
Renderer::addDrawingSegment( const Point3d point, const string &ID ) { 
  cerr << "In Renderer::addDrawingSegment\n";
  CollaborationItem *ci = getID( ID );
  
  if ( ci == NULL ) {
    Log::log( WARNING, "Cannot add point to drawing " + ID +
	      " - ID not found." );
    return;
  }
  else
    Log::log( DEBUG, "Adding point " + point.toString() + " to drawing " +
	      ID );
  ci->addSegmentPoint( point );
  //System.out.println( "Adding a point " + point + " to drawing " + ID );

  cerr << "End of Renderer::addDrawingSegment\n";
}

/**
   * Finishes the drawing, and sends it to the server...
   *
   * @param       ID            ID of the drawing
   */
void   
Renderer::finishDrawing( const string &ID ) {
  cerr << "In Renderer::finishDrawing\n";
  CollaborationItem *ci = getID( ID );

  if ( ci == NULL )
    return;

  Collaborate *c = scinew Collaborate( false );

  c->addDrawing( ID, false, lineWidth, drawingColor );

  vector<Point3d>& segments = ci->getSegments();
    for ( unsigned i = 0; i < segments.size(); i++ )
    c->addDrawingSegment( ID, segments[i].x, segments[i].y,
			  segments[i].z );
  NetInterface::getInstance().sendDataToServer( c );
  cerr << "End of Renderer::finishDrawing\n";  
  
}

void    
Renderer::drawFPS( const bool lighting ) {
  cerr << "In Renderer::drawFPS\n";
  /* FIXME - remove this! */
  cerr << "End of Renderer::drawFPS\n";
}

vector<string>     
Renderer::getAnnotationIDs() {
  cerr << "In Renderer::getAnnotationIDs\n";
  vector<string> ids( collaborationItemList.size() );

  for ( unsigned j = 0; j < collaborationItemList.size(); j++ ) {
    if ( collaborationItemList[j].getType() ==
	 CollaborationItem::POINTER )
      ids[j] = "P" + collaborationItemList[j].getID();
    else if ( collaborationItemList[j].getType() ==
	      CollaborationItem::TEXT )
      ids[j] = "T" + collaborationItemList[j].getID();
    else if ( collaborationItemList[j].getType() ==
	      CollaborationItem::DRAWING )
      ids[j] = "D" + collaborationItemList[j].getID();
  }
  cerr << "End of Renderer::getAnnotationIDs\n";
  return ids;
}

}
