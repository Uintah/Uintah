/*
 *
 * Collaborate: Message that encapsulates a collaborate message.
 *
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#ifndef __COLLABORATE_H_
#define __COLLABORATE_H_

#include <vector>

#include <Logging/Log.h>
#include <Message/MessageBase.h>
#include <Util/Color.h>
#include <Util/Point.h>

namespace SemotusVisum {

/**
 * This class encapsulates all the data needed for a pointer annotation.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class PointerData {
public:
  /**
   *  Constructor.
   *
   * @param ID       ID for the pointer.
   * @param erase    True if this pointer should be removed.
   * @param x        X coordinate
   * @param y        Y coordinate
   * @param z        Z coordinate
   * @param theta    Rotation around Z axis
   * @param width    Width of the pointer lines
   * @param color    Color of the pointer
   */
  PointerData( string ID, bool erase, float x, float y, float z,
	       double theta, int width, Color color ) : ID(ID),
							erase(erase),
							x(x),
							y(y), z(z),
							theta(theta),
							width(width),
							color(color)
  {}
  
  /**
   *  Destructor
   *
   */
  ~PointerData() {}

  /**
   * Returns a string representation of the pointer.
   *
   * @return String rep of the pointer.
   */
  inline string output() {
    return "ID: " + ID + " erase = " + mkString(erase) + " X/Y/Z = " +
      mkString(x) + " " + mkString(y) + " " + mkString(z) + " Theta = " +
      mkString(theta) + " Width = " + mkString(width) + " Color = " +
      color.toString();
  }

  /** ID */
  string ID;

  /** True if the pointer is being removed */
  bool erase;

  /** Pointer coordinate */
  float x, y, z;

  /** Rotation around Z axis */
  double theta;

  /** Pointer line width */
  int width;

  /** Pointer color */
  Color color;
};

/**
 * This class encapsulates all the data needed for a text annotation.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class TextData {
public:
  /**
   *  Constructor.
   *
   * @param ID       ID for the text
   * @param erase    True if this text should be removed.
   * @param x        X coordinate
   * @param y        Y coordinate
   * @param text     Text of the annotation
   * @param size     Text size
   * @param color    Color of the text
   */
  TextData( string ID, bool erase, float x, float y, string text,
	    int size, Color color ) : ID(ID),
				      erase(erase),
				      x(x),
				      y(y),
				      _text(text),
				      size(size),
				      color(color)
  {}
  
  /**
   *  Destructor.
   *
   */
  ~TextData() {}
  
  /**
   * Returns a string representation of the text
   *
   * @return String rep of the text
   */
  inline string output() {
    return "ID: " + ID + " erase = " + mkString(erase) + " X/Y = " +
      mkString(x) + " " + mkString(y) + " Text = " + _text + " Size = " +
      mkString(size) + " Color = " + color.toString();
  }
  
  /** ID */
  string ID;
  
  /** True if the text is being removed */
  bool erase;

  /** Text coordinate */
  float x, y;

  /** Text of the annotation */
  string _text;

  /** Size of the text */
  int size;

  /** Text color */
  Color color;
};


/**
 * This class encapsulates all the data needed for a drawing annotation.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class DrawingData {
public:
  /**
   * Constructor.
   *
   * @param ID      Annotation ID
   * @param erase   True to erase annotation
   * @param width   Line width
   * @param color   Line color
   */
  DrawingData( string ID, bool erase, int width, Color color ) : ID(ID),
								 erase(erase),
								 width(width),
								 color(color)
  {}
  
  /**
   *  Destructor.
   *
   */
  ~DrawingData() {}
  
  /**
   *  Adds a segment to the drawing
   *
   * @param x     X coordinate
   * @param y     Y coordinate    
   * @param z     Z coordinate    
   */
  inline void addSegment( float x, float y, float z ) {
    segments.push_back( Point3d( x, y, z) );
  }
  
  /**
   * Returns the number of segments.
   *
   * @return Number of segments
   */
  inline int numSegments() const { return segments.size(); }

  
  /**
   * Returns the point at the given index.
   *
   * @param index    Index into the list of segments
   * @return         Point, or (-1,-1,-1) if not found.
   */
  inline Point3d getSegment( int index ) {
    if ( index > numSegments() ) return Point3d( -1,-1,-1 );
    return segments[index];
  }
  
  /**
   * Returns a string representation of the drawing.
   *
   * @return String rep of the drawing.
   */
  inline string output() {
    string returnval = "ID: " + ID + " erase = " + mkString(erase) +
      " Width = " + mkString(width) + " Color = " + color.toString() + "\n";
    for ( unsigned i = 0; i < segments.size(); i++ )
      returnval += "Segment " + mkString(i) + " : " + segments[i].toString() +
	"\n";
    return returnval;
  }

  /** ID */
  string ID;

  
  /** True if the drawing is being removed */
  bool erase;
  
  /** Text line width */
  int width;

  /** Text color */
  Color color;
  
protected:
  
  /** List of segments */
  vector<Point3d> segments;
  
};

/**
 * This class provides the infrastructure to create, read, and serialize
 *  a Collaborate message.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Collaborate : public MessageBase {
public:

  /**
   *  Constructor. By default all messages are incoming.
   *
   * @param request       
   */
  Collaborate( bool request = false );

  /**
   *  Destructor. Deallocates all memory.
   *
   */
  ~Collaborate();

  /**
   *  Adds a pointer to the message.
   *
   * @param ID       ID for the pointer.
   * @param erase    True if this pointer should be removed.
   * @param x        X coordinate
   * @param y        Y coordinate
   * @param z        Z coordinate
   * @param theta    Rotation around Z axis
   * @param width    Width of the pointer lines
   * @param color    Color of the pointer
   */
  void addPointer( string ID, bool erase, float x, float y, float z,
		   double theta, int width, Color color );
  /**
   *  Adds text to the message
   *
   * @param ID       ID for the text
   * @param erase    True if this text should be removed.
   * @param x        X coordinate
   * @param y        Y coordinate
   * @param text     Text of the annotation
   * @param size     Text size
   * @param color    Color of the text
   */
  void addText( string ID, bool erase, float x, float y, string text, 
		int size, Color color );
  
  /**
   *  Adds a drawing to the message
   *
   * @param ID      Annotation ID
   * @param erase   True to erase annotation
   * @param width   Line width
   * @param color   Line color
   */
  void addDrawing( string ID, bool erase, int width, Color color );

  /**
   * Adds a drawing segment to the drawing with the given ID
   *
   * @param ID    Drawing ID
   * @param x     X coordinate of segment
   * @param y     Y coordinate of segment   
   * @param z     Z coordinate of segment    
   */
  void addDrawingSegment( string ID, float x, float y, float z );
  
  /**
   * Returns the number of pointers
   *
   * @return Number of pointers
   */
  inline int  numPointers() const { return pointers.size(); }

  
  /**
   * Returns the number of text annotations
   *
   * @return Number of text annotations
   */
  inline int  numText() const { return text.size(); }

  /**
   * Returns the number of drawings
   *
   * @return Number of drawings
   */
  inline int  numDrawings() const { return drawings.size(); }
  
  /**
   * Returns the pointer at the given index in the list of pointers
   *
   * @param index   Index of pointer
   * @return        Pointer, or NULL if index is out of range.
   */
  PointerData * getPointer( int index );
  
  /**
   * Returns the text at the given index in the list of text
   *
   * @param index    Index of text
   * @return         Text, or NULL if index is out of range.
   */
  TextData *    getText( int index );

  
  /**
   * Returns the drawing at the given index in the list of drawings
   *
   * @param index    Index of drawing
   * @return         Drawing, or NULL if index is out of range.
   */
  DrawingData * getDrawing( int index );
  
  /**
   *  Finishes serializing the message.
   *
   */
  void finish();
  
  /**
   *  Returns true if this is a request; else returns false.
   *
   * @return True if a request (inbound); false if this is outbound.
   */
  inline bool isRequest() const { return request; }

  /**
   *  Returns a Collaborate message from the given raw data.
   *
   * @param data    Raw data
   * @return        New message, or NULL on error
   */
  static Collaborate * mkCollaborate( void * data );
  
protected:

  /** Returns true if the annotation with ID exists in the message */
  bool exists( string ID );

  /** Returns an int from a string in the attributes */
  static int getIntFromString( const string param, String element,
			       Attributes attributes );
  
  /** Returns a double from a string in the attributes */
  static double getDoubleFromString( const string param, String element,
				     Attributes attributes );
  
  /** True if this message is a request. */
  bool     request;

  /** List for pointers */
  vector<PointerData> pointers;

  /** List for text */
  vector<TextData>    text;

  /** List for drawings */
  vector<DrawingData> drawings;
};


}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:25  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:09  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/09/25 14:44:58  luke
// Got collaboration working
//
// Revision 1.1  2001/09/23 02:24:11  luke
// Added collaborate message
//

