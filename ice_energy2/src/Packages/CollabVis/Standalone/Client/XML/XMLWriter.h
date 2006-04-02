/*
 *
 * XMLWriter: Creates XML Documents
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */

#ifndef __XMLWriter_h_
#define __XMLWriter_h_

#include <map>
#include <stack>
#include <iterator>
#include <strstream>

#ifdef __sgi
#pragma set woff 1375
#pragma set woff 3303
#pragma set woff 3201
#pragma set woff 1424
#endif

#include <xercesc/dom/deprecated/DOMString.hpp>
#include <xercesc/dom/deprecated/DOM_NamedNodeMap.hpp>
#include <xercesc/dom/deprecated/DOM_Attr.hpp>
#include <xercesc/dom/deprecated/DOM_Document.hpp>

#ifdef __sgi
#pragma reset woff 1375
#pragma reset woff 3303
#pragma reset woff 3201
#pragma reset woff 1424
#endif

#include <XML/XML.h>
#include <XML/XMLElement.h>
#include <XML/Attributes.h>

#define WRITE( target, text ) (target).write( (text), strlen( (text) ) ) 

namespace SemotusVisum {

/**
 * XMLWriter provides a convenient and implementation-independent
 * way to create and populate an XML document.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class XMLWriter {
  
  /**
   * Overloaded output operator that takes in a node and writes it
   * to the stream.
   *
   * @param target    Input target
   * @param toWrite   Node to output
   *
   * @return          Stream with node appended.
   */
  friend strstream& operator<<(strstream& target, const DOM_Node& toWrite);

  /**
   * Overloaded output operator that takes in a DOMString and writes it
   * to the stream.
   *
   * @param target    Input target
   * @param toWrite   String to output
   *
   * @return          Stream with string appended.
   */
  friend strstream& operator<<(strstream& target, const DOMString& toWrite);
  
public:

  /**
   * Default constructor.
   *
   */
  XMLWriter();


  /**
   * Destructor
   *
   */
  ~XMLWriter();

  /**
   * Adds a full XML Element to the current document.
   *
   * @param e     Element to add
   */
  void         addElement(XMLElement e);

  /**
   *  Adds an element to the current document using DOMStrings
   *
   * @param elementName    Name of the element
   * @param attributes     Element attributes
   * @param text           Element text, if any
   */
  void         addElement(String  elementName,
			  Attributes attributes,
			  String text);
  
  /**
   * Adds an element to the current document using std strings
   *
   * @param elementName    Name of the element 
   * @param attributes     Element attributes    
   * @param text           Element text, if any  
   */
  void         addElement(string elementName,
			  Attributes attributes,
			  string text) {
    char * buffer = toChar( elementName );
    String s = buffer;
    delete buffer;
    buffer = toChar( text );
    String t = buffer;
    delete buffer;
    addElement( s,
		attributes,
		t );
    
  }

  /**
   * Adds an element to the current document using char pointers
   *
   * @param elementName    Name of the element 
   * @param attributes     Element attributes    
   * @param text           Element text, if any  
   */
  void         addElement(char * elementName,
			  Attributes attributes,
			  char * text) {
    addElement( String(elementName),
		attributes,
		String(text) );
  }

  /**
   * Adds an element to the current document with no attributes or text.
   *
   * @param elementName   Element name.
   */
  void         addElement( string elementName ) {
    Attributes a;
    char * buffer = toChar( elementName );
    String s = buffer;
    addElement( s, a, String(0) );
    delete buffer;
  }
  
  /**
   * Creates a new document for this XML Writer. Must be called before
   * creating a new document.
   *
   */
  void         newDocument();

  /**
   *  Goes up a level in the XML hierarchy
   *
   */
  void         pop();

  /**
   *Pushes down a level in the XML hierarchy, so that all nodes added
   * after will be children of the last node added. <long-description>
   *
   */
  void         push();

  /**
   * Serializes the XML document and returns it in text format.
   *
   * @return   Textual version of the XML tree
   */
  string       writeOutputData();
  
protected:
  /// Tree structure
  stack<DOM_Node>         treeStack;

  /// Parent node (for pushing a level)
  DOM_Node                parentNode;

  /// Last node before pushing
  DOM_Node                lastNode;

  /// The low-level document
  DOM_Document            theDoc;


  /// DOM implementation object
  DOM_DOMImplementation   impl;

  /// Outputs the content of the string
  static void         outputContent(strstream& target,
				    const DOMString &toWrite);

  /// Outputs the content of the node, and appends it to start.
  static string dumpNode( string start, const DOM_Node& toWrite );
};

}
#endif

//
// $Log$
// Revision 1.1  2003/07/22 21:00:00  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:05:33  simpson
// Adding CollabVis files/dirs
//
