/*
 *
 * XMLReader: Parses XML Documents
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */

#ifndef __XMLReader_h_
#define __XMLReader_h_

#include <map>
#include <stack>

#ifdef __sgi
#pragma set woff 1375
#pragma set woff 3303
#pragma set woff 3201
#pragma set woff 1424
#endif

#include <xercesc/framework/MemBufInputSource.hpp>
#include <xercesc/dom/deprecated/DOMString.hpp>
#include <xercesc/dom/deprecated/DOM_Document.hpp>
#include <xercesc/dom/deprecated/DOM_NamedNodeMap.hpp>
#include <xercesc/dom/deprecated/DOM_Attr.hpp>
#include <xercesc/dom/deprecated/DOMParser.hpp>

#ifdef __sgi
#pragma reset woff 1375
#pragma reset woff 3303
#pragma reset woff 3201
#pragma reset woff 1424
#endif

#include <XML/XML.h>
#include <XML/Attributes.h>

namespace SemotusVisum {

/**************************************
 
CLASS
   XMLReader
   
KEYWORDS
   XML
   
DESCRIPTION

   XMLReader provides a convenient and implementation-independent
   way to parse and navigate an XML document.
   
****************************************/
/**
 * XMLReader provides a convenient and implementation-independent
 * way to parse and navigate an XML document.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class XMLReader {
  
public:

  /**
   * Default constructor.
   *
   */
  XMLReader();

  /**
   * Constructor that sets the input source for the XML document.
   *
   * @param input  XML input source
   */
  XMLReader( MemBufInputSource *input );

  /**
   * Constructor that sets the input data for the XML document.
   *
   * @param input   String XML input
   */
  XMLReader( string input );
  
  /**
   * Destructor
   *
   */
  ~XMLReader();

  /**
   * Returns the tag of the current element
   *
   * @return Tag of the current element
   */
  String        currentElement();

  /**
   *  Returns the list of attributes of this element
   *
   * @return  List of attributes of this element.
   */
  Attributes    getAttributes();

  /**
   * Returns any text associated with this element
   *
   * @return  Text of the current element
   */
  String        getText();

  /**
   * Returns true if the current node has any child elements
   *
   * @return    True if the current node has any child elements
   */
  bool          hasChildren();

  /**
   * Returns the tag of the next element in the document, or null if
   * there are no more elements.
   *
   * @return    Tag of next element, or NULL if there are no more elements
   */
  String        nextElement();

  /**
   * Parses the textual XML input. Must be called before retrieving
   * elements.
   *
   */
  void          parseInputData();

  /**
   * Pops up to the next level in the tree. Used to ignore children of
   * the current node.
   *
   */
  void          pop();

  /**
   * Sets the input XML source. Must be called (if not set in the
   * constructor) before parsing the document.
   *
   * @param input 
   */
  void          setInputData( MemBufInputSource *input);
  
protected:
  /// Input source
  MemBufInputSource *input;

  /// Low-level document
  DOM_Document       doc;

  /// Document parser
  DOMParser          parser;

  /// Current node
  DOM_Node           currentNode;

  /// Current index in node stack
  int                currentIndex;

  /// Stack of nodes (tree-style)
  stack<DOM_Node>    treeStack;

  /// Stack of indices of nodes at a given level
  stack<int>         indexStack;

  /// Attributes of current node
  Attributes         attributes;

  /// Text of current node
  String             text;

  /// True if we must dispose of the input source.
  bool               isInputOurs;
  

  /// Returns a count of the nodes in the node list.
  int  countElementNodes( DOM_NodeList nodes);


  /** Selects the next non-text element in the document. Returns true if
   *  there are more elements; else returns false.
   */
  bool getNextAvailableElement( bool recursed );

  /** Builds an internal list of the current element's attributes and
   *  associated text.
   */
  void makeTextAndAttributes();
  
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
