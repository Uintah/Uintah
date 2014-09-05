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
#include <string>

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

#ifndef DEBUG
#define  DEBUG  0
#endif

namespace SemotusVisum {
namespace XML {

/**************************************
 
CLASS
   XMLReader
   
KEYWORDS
   XML
   
DESCRIPTION

   XMLReader provides a convenient and implementation-independent
   way to parse and navigate an XML document.
   
****************************************/

class XMLReader {
  
public:

  //////////
  // Default constructor.
  XMLReader();

  //////////
  // Constructor that sets the input source for the XML document.
  XMLReader( MemBufInputSource *input );

  //////////
  // Constructor that sets the input data for the XML document.
  XMLReader( char * input );
  
  //////////
  // Destructor
  ~XMLReader();

  //////////
  // Returns the tag of the current element
  String        currentElement();

  //////////
  // Returns the list of attributes of this element
  Attributes    getAttributes();

  //////////
  // Returns any text associated with this element
  String        getText();

  //////////
  // Returns true if the current node has any child elements
  bool          hasChildren();

  //////////
  // Returns the tag of the next element in the document, or null if
  // there are no more elements.
  String        nextElement();

  //////////
  // Parses the textual XML input. Must be called before retrieving
  // elements.
  void          parseInputData();

  //////////
  // Pops up to the next level in the tree. Used to ignore children of
  // the current node.
  void          pop();

  //////////
  // Sets the input XML source. Must be called (or used with the
  // constructor) before parsing the document.
  void          setInputData( MemBufInputSource *input);
  
protected:
  MemBufInputSource *input;
  DOM_Document       doc;
  DOMParser          parser;
  DOM_Node           currentNode;
  int                currentIndex;
  stack<DOM_Node>    treeStack;
  stack<int>         indexStack;
  Attributes         attributes;
  String             text;
  bool               isInputOurs;
  
  //////////
  // Returns a count of the nodes in the node list.
  int  countElementNodes( DOM_NodeList nodes);

  //////////
  // Selects the next non-text element in the document. Returns true if
  // there are more elements; else returns false.
  bool getNextAvailableElement( bool recursed );

  //////////
  // Builds an internal list of the current element's attributes and
  // associated text.
  void makeTextAndAttributes();
  
};

} // namespace XML
}
#endif 
//
// $Log$
// Revision 1.1  2003/07/22 15:46:56  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:26:55  simpson
// Adding CollabVis files/dirs
//
// Revision 1.5  2001/08/01 21:40:50  luke
// Fixed a number of memory leaks
//
// Revision 1.4  2001/02/08 23:53:34  luke
// Added network stuff, incorporated SemotusVisum namespace
//
// Revision 1.3  2001/01/31 20:45:34  luke
// Changed Properties to Attributes to avoid name conflicts with client and server properties
//
// Revision 1.2  2001/01/29 18:48:47  luke
// Commented XML
//
