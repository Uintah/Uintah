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

static const int DEBUG = 0;

#define WRITE( target, text ) (target).write( (text), strlen( (text) ) ) 

namespace SemotusVisum {
namespace XML {

/**************************************
 
CLASS
   XMLWriter
   
KEYWORDS
   XML
   
DESCRIPTION

   XMLWriter provides a convenient and implementation-independent
   way to create and populate an XML document.
   
****************************************/

class XMLWriter {
  
  //////////
  // Overloaded output operator that takes in a node and writes it
  // to the stream.
  friend strstream& operator<<(strstream& target, const DOM_Node& toWrite);

  //////////
  // Overloaded output operator that takes in a DOMString and writes it
  // to the stream.
  friend strstream& operator<<(strstream& target, const DOMString& toWrite);
  
public:
  //////////
  // Default constructor.
  XMLWriter();

  //////////
  // Destructor
  ~XMLWriter();

  //////////
  // Adds a full XML Element to the current document.
  void         addElement(XMLElement e);

  //////////
  // Adds an element to the current document. 
  void         addElement(String  elementName,
			  Attributes attributes,
			  String text);

  //////////
  // Creates a new document for this XML Writer. Must be called before
  // creating a new document.
  void         newDocument();

  //////////
  // Goes up a level in the XML hierarchy
  void         pop();

  //////////
  // Pushes down a level in the XML hierarchy, so that all nodes added
  // after will be children of the last node added.
  void         push();

  //////////
  // Serializes the XML document and returns it in text format.
  char *       writeOutputData();
  
protected:
  stack<DOM_Node>         treeStack;
  DOM_Node                parentNode;
  DOM_Node                lastNode;
  DOM_Document            theDoc;
  DOM_DOMImplementation   impl;
  static void         outputContent(strstream& target,
				    const DOMString &toWrite);
};

}
}
#endif

//
// $Log$
// Revision 1.1  2003/07/22 15:46:57  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:26:55  simpson
// Adding CollabVis files/dirs
//
// Revision 1.5  2001/05/29 03:43:13  luke
// Merged in changed to allow code to compile/run on IRIX. Note that we have a problem with network byte order in the networking code....
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
