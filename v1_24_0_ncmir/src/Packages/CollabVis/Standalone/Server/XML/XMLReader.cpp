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

#include <XML/XMLReader.h>
#include <Malloc/Allocator.h>
namespace SemotusVisum {
namespace XML {
using namespace SCIRun;

// Instantiate XMLI mutex
Mutex
XMLI::mutex( "XML Mutex" );

XMLReader::XMLReader() 
{
  input = NULL;
  isInputOurs = false;
  XMLI::initialize();
  
}

XMLReader::XMLReader( MemBufInputSource *input ) {
  this->input = input;
  isInputOurs = false;
  XMLI::initialize();
}

XMLReader::XMLReader( char * input ) {

  XMLByte * in = (XMLByte *) input;
  MemBufInputSource *inputSource
    = scinew MemBufInputSource( in, strlen( input ), "foobar" );
  this->input = inputSource;
  isInputOurs = true;
}


XMLReader::~XMLReader() {
  if ( isInputOurs )
    delete input;
}

String
XMLReader::currentElement() {

  if (currentNode == NULL)
    return String(0); // Not on a current element.
  
  return currentNode.getNodeName();
}

Attributes
XMLReader::getAttributes() {
  return attributes;
}

String 
XMLReader::getText() {
  return text;
}

bool 
XMLReader::hasChildren() {
  return (countElementNodes(currentNode.getChildNodes()) > 0);
}

String
XMLReader::nextElement() {
  // Get the next child element
  
  /* If the node is a document node, push it on the stack and 
     start on its children */
  if (currentNode.getNodeType() == DOM_Node::DOCUMENT_NODE) {
    treeStack.push(currentNode);

    /* Ensure that we set the current node to be an element, and not 
       some other type of node */
    DOM_NodeList children = currentNode.getChildNodes();
    
    for (currentIndex = 0; 
	 (unsigned)currentIndex < children.getLength(); 
	 currentIndex++) {
      currentNode = children.item(currentIndex);
      
      if (currentNode.getNodeType() == DOM_Node::ELEMENT_NODE) {
	
	// Build metainfo
	makeTextAndAttributes();
	
	// Return the current information
	return currentElement();
      }
    }
  
    // If there are no more children, return NULL. 
    if ((unsigned)currentIndex == children.getLength())
      return String(0);
    
  }
  
  /* Get the next available element node */
  if (getNextAvailableElement(false) == false)
    return String(0); // No more elements
  
  /* Build metainfo */
  makeTextAndAttributes();
  
  // Return the current information
  return currentElement();
}

void
XMLReader::parseInputData() {

  parser.parse(*(this->input));
  
  doc = parser.getDocument();
  currentNode = doc;

  /* Clear the stacks */
  while (!treeStack.empty())
    treeStack.pop();

  while (!indexStack.empty())
    indexStack.pop();
}

void
XMLReader::pop() { 
  currentNode  = treeStack.top();
  currentIndex = indexStack.top();
  
  treeStack.pop();
  indexStack.pop();
}

void
XMLReader::setInputData( MemBufInputSource *input) {
  this->input = input;
}

int
XMLReader::countElementNodes( DOM_NodeList nodes) {
  int total = 0;
  
  for (int index = 0; (unsigned)index < nodes.getLength(); index++)
    if (nodes.item(index).getNodeType() == DOM_Node::ELEMENT_NODE)
      total++;
  
  return total;
}

bool
XMLReader::getNextAvailableElement( bool recursed ) {
  DOM_NodeList children = currentNode.getChildNodes();
	

  /* If this element has children, use them */
  if (!recursed && countElementNodes(children) > 0) {
    
    // Push the current element and index onto the stack.
    treeStack.push(currentNode);
    indexStack.push(currentIndex);
	  
    // Iterate through the children. When we get an element, return.
    for (currentIndex = 0; 
	 (unsigned)currentIndex < children.getLength(); 
	 currentIndex++)
      
      if (children.item(currentIndex).getNodeType() ==
	  DOM_Node::ELEMENT_NODE) {
	currentNode = children.item(currentIndex);
	return true;
      }
    
  }
  /* We have no children. Thus, we need to continue with the elements
     at the current level. */
  else {
    currentIndex++;
    currentNode = 
      ((DOM_Node)treeStack.top()).getChildNodes().item(currentIndex);
    
    while ( (currentNode != NULL) &&
	    (currentNode.getNodeType() != DOM_Node::ELEMENT_NODE) ) {
      currentIndex++;
      currentNode = 
	((DOM_Node)treeStack.top()).getChildNodes().item(currentIndex);
    }
    
    /* If currentNode is NULL, then we have no more elements at this level.
       We need to pop the last element off the stack and continue 
       at that level. */
    if (currentNode == NULL) {

      if ( indexStack.empty() )
	return false;
      
      currentIndex = indexStack.top();
      indexStack.pop();
      
    
      // Pop the last element off the stack.
      currentNode = treeStack.top();
      treeStack.pop();
      
      /* We now have restored our previous level in the hierarchy. Recurse
	 through this upper level. */
      return getNextAvailableElement(true);
    }
    else return true; // We have found the next element.
    
  }
  return false;
}

void
XMLReader::makeTextAndAttributes() {
  /* First, build the attribute list. */
  DOM_NamedNodeMap attrs;
  DOM_Node attribute;

  this->attributes.clear();
  attrs = currentNode.getAttributes();
  
  // For each attribute, add it to the properties list.
  for ( int i = 0; (unsigned)i < attrs.getLength(); i++) {
    attribute = attrs.item(i);
    //attributes.setAttribute( attribute.getNodeName().transcode(),
    //			     attribute.getNodeValue().transcode() );
    attributes.setAttribute( XMLI::getChar( attribute.getNodeName() ),
			     XMLI::getChar( attribute.getNodeValue() ) );
  }
  
  
  /* Now, find the text node (if any) */
  DOM_NodeList children = currentNode.getChildNodes();

  this->text = String(0);
  
  // For each node, test to see if it's the text node. 
  // If so, set it to be the text and exit.
  for (int index = 0; (unsigned)index < children.getLength(); index++) {
    if (children.item(index).getNodeType() == DOM_Node::TEXT_NODE) {
      text = children.item(index).getNodeValue();
      break;
    }
  }
}


} // namespace XML
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:56  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:26:54  simpson
// Adding CollabVis files/dirs
//
// Revision 1.11  2001/10/11 16:38:08  luke
// Foo
//
// Revision 1.10  2001/08/01 21:40:50  luke
// Fixed a number of memory leaks
//
// Revision 1.9  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.8  2001/07/17 23:22:37  luke
// Sample server more stable. Now we can send geom or image data to multiple clients.
//
// Revision 1.7  2001/05/14 22:39:02  luke
// Fixed compiler warnings
//
// Revision 1.6  2001/05/12 03:32:49  luke
// Moved driver to new location
//
// Revision 1.5  2001/04/04 21:35:33  luke
// Added XML initialization to reader and writer constructors
//
// Revision 1.4  2001/02/08 23:53:33  luke
// Added network stuff, incorporated SemotusVisum namespace
//
// Revision 1.3  2001/01/31 20:45:34  luke
// Changed Properties to Attributes to avoid name conflicts with client and server properties
//
// Revision 1.2  2001/01/29 18:48:47  luke
// Commented XML
//
