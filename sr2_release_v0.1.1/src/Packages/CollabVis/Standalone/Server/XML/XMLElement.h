/*
 *
 * XMLElement: Abstraction for XML element.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */

#ifndef __XMLElement_H_
#define __XMLElement_H_

#include <XML/XML.h>
#include <XML/Attributes.h>

namespace SemotusVisum {
namespace XML {

/**************************************
 
CLASS
   XMLElement
   
KEYWORDS
   XML
   
DESCRIPTION

   XMLElement provides an abstract interface to an XML element.
   
****************************************/
class XMLElement {

  
public:

  //////////
  // Constructor. Sets the element tag name, attributes, and text.
  XMLElement(String elementName, 
	     Attributes attributes, 
	     String text) { 
    
    this->elementName = elementName; 
    this->attributes  = attributes; 
    this->text        = text; 
  }

  //////////
  // Destructor
  ~XMLElement() {}

  //////////
  // Returns the tag name of this element
  inline String getName() const {
    return elementName;
  }

  //////////
  // Returns the attributes of this element
  inline Attributes getAttributes() const {
    return attributes;
  }

  //////////
  // Returns the text associated with this element.
  inline String getText() const {
    return text;
  }

protected:
  String      elementName;
  Attributes  attributes;
  String      text;
  
};

} // namespace XML
}

#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:56  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:26:54  simpson
// Adding CollabVis files/dirs
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
