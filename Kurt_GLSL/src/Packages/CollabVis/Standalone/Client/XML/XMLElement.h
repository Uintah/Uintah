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

/**
 * XMLElement provides an abstract interface to an XML element.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class XMLElement {
public:

  /**
   * Constructor. Sets the element tag name, attributes, and text.
   *
   * @param elementName   Element name
   * @param attributes    Any element attributes
   * @param text          Any element text
   */
  XMLElement(String elementName, 
	     Attributes attributes, 
	     String text) { 
    
    this->elementName = elementName; 
    this->attributes  = attributes; 
    this->text        = text; 
  }

  /**
   * Destructor
   *
   */
  ~XMLElement() {}

  /**
   *  Returns the tag name of this element
   *
   * @return Element tag
   */
  inline String getName() const {
    return elementName;
  }

  /**
   * Returns the attributes of this element
   *
   * @return Map of attributes, if any.
   */
  inline Attributes getAttributes() const {
    return attributes;
  }

  /**
   * Returns the text associated with this element.
   *
   * @return Element text, if any
   */
  inline String getText() const {
    return text;
  }

protected:
  /// Element tag
  String      elementName;

  /// Element attributes
  Attributes  attributes;


  /// Element text
  String      text;
  
};

}

#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:59  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:05:32  simpson
// Adding CollabVis files/dirs
//
