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


#include <XML/XMLWriter.h>

namespace SemotusVisum {

XMLWriter::XMLWriter() {
  XMLI::initialize();
}

XMLWriter::~XMLWriter() {
}

void
XMLWriter::addElement(XMLElement e) {
  addElement( e.getName(),
	      e.getAttributes(),
	      e.getText()
	      );
}

void
XMLWriter::addElement(String elementName,
		      Attributes attributes,
		      String text) {
  
  // Create an element
  DOM_Element elem = theDoc.createElement(elementName);

  // If we have any attributes
  if (attributes.empty() == false) {
	  
    // For each set of properties specified in Attributes
    for (cmap::const_iterator e = attributes.getIterator();
	 !attributes.mapEnd(e); ) {
      
      char * buffer = toChar( e->first );
      e++;
      String key = buffer;
      delete buffer;
      
      buffer = toChar( attributes.getAttribute( key.transcode() ) );
      String val = buffer;
      delete buffer;
      
      // Add the attribute to the element
      elem.setAttribute( key, val );
    } 
  }    
  
  // If we have any text, add that to the element
  if ( text != 0 ) {
    elem.appendChild( theDoc.createTextNode(text) );
  }
  
  // Add the element to the current parent node
  if (parentNode == NULL) {
    theDoc.appendChild(elem);
    parentNode = elem;
  }
  else { 
    parentNode.appendChild(elem);
    lastNode = elem;
  }
}

void
XMLWriter::newDocument() {
  theDoc = DOM_Document::createDocument();
}

void
XMLWriter::pop() {
  lastNode = parentNode;
  parentNode = treeStack.top();
  treeStack.pop();
}

void
XMLWriter::push() {
  treeStack.push(parentNode);
  parentNode = lastNode;
  lastNode = NULL;
}

string
XMLWriter::writeOutputData() {
  //strstream s;
  //s << theDoc << ends;

  //  return s.str();
  string s;
  return dumpNode( s, theDoc );
}

string
XMLWriter::dumpNode( string start, const DOM_Node& toWrite ) {
  /* Shamelessly (mostly) copied from DOMPrint (from Xerces samples) */

  // Get the name and value out for convenience
  DOMString   nodeName = toWrite.getNodeName();
  DOMString   nodeValue = toWrite.getNodeValue();

  switch (toWrite.getNodeType())
  {
  case DOM_Node::TEXT_NODE:
    {
      start += XMLI::getChar(nodeValue);
      break;
    }
  case DOM_Node::PROCESSING_INSTRUCTION_NODE :
    {
      break;
    }

  case DOM_Node::DOCUMENT_NODE :
    {
      // Bug here:  we need to find a way to get the encoding name
      //   for the default code page on the system where the
      //   program is running, and plug that in for the encoding
      //   name.  
      start += "<?xml version='1.0' encoding='ISO-8859-1' ?>\n";
      DOM_Node child = toWrite.getFirstChild();
      while( child != 0)
      {
	start = dumpNode(start, child);
	child = child.getNextSibling();
      }
      
      break;
    }

  case DOM_Node::ELEMENT_NODE :
    {
      // Output the element start tag.
      start += "<" + XMLI::getChar(nodeName);
      
      // Output any attributes on this element
      DOM_NamedNodeMap attributes = toWrite.getAttributes();
      int attrCount = attributes.getLength();
      for (int i = 0; i < attrCount; i++)
      {
	DOM_Node  attribute = attributes.item(i);

	start += " ";
	start += XMLI::getChar(attribute.getNodeName());
	start += " = \"";
	start += XMLI::getChar(attribute.getNodeValue());
	start += "\"";
      }
      
      //
      //  Test for the presence of children, which includes both
      //  text content and nested elements.
      //
      DOM_Node child = toWrite.getFirstChild();
      if (child != 0)
      {
	// There are children. Close start-tag, and output children.
	start += ">";
	while( child != 0)
	{
	  start = dumpNode(start,child);
	  child = child.getNextSibling();
	}
	
	// Done with children.  Output the end tag.
	start += "</";
	start += XMLI::getChar(nodeName);
	start += ">";
      }
      else
      {
	//
	//  There were no children.  Output the short form close of the
	//  element start tag, making it an empty-element tag.
	start += "/>";
      }
      break;
    }
  
  case DOM_Node::ENTITY_REFERENCE_NODE:
    {
      DOM_Node child;
      for (child = toWrite.getFirstChild();
	   child != 0;
	   child = child.getNextSibling())
	start = dumpNode(start, child);
      break;
    }
    
  case DOM_Node::CDATA_SECTION_NODE:
    {
      start += "<![CDATA[";
      start += XMLI::getChar(nodeValue);
      start += "]]>";
      break;
    }
    
  case DOM_Node::COMMENT_NODE:
    {
      start += "<!--";
      start += XMLI::getChar(nodeValue);
      start += "-->";
      break;
    }
    
  default:
    std::cerr << "Unrecognized node type = "
	      << (long)toWrite.getNodeType() << endl;
  }
  return start;
}

strstream&
operator<<(strstream& target, const DOM_Node& toWrite) {
  /* Shamelessly (mostly) copied from DOMPrint (from Xerces samples) */

  // Get the name and value out for convenience
  DOMString   nodeName = toWrite.getNodeName();
  DOMString   nodeValue = toWrite.getNodeValue();

  switch (toWrite.getNodeType())
  {
  case DOM_Node::TEXT_NODE:
    {
      target << nodeValue;
      break;
    }

  case DOM_Node::PROCESSING_INSTRUCTION_NODE :
    {
      /*target  << "<?"
	      << nodeName << ' '
	      << nodeValue
	      << "?>";*/
      break;
    }

  case DOM_Node::DOCUMENT_NODE :
    {
      // Bug here:  we need to find a way to get the encoding name
      //   for the default code page on the system where the
      //   program is running, and plug that in for the encoding
      //   name.  
      //target << (char *)"<?xml version='1.0' encoding='ISO-8859-1' ?>\n";
      //  target.write("<?xml version='1.0' encoding='ISO-8859-1' ?>\n",
      //	   strlen( "<?xml version='1.0' encoding='ISO-8859-1' ?>\n" ));
      WRITE( target, "<?xml version='1.0' encoding='ISO-8859-1' ?>\n" );
      DOM_Node child = toWrite.getFirstChild();
      while( child != 0)
      {
	target << child << endl;
	child = child.getNextSibling();
      }
      
      break;
    }

  case DOM_Node::ELEMENT_NODE :
    {
      // Output the element start tag.
      //target << '<';
      WRITE( target, "<" );
      target << nodeName;
      
      // Output any attributes on this element
      DOM_NamedNodeMap attributes = toWrite.getAttributes();
      int attrCount = attributes.getLength();
      for (int i = 0; i < attrCount; i++)
      {
	DOM_Node  attribute = attributes.item(i);
	
	//target  << ' ';
	WRITE( target, " " );
	target << attribute.getNodeName();
	//target  << " = \"";
	WRITE( target, " = \"" );
	//  Note that "<" must be escaped in attribute values.
	target << attribute.getNodeValue();
	
	//target << '"';
	WRITE( target, "\"" );
      }
      
      //
      //  Test for the presence of children, which includes both
      //  text content and nested elements.
      //
      DOM_Node child = toWrite.getFirstChild();
      if (child != 0)
      {
	// There are children. Close start-tag, and output children.
	//target << ">";
	WRITE( target, ">" );
	while( child != 0)
	{
	  target << child;
	  child = child.getNextSibling();
	}
	
	// Done with children.  Output the end tag.
	//target << "</";
	WRITE( target, "</" );
	target << nodeName;
	//target << ">";
	WRITE( target, ">" );
      }
      else
      {
	//
	//  There were no children.  Output the short form close of the
	//  element start tag, making it an empty-element tag.
	//
	//target << "/>";
	WRITE( target, "/>" );
      }
      break;
    }
  
  case DOM_Node::ENTITY_REFERENCE_NODE:
    {
      DOM_Node child;
      for (child = toWrite.getFirstChild(); child != 0; child = child.getNextSibling())
	target << child;
      break;
    }
    
  case DOM_Node::CDATA_SECTION_NODE:
    {
      WRITE( target, "<![CDATA[" );
      //target << "<![CDATA[";
      target << nodeValue;
      //target << "]]>";
      WRITE( target, "]]>" );
      break;
    }
    
  case DOM_Node::COMMENT_NODE:
    {
      //target << "<!--";
      WRITE( target, "<!--" );
      target << nodeValue;
      //target << "-->";
      WRITE( target, "-->" );
      break;
    }
    
  default:
    std::cerr << "Unrecognized node type = "
	      << (long)toWrite.getNodeType() << endl;
  }
  return target;
}

strstream&
operator<<(strstream& target, const DOMString &toWrite) {
  /* Copied from Xerces samples (DOMPrint) */
  //char * p = XMLI::getChar( toWrite ).pointer;//toWrite.transcode();
  //target << p;
  //WRITE( target, p );
  //delete[] p;
  //target << toWrite << endl;
  return target;
}

}
//
// $Log$
// Revision 1.1  2003/07/22 21:00:00  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:05:33  simpson
// Adding CollabVis files/dirs
//
