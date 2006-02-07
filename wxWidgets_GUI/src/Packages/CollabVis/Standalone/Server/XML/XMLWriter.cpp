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
namespace XML {

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
	
      String key = (e++)->first;
      String val = attributes.getAttribute( key.transcode() );
      
      if (DEBUG) {
	std::cerr << "Adding Attribute -" << key.transcode() << "-, value " <<
	val.transcode() << endl;
      }

      // Add the attribute to the element
      elem.setAttribute( key, val );
    } 
  }    
  
  // If we have any text, add that to the element
  if ( text != 0 ) {
    if (DEBUG) {
      std::cerr << "Adding text node: " << text.transcode() << endl;
    }
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

char *
XMLWriter::writeOutputData() {
  strstream s;
  s << theDoc << ends;

  /* This makes up for a weird bug - sometimes the string output will have
     extra junk data at the end. I don't know if it's in my code or the
     underlying DOM stuff. Either way, this looks for the closing brace
     of the XML and ensures that the next character is \000 - a string
     terminator. FIXED = added ends to prev string.*/
#if 0
  char * foo = s.str();
  int index = 0;
  int i;
  
  for ( i = 0; i < strlen( foo ); i++ )
    if ( foo[ i ] == '>' )
      index = i;
  
  if ( index != i-2 ) { // Error here!
    //std::cerr << "Last bit of XML is not a closing brace!" << endl;
    //std::cerr << "Index = " << index << ":" << foo[ index ] <<
    // "I = " << i-2 << ":" << foo[i-2] << endl;
      
    foo[index+2] = 0;
  }
  return foo;
#else
  return s.str();
#endif
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
  char * p = XMLI::getChar( toWrite );//toWrite.transcode();
  //target << p;
  WRITE( target, p );
  delete[] p;
  //target << toWrite << endl;
  return target;
}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:56  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:26:55  simpson
// Adding CollabVis files/dirs
//
// Revision 1.10  2001/07/31 22:48:33  luke
// Pre-SGI port
//
// Revision 1.9  2001/07/17 23:22:37  luke
// Sample server more stable. Now we can send geom or image data to multiple clients.
//
// Revision 1.8  2001/05/31 21:37:43  luke
// Fixed problem with XML spewing random extra stuff at the end of output. Can now connect & run with linux client
//
// Revision 1.7  2001/05/29 03:43:13  luke
// Merged in changed to allow code to compile/run on IRIX. Note that we have a problem with network byte order in the networking code....
//
// Revision 1.6  2001/05/12 03:32:49  luke
// Moved driver to new location
//
// Revision 1.5  2001/04/04 21:35:33  luke
// Added XML initialization to reader and writer constructors
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
