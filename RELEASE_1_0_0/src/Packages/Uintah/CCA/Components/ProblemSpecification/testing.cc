#include <iostream>

#include <util/PlatformUtils.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_Node.hpp>
#include <dom/DOM_NamedNodeMap.hpp>
#include "DOMTreeErrorReporter.hpp"
#include <string>
#include <stdlib.h>

void outputContent(ostream& target, const DOMString &s);
ostream& operator<<(ostream& target, const DOMString& toWrite);
ostream& operator<<(ostream& target, DOM_Node& toWrite);
void processNode(DOM_Node &node,DOMString name);

static bool     doEscapes       = true;

int main(int argc, char *argv[]){
  
 
  try {
    XMLPlatformUtils::Initialize();
  }
  catch(const XMLException& toCatch) {
    cerr << "Error during Xerces-c Initialization.\n"
	 << "  Exception message:"
	 << DOMString(toCatch.getMessage()) << endl;
    return 1;
  }

  if (argc == 1) {
    cout << argv[0] << endl;
    exit(1);
  }
  else if (argc == 2)
    cout << argv[0] << " " << argv[1] << endl;

  string xmlFile = argv[1];

  DOMParser parser;

  parser.parse(xmlFile.c_str());

  DOM_Document doc = parser.getDocument();
 
  DOM_NodeList node_list = doc.getElementsByTagName("Uintah_specification");

  int nlist = node_list.getLength();
  cout << "Number of items in list " << nlist << endl;
  cout << "Name = " << node_list.item(0).getNodeName() << endl;
  
  // Process all of the nodes in the document

  for (int i = 0; i < nlist; i++) {
    DOM_Node n_child = node_list.item(i);
    cout << "First node in tree: " << n_child.getNodeName() << endl;
    cout << "Type " << n_child.getNodeType() << endl;
    cout << "Contents " << n_child.getNodeValue() << endl;
   
    for (DOM_Node sibling = n_child.getFirstChild(); sibling != 0;
	 sibling = sibling.getNextSibling()) {
      cout << "Sibling node in tree: " << sibling.getNodeName() << endl;
      cout << "Type " << sibling.getNodeType() << endl;
      cout << "Contents " << sibling.getNodeValue() << endl;
    }
   
  }
 

  //  Input a tag name, i.e. such as Meta.  Return the node for this tag.
  //  Then input a child tag that has a value associated with it, i.e.
  //  title and return its contents.
  
 
  DOM_NodeList top_node_list = doc.getElementsByTagName("Time");
  cout << "Processing doc nodes" << endl;
  DOM_Node tmp = doc.cloneNode(true);
  string search("initTime");
  DOMString dsearch(search.c_str());
  processNode(tmp,dsearch);

  // Check if tmp is Null

  if (tmp != 0) {
    cout << "tmp node name is " << tmp.getNodeName() << endl;
  }

  

  exit(1);
}


void processNode(DOM_Node &node,DOMString name)
{
  //  Process the node

  cout << "Name =  " << node.getNodeName() << endl;
  cout << "Type = " <<  node.getNodeType() << endl;
  cout << "Value = " << node.getNodeValue() << endl;
      
  DOM_Node child = node.getFirstChild();
    
  while (child != 0) {
    DOMString child_name = child.getNodeName();
    cout << "child name = " << child_name << endl;
    cout << "name = " << name << endl;
    if (child_name.equals(name) ) {
      node = child.cloneNode(true);
      cout << "node name is now = " << node.getNodeName() << endl;
      return;
    }
    processNode(child,name);
    child = child.getNextSibling();
  }
 
  
 
}
  

// ---------------------------------------------------------------------------
//  ostream << DOM_Node   
//                Stream out a DOM node, and, recursively, all of its children.
//                This function is the heart of writing a DOM tree out as
//                XML source.  Give it a document node and it will do the whole thing.
// ---------------------------------------------------------------------------
ostream& operator<<(ostream& target, DOM_Node& toWrite)
{
    // Get the name and value out for convenience
    DOMString   nodeName = toWrite.getNodeName();
    DOMString   nodeValue = toWrite.getNodeValue();

	switch (toWrite.getNodeType())
    {
		case DOM_Node::TEXT_NODE:
        {
            outputContent(target, nodeValue);
            break;
        }

        case DOM_Node::PROCESSING_INSTRUCTION_NODE :
        {
            target  << "<?"
                    << nodeName
                    << ' '
                    << nodeValue
                    << "?>";
            break;
        }

        case DOM_Node::DOCUMENT_NODE :
        {
            // Bug here:  we need to find a way to get the encoding name
            //   for the default code page on the system where the
            //   program is running, and plug that in for the encoding
            //   name.  
            target << "<?xml version='1.0' encoding='ISO-8859-1' ?>\n";
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
            target << '<' << nodeName;

            // Output any attributes on this element
            DOM_NamedNodeMap attributes = toWrite.getAttributes();
            int attrCount = attributes.getLength();
            for (int i = 0; i < attrCount; i++)
            {
                DOM_Node  attribute = attributes.item(i);

                target  << ' ' << attribute.getNodeName()
                        << " = \"";
                        //  Note that "<" must be escaped in attribute values.
                        outputContent(target, attribute.getNodeValue());
                        target << '"';
            }

            //  Test for the presence of children, which includes both
            //  text content and nested elements.
            DOM_Node child = toWrite.getFirstChild();
            if (child != 0)
            {
                // There are children. Close start-tag, and output children.
                target << ">";
                while( child != 0)
                {
                    target << child;
                    child = child.getNextSibling();
                }

                // Done with children.  Output the end tag.
                target << "</" << nodeName << ">";
            }
            else
            {
                //  There were no children.  Output the short form close of the
                //  element start tag, making it an empty-element tag.
                target << "/>";
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
            target << "<![CDATA[" << nodeValue << "]]>";
            break;
        }

        case DOM_Node::COMMENT_NODE:
        {
            target << "<!--" << nodeValue << "-->";
            break;
        }

        default:
            cerr << "Unrecognized node type = "
                 << (long)toWrite.getNodeType() << endl;
    }
	return target;
}


// ---------------------------------------------------------------------------
//  outputContent  - Write document content from a DOMString to a C++ ostream.
//                   Escape the XML special characters (<, &, etc.) unless this
//                   is suppressed by the command line option.
// ---------------------------------------------------------------------------
void outputContent(ostream& target, const DOMString &toWrite)
{
    
    if (doEscapes == false)
    {
        target << toWrite;
    }
     else
    {
        int            length = toWrite.length();
        const XMLCh*   chars  = toWrite.rawBuffer();
        
        int index;
        for (index = 0; index < length; index++)
        {
            switch (chars[index])
            {
            case chAmpersand :
                target << "&amp;";
                break;
                
            case chOpenAngle :
                target << "&lt;";
                break;
                
            case chCloseAngle:
                target << "&gt;";
                break;
                
            case chDoubleQuote :
                target << "&quot;";
                break;
                
            default:
                // If it is none of the special characters, print it as such
                target << toWrite.substringData(index, 1);
                break;
            }
        }
    }

    return;
}


// ---------------------------------------------------------------------------
//  ostream << DOMString    Stream out a DOM string.
//                          Doing this requires that we first transcode
//                          to char * form in the default code page
//                          for the system
// ---------------------------------------------------------------------------
ostream& operator<<(ostream& target, const DOMString& s)
{
    char *p = s.transcode();
    target << p;
    delete [] p;
    return target;
}
