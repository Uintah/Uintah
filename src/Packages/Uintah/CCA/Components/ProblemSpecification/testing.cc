#include <iostream>

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include "DOMTreeErrorReporter.hpp"
#include <string>
#include <stdlib.h>

void outputContent(ostream& target, const DOMString &s);
ostream& operator<<(ostream& target, const DOMString& toWrite);
ostream& operator<<(ostream& target, DOMNode& toWrite);
void processNode(DOMNode &node,DOMString name);

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

  XercesDOMParser* parser = new XercesDOMParser;

  parser->parse(xmlFile.c_str());

  DOMDocument* doc = parser->getDocument();
 
  DOMNodeList* node_list = doc->getElementsByTagName("Uintah_specification");

  int nlist = node_list.getLength();
  cout << "Number of items in list " << nlist << endl;
  cout << "Name = " << node_list->item(0)->getNodeName() << endl;
  
  // Process all of the nodes in the document

  for (int i = 0; i < nlist; i++) {
    DOMNode* n_child = node_list->item(i);
    cout << "First node in tree: " << n_child->getNodeName() << endl;
    cout << "Type " << n_child->getNodeType() << endl;
    cout << "Contents " << n_child->getNodeValue() << endl;
   
    for (DOMNode* sibling = n_child->getFirstChild(); sibling != 0;
	 sibling = sibling->getNextSibling()) {
      cout << "Sibling node in tree: " << sibling->getNodeName() << endl;
      cout << "Type " << sibling->getNodeType() << endl;
      cout << "Contents " << sibling->getNodeValue() << endl;
    }
   
  }
 

  //  Input a tag name, i.e. such as Meta.  Return the node for this tag.
  //  Then input a child tag that has a value associated with it, i.e.
  //  title and return its contents.
  
 
  DOMNodeList* top_node_list = doc->getElementsByTagName("Time");
  cout << "Processing doc nodes" << endl;
  DOMNode* tmp = doc->cloneNode(true);
  string search("initTime");
  DOMString dsearch(search.c_str());
  processNode(tmp,dsearch);

  // Check if tmp is Null

  if (tmp != 0) {
    cout << "tmp node name is " << tmp->getNodeName() << endl;
  }

  
  delete parser;
  exit(1);
}


void processNode(DOMNode &node,DOMString name)
{
  //  Process the node

  cout << "Name =  " << node.getNodeName() << endl;
  cout << "Type = " <<  node.getNodeType() << endl;
  cout << "Value = " << node.getNodeValue() << endl;
      
  DOMNode child = node.getFirstChild();
    
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
//  ostream << DOMNode   
//                Stream out a DOM node, and, recursively, all of its children.
//                This function is the heart of writing a DOM tree out as
//                XML source.  Give it a document node and it will do the whole thing.
// ---------------------------------------------------------------------------
ostream& operator<<(ostream& target, DOMNode& toWrite)
{
    // Get the name and value out for convenience
    DOMString   nodeName = toWrite.getNodeName();
    DOMString   nodeValue = toWrite.getNodeValue();

	switch (toWrite.getNodeType())
    {
		case DOMNode::TEXT_NODE:
        {
            outputContent(target, nodeValue);
            break;
        }

        case DOMNode::PROCESSING_INSTRUCTION_NODE :
        {
            target  << "<?"
                    << nodeName
                    << ' '
                    << nodeValue
                    << "?>";
            break;
        }

        case DOMNode::DOCUMENT_NODE :
        {
            // Bug here:  we need to find a way to get the encoding name
            //   for the default code page on the system where the
            //   program is running, and plug that in for the encoding
            //   name.  
            target << "<?xml version='1.0' encoding='ISO-8859-1' ?>\n";
            DOMNode child = toWrite.getFirstChild();
            while( child != 0)
            {
                target << child << endl;
                child = child.getNextSibling();
            }

            break;
        }

        case DOMNode::ELEMENT_NODE :
        {
            // Output the element start tag.
            target << '<' << nodeName;

            // Output any attributes on this element
            DOMNamedNodeMap attributes = toWrite.getAttributes();
            int attrCount = attributes.getLength();
            for (int i = 0; i < attrCount; i++)
            {
                DOMNode  attribute = attributes.item(i);

                target  << ' ' << attribute.getNodeName()
                        << " = \"";
                        //  Note that "<" must be escaped in attribute values.
                        outputContent(target, attribute.getNodeValue());
                        target << '"';
            }

            //  Test for the presence of children, which includes both
            //  text content and nested elements.
            DOMNode child = toWrite.getFirstChild();
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

        case DOMNode::ENTITY_REFERENCE_NODE:
        {
            DOMNode child;
            for (child = toWrite.getFirstChild(); child != 0; child = child.getNextSibling())
                target << child;
            break;
        }

        case DOMNode::CDATA_SECTION_NODE:
        {
            target << "<![CDATA[" << nodeValue << "]]>";
            break;
        }

        case DOMNode::COMMENT_NODE:
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
