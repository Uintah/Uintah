
#include "ProblemSpecReader.h"
#include <Uintah/Exceptions/ProblemSetupException.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <sax/SAXException.hpp>
#include <sax/SAXParseException.hpp>
#include <sax/ErrorHandler.hpp>
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace Uintah;

void outputContent(ostream& target, const DOMString &s);
ostream& operator<<(ostream& target, const DOMString& toWrite);
ostream& operator<<(ostream& target, DOM_Node& toWrite);
static bool     doEscapes       = true;

static string to_string(int i)
{
    char buf[20];
    sprintf(buf, "%d", i);
    return string(buf);
}

static string xmlto_string(const DOMString& str)
{
    char* s = str.transcode();
    string ret = string(s);
    delete[] s;
    return ret;
}

static string xmlto_string(const XMLCh* const str)
{
    char* s = XMLString::transcode(str);
    string ret = string(s);
    delete[] s;
    return ret;
}

class MyErrorHandler : public ErrorHandler {
public:
    bool foundError;

    MyErrorHandler();
    ~MyErrorHandler();

    void warning(const SAXParseException& e);
    void error(const SAXParseException& e);
    void fatalError(const SAXParseException& e);
    void resetErrors();

private :
    MyErrorHandler(const MyErrorHandler&);
    void operator=(const MyErrorHandler&);
};

MyErrorHandler::MyErrorHandler()
{
    foundError=false;
}

MyErrorHandler::~MyErrorHandler()
{
}

static void postMessage(const string& errmsg, bool err=true)
{
    cerr << errmsg << '\n';
}

void MyErrorHandler::error(const SAXParseException& e)
{
    foundError=true;
    postMessage(string("Error at (file ")+xmlto_string(e.getSystemId())
		+", line "+to_string((int)e.getLineNumber())
		+", char "+to_string((int)e.getColumnNumber())
		+"): "+xmlto_string(e.getMessage()));
}

void MyErrorHandler::fatalError(const SAXParseException& e)
{
    foundError=true;
    postMessage(string("Fatal Error at (file ")+xmlto_string(e.getSystemId())
		+", line "+to_string((int)e.getLineNumber())
		+", char "+to_string((int)e.getColumnNumber())
		+"): "+xmlto_string(e.getMessage()));
}

void MyErrorHandler::warning(const SAXParseException& e)
{
    postMessage(string("Warning at (file ")+xmlto_string(e.getSystemId())
		+", line "+to_string((int)e.getLineNumber())
		+", char "+to_string((int)e.getColumnNumber())
		+"): "+xmlto_string(e.getMessage()));
}

void MyErrorHandler::resetErrors()
{
}

static string toString(const DOMString& s)
{
    char *p = s.transcode();
    string r (p);
    delete [] p;
    return r;
}

ProblemSpecReader::ProblemSpecReader(const std::string& filename)
    : filename(filename)
{

}

ProblemSpecReader::~ProblemSpecReader()
{
}

ProblemSpecP ProblemSpecReader::readInputFile()
{
  
  try {
    XMLPlatformUtils::Initialize();
  }
  catch(const XMLException& toCatch) {
      throw ProblemSetupException("XML Exception: "+toString(toCatch.getMessage()));
  }
  

  ProblemSpecP prob_spec;
  try {
      // Instantiate the DOM parser.
      DOMParser parser;
      parser.setDoValidation(false);

      MyErrorHandler handler;
      parser.setErrorHandler(&handler);

      // Parse the input file
      // No exceptions just yet, need to add

      cout << "Parsing " << filename << endl;
      parser.parse(filename.c_str());

      if(handler.foundError)
	  throw ProblemSetupException("Error reading file: "+filename);

      // Add the parser contents to the ProblemSpecP d_doc

      DOM_Document doc = parser.getDocument();

      prob_spec = new ProblemSpec;
      DOM_Node null_node;
 
      prob_spec->setDoc(doc);
      prob_spec->setNode(null_node);
  } catch(const XMLException& ex) {
      throw ProblemSetupException("XML Exception: "+toString(ex.getMessage()));
  }

  return prob_spec;
}

// ---------------------------------------------------------------------------
//
//  ostream << DOM_Node   
//
//                Stream out a DOM node, and, recursively, all of its children.
//                This function is the heart of writing a DOM tree out as
//                XML source.  Give it a document node and it will do the whole thing.
//
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

            //
            //  Test for the presence of children, which includes both
            //  text content and nested elements.
            //
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
                //
                //  There were no children.  Output the short form close of the
                //  element start tag, making it an empty-element tag.
                //
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
//
//  outputContent  - Write document content from a DOMString to a C++ ostream.
//                   Escape the XML special characters (<, &, etc.) unless this
//                   is suppressed by the command line option.
//
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
//
//  ostream << DOMString    Stream out a DOM string.
//                          Doing this requires that we first transcode
//                          to char * form in the default code page
//                          for the system
//
// ---------------------------------------------------------------------------
ostream& operator<<(ostream& target, const DOMString& s)
{
    char *p = s.transcode();
    target << p;
    delete [] p;
    return target;
}
