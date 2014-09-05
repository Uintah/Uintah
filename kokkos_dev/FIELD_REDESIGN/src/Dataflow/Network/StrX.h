#ifndef PSECore_Dataflow_StrX_h
#define PSECORE_Dataflow_StrX_h 1

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#include <sax/SAXException.hpp>
#include <sax/SAXParseException.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_NamedNodeMap.hpp>
#include <sax/ErrorHandler.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif

#include <iostream>

namespace PSECore {
namespace Dataflow {


// ---------------------------------------------------------------------------
//  This is a simple class that lets us do easy (though not terribly efficient)
//  trancoding of XMLCh data to local code page for display.
// ---------------------------------------------------------------------------
class StrX
{
public :
  // -----------------------------------------------------------------------
  //  Constructors and Destructor
  // -----------------------------------------------------------------------
  StrX(const XMLCh* const toTranscode)
  {
    // Call the private transcoding method
    fLocalForm = XMLString::transcode(toTranscode);
  }
  
  StrX(const DOMString& str)
  {
    // Call the transcoding method
    fLocalForm = str.transcode();
  }
  
  ~StrX()
  {
    delete [] fLocalForm;
  }
  
  
  // -----------------------------------------------------------------------
  //  Getter methods
  // -----------------------------------------------------------------------
  const char* localForm() const
  {
    return fLocalForm;
  }
  
private :
  // -----------------------------------------------------------------------
  //  Private data members
  //
  //  fLocalForm
  //      This is the local code page form of the string.
  // -----------------------------------------------------------------------
  char*   fLocalForm;
};

std::ostream& operator<<(std::ostream& target, const StrX& toDump);

} // Dataflow
} // PSECore

#endif
