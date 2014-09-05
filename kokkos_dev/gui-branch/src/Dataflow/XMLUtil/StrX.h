/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#ifndef Dataflow_Dataflow_StrX_h
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

namespace SCIRun {


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
  //  fLocalForm
  //      This is the local code page form of the string.
  // -----------------------------------------------------------------------
  char*   fLocalForm;
};

std::ostream& operator<<(std::ostream& target, const StrX& toDump);

} // End namespace SCIRun

#endif
