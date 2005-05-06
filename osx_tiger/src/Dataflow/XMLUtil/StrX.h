/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


#ifndef Dataflow_Dataflow_StrX_h
#define PSECORE_Dataflow_StrX_h 1

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#define IRIX
#pragma set woff 1375
#pragma set woff 3303
#endif
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMText.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#pragma reset woff 3303
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
  
//   StrX(const DOMText& str)
//   {
//     // Call the transcoding method
//     fLocalForm = str.transcode();
//   }
  
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
