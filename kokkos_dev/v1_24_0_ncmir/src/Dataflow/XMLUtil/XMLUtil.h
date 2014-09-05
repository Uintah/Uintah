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


#ifndef Dataflow_XMLUtil_XMLUtil_H
#define Dataflow_XMLUtil_XMLUtil_H

namespace SCIRun {
  class Point;
  class Vector;
  class IntVector;
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#define IRIX
#pragma set woff 1375
#pragma set woff 3303
#endif
#include <xercesc/util/TransService.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOMText.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/util/XMLUni.hpp>
#ifdef XERCESDEFS_HPP
#include <xercesc/util/XMLUniDefs.hpp>
#endif
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#pragma reset woff 3303
#endif
#include <string>
#include <iosfwd>

/* NOT_SET is used to indicate active 
   fields inside of data structures that
   represent XML element trees */

#define NOT_SET ((char*)_NOTSET_)

namespace SCIRun {
using std::string;

static XMLLCPTranscoder* lcptrans_ = 0;

template <class To, class From>
bool convert(To*& to_str, From*& from_str) {
  if (!lcptrans_) {
    lcptrans_ = XMLPlatformUtils::fgTransService->makeNewLCPTranscoder();
  }
  static unsigned int buf_sz_ = 255;
  static To* char_buf_ = new To[buf_sz_ + 1];
  unsigned int sz = lcptrans_->calcRequiredSize(from_str);
  if (sz > buf_sz_) {
    buf_sz_ = sz;
    delete[] char_buf_;
    char_buf_ = new To[buf_sz_ + 1];
  }
  
  if (lcptrans_->transcode(from_str, char_buf_, buf_sz_)) {
    // make sure we are null terminated
    char_buf_[sz] = '\0';
    to_str = char_buf_;
    return true;
  }
  return false;
}

// same as above, but utilizes a second buffer (for attribute maps)
template <class To, class From>
bool convert2(To*& to_str, From*& from_str) {
  if (!lcptrans_) {
    lcptrans_ = XMLPlatformUtils::fgTransService->makeNewLCPTranscoder();
  }
  static unsigned int buf_sz_ = 255;
  static To* char_buf_ = new To[buf_sz_ + 1];
  unsigned int sz = lcptrans_->calcRequiredSize(from_str);
  if (sz > buf_sz_) {
    buf_sz_ = sz;
    delete[] char_buf_;
    char_buf_ = new To[buf_sz_ + 1];
  }
  
  if (lcptrans_->transcode(from_str, char_buf_, buf_sz_)) {
    // make sure we are null terminated
    char_buf_[sz] = '\0';
    to_str = char_buf_;
    return true;
  }
  return false;
}


inline
const char* to_char_ptr(const XMLCh  *t) {
  char* rval;
  if (convert(rval, t)) return rval;
  else return 0;
}

inline
const XMLCh* to_xml_ch_ptr(const char *t) {
  XMLCh* rval;
  if (convert(rval, t)) return rval;
  else return 0;
}

// same as above but utilizes a second buffer (for attribute maps)
inline
const XMLCh* to_xml_ch_ptr2(const char *t) {
  XMLCh* rval;
  if (convert2(rval, t)) return rval;
  else return 0;
}

inline
bool string_is(const XMLCh *childname, const char *const name) {
  return (XMLString::compareString(childname, to_xml_ch_ptr(name)) == 0);
}
const DOMNode* findNode(const std::string &name, const DOMNode *node);
const DOMNode* findNextNode(const std::string& name, const DOMNode* node);
const DOMNode* findTextNode(const DOMNode* node);
void outputContent(std::ostream& target, const char *s);
std::ostream& operator<<(std::ostream& target, const DOMText* toWrite);
std::ostream& operator<<(std::ostream& target, const DOMNode* toWrite);
void appendElement(DOMElement* root, const DOMText* name,
		   const std::string& value);
void appendElement(DOMElement* root, const DOMText* name,
		   int value);
void appendElement(DOMElement* root, const DOMText* name,
		   const IntVector& value);
void appendElement(DOMElement* root, const DOMText* name,
		   const Point& value);
void appendElement(DOMElement* root, const DOMText* name,
		   const Vector& value);
void appendElement(DOMElement* root, const DOMText* name,
		   long value);
void appendElement(DOMElement* root, const DOMText* name,
		   double value);
bool get(const DOMNode* node, int &value);
bool get(const DOMNode* node,
	 const std::string& name, int &value);
bool get(const DOMNode* node, long &value);
bool get(const DOMNode* node,
	 const std::string& name, long &value);
bool get(const DOMNode* node,
	 const std::string& name, double &value);
bool get(const DOMNode* node,
	 const std::string& name, std::string &value);
bool get(const DOMNode* node,
	 const std::string& name, Vector& value);
bool get(const DOMNode* node,
	 const std::string& name, Point& value);
bool get(const DOMNode* node,
	 const std::string& name, IntVector &value);
bool get(const DOMNode* node,
	 const std::string& name, bool &value);

extern const char _NOTSET_[];      
      
//////////////////////////////
// getSerializedAttributes()
// returns a string that has an XML format
// which represents the attributes of "node" 
      
char* getSerializedAttributes(DOMNode* node);
      
      
//////////////////////////////
// getSerializedChildren()
// returns a string in XML format that
// represents the children of the node
// named "node".
      
char* getSerializedChildren(DOMNode* node);
      
      
string xmlto_string(const DOMText* str);
string xmlto_string(const XMLCh* const str);
void invalidNode(const DOMNode* n, const string& filename);
const XMLCh* findText(DOMNode* node);
	

} // End namespace SCIRun

#endif
