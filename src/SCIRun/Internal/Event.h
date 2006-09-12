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

#ifndef SCIRun_Internal_Event_h
#define SCIRun_Internal_Event_h

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Internal/EventService.h>
#include <SCIRun/SCIRunFramework.h>

namespace SCIRun {
/**
 * \class Event
 *
 * An Event contains a header (cca.TypeMap) and a body (cca.TypeMap).
 * (more class desc...)
 *
 */
class Event : public sci::cca::Event {
public:
  Event() {}
  Event(const sci::cca::TypeMap::pointer& theHeader, const sci::cca::TypeMap::pointer& theBody);
  virtual void setHeader(const sci::cca::TypeMap::pointer& h);
  virtual void setBody(const sci::cca::TypeMap::pointer& b);

  virtual sci::cca::TypeMap::pointer getHeader();
  virtual sci::cca::TypeMap::pointer getBody();

private:
  sci::cca::TypeMap::pointer header;
  sci::cca::TypeMap::pointer body;
};

}

#endif
