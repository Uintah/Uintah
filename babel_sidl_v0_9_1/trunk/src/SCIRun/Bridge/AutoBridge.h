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


/*
 *  AutoBridge.h: Interface from framework to automatic bridge gen tools 
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   February 2004
 *
 */

#ifndef SCIRun_Bridge_AutoBridge_h
#define SCIRun_Bridge_AutoBridge_h

#include <Core/CCA/tools/strauss/strauss.h>
#include <SCIRun/PortInstance.h>
#include <set>

namespace SCIRun {
  class AutoBridge {
  public:
    AutoBridge(); 
    virtual ~AutoBridge();
    std::string genBridge(std::string modelFrom, std::string cFrom, std::string modelTo, std::string cTo);
    bool canBridge(PortInstance* pr1, PortInstance* pr2);
  private:
    ///////
    //list of bridges that just happened to exist in directory
    std::set<std::string > oldB;

    //////
    //runtime cache used to maintain a list of generated bridges 
    std::set<std::string > runC;

    /////
    //Compare CRC of existing files found in oldB to the strauss emitted ones
    //Return true if they match. (Used for caching between different runs)
    bool isSameFile(std::string name, Strauss* strauss);
  };
}

#endif
