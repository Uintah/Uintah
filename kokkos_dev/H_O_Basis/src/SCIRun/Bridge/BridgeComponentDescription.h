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
 *  BridgeComponentDescription.h: 
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   September 2003
 *
 */

#ifndef SCIRun_Framework_BridgeComponentDescription_h
#define SCIRun_Framework_BridgeComponentDescription_h

#include <SCIRun/ComponentDescription.h>
#include <Core/CCA/spec/cca_sidl.h>

namespace SCIRun {
  class BridgeComponentModel;
  class BridgeComponentDescription : public ComponentDescription {
  public:
    BridgeComponentDescription(BridgeComponentModel* model);
    virtual ~BridgeComponentDescription();

    virtual std::string getType() const;
    virtual const ComponentModel* getModel() const;
    virtual std::string getLoaderName() const;
    void setLoaderName(const std::string& loaderName);
  protected:
    friend class BridgeComponentModel;
    friend class SCIRunLoader;
    BridgeComponentModel* model;
    std::string type;
    std::string loaderName;
  private:
    BridgeComponentDescription(const BridgeComponentDescription&);
    BridgeComponentDescription& operator=(const BridgeComponentDescription&);
  };
}

#endif
