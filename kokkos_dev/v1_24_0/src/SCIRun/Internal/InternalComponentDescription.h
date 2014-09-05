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
 *  InternalComponentDescription.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Internal_InternalComponentDescription_h
#define SCIRun_Internal_InternalComponentDescription_h

#include <SCIRun/ComponentDescription.h>
#include <string>

namespace SCIRun {

class InternalComponentInstance;
class InternalComponentModel;

/** \class InternalComponentDescription
 *
 *
 */
class InternalComponentDescription : public ComponentDescription
{
public:
  InternalComponentDescription(InternalComponentModel* model,
                               const std::string& serviceType,
                               InternalComponentInstance* (*create)(SCIRunFramework*,
                                                                    const std::string&),
                               bool isSingleton);
  virtual ~InternalComponentDescription();

  /** */
  virtual std::string getType() const;
  /** */
  virtual const ComponentModel* getModel() const;
  
private:
  friend class InternalComponentModel;
  InternalComponentModel* model;
  std::string serviceType;
  InternalComponentInstance* (*create)(SCIRunFramework*, const std::string&);
  InternalComponentInstance* singleton_instance;
  bool isSingleton;
  InternalComponentDescription(const InternalComponentDescription&);
  InternalComponentDescription& operator=(const InternalComponentDescription&);
};

} // end namespace SCIRun

#endif
