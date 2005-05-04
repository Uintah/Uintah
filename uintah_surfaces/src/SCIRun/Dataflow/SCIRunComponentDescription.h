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
 *  SCIRunComponentDescription.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_SCIRunComponentDescription_h
#define SCIRun_Framework_SCIRunComponentDescription_h

#include <SCIRun/ComponentDescription.h>
#include <string>

namespace SCIRun {

class ComponentModel;
class SCIRunComponentModel;

/**
 * \class SCIRunCompontDescription
 * A refinement of ComponentDescription for the SCIRun dataflow component model.  See
 * ComponentDescription for more information.
 *
 * \sa BabelComponentDescription ComponentDescription VtkComponentDescription
 * \sa InternalComponentDescription CCAComponentDescription
 *
 */
class SCIRunComponentDescription : public ComponentDescription
{
public:
  SCIRunComponentDescription(SCIRunComponentModel* model,
                             const std::string& package,
                             const std::string& category,
                             const std::string& module);
  virtual ~SCIRunComponentDescription();
  
  /** Returns the component type name (a string). */
  virtual std::string getType() const;
  /** Returns a pointer to the component model type. */
  virtual const ComponentModel* getModel() const;
private:
  SCIRunComponentModel* model;
  std::string package;
  std::string category;
  std::string module;
  
  SCIRunComponentDescription(const SCIRunComponentDescription&);
  SCIRunComponentDescription& operator=(const SCIRunComponentDescription&);
};

} // end namespace SCIRun

#endif
