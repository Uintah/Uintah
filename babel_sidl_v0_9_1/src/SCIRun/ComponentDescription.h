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
 *  ComponentDescription.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_ComponentDescription_h
#define SCIRun_ComponentDescription_h

#include <string>

namespace SCIRun
{

class ComponentModel;
class SCIRunFramework;

/** \class ComponentDescription
 *
 * A container for information necessary to locate and instantiate a specific
 * component type in the SCIRun framework.  This class holds the type name of
 * the component and the component model to which it belongs.  Subclasses of
 * ComponentDescription may contain additional information specific to
 * a particular component model.  The SCIRun framework maintains a list of
 * ComponentDescriptions that are used during component instantiation.
 * ComponentDescriptions are usually created from information contained in XML
 * files.
 *
 * \sa BabelComponentDescription
 * \sa CCAComponentDescription
 * \sa VtkComponentDescription
 * \sa ComponentModel
 * \sa SCIRunFramework
 */
class ComponentDescription
{
public:
  ComponentDescription();
  virtual ~ComponentDescription();

  /** Returns the type name (a string) the component described by this class. */
  virtual std::string getType() const = 0;

  /** Returns a pointer to the component model under which the component type
      is defined. */
  virtual const ComponentModel* getModel() const = 0;

  /** ?  */
  virtual std::string getLoaderName() const;
private:
  ComponentDescription(const ComponentDescription&);
  ComponentDescription& operator=(const ComponentDescription&);
};

} // end namespace SCIRun

#endif
