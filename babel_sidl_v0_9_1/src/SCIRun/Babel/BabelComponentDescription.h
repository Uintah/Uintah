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
 *  BabelComponentDescription.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_BabelComponentDescription_h
#define SCIRun_Framework_BabelComponentDescription_h

#include <SCIRun/ComponentDescription.h>

namespace SCIRun {
class BabelComponentModel;

/**
 * \class BabelComponentDescription
 *
 * A refinement of ComponentDescription for the SCIRun Babel component
 * model. Used as a record of information necessary to find and instantiate a
 * particular type of Babel component.
 *
 * See ComponentDescription for more information.
 *
 * \sa ComponentDescription
 */
class BabelComponentDescription : public ComponentDescription
{
public:
  BabelComponentDescription(BabelComponentModel* model);
  virtual ~BabelComponentDescription();

  /** Returns the type name (a string) the component described by this class. */
  virtual std::string getType() const;

  /** Returns a pointer to the component model under which the component type
      is defined. */
  virtual const ComponentModel* getModel() const;
protected:
  friend class BabelComponentModel;
  BabelComponentModel* model;
  std::string type;
  
private:
  BabelComponentDescription(const BabelComponentDescription&);
  BabelComponentDescription& operator=(const BabelComponentDescription&);
};

} //namespace SCIRun

#endif

