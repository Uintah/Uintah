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
 *  CCAComponentDescription.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_CCAComponentDescription_h
#define SCIRun_Framework_CCAComponentDescription_h

#include <SCIRun/ComponentDescription.h>
#include <Core/CCA/spec/cca_sidl.h>

namespace SCIRun {

class CCAComponentModel;

/**
 * \class CCAComponentDescription
 *
 * A refinement of ComponentDescription for the CCA Component model.  See
 * ComponentDescription for more information.
 *
 * \sa BabelComponentDescription ComponentDescription VtkComponentDescription
 * \sa InternalComponentDescription
 */
class CCAComponentDescription : public ComponentDescription
{
public:
  CCAComponentDescription(CCAComponentModel* model);
  virtual ~CCAComponentDescription();

  /** Returns the component type name (a string). */
  virtual std::string getType() const;
  /** Returns a pointer to the component model type. */
  virtual const ComponentModel* getModel() const;
  /** ? */
  virtual std::string getLoaderName() const;
  /** ? */
  void setLoaderName(const std::string& loaderName);

  /** Get/Set the name of the DLL for this component.  The loader will search
   *    the SIDL_DLL_PATH for a matching library name. */
  std::string getLibrary() const
  { return library; }
  void setLibrary(const std::string &l)
  { library = l; }

protected:
  friend class CCAComponentModel;
  friend class SCIRunLoader;
  CCAComponentModel* model;
  std::string type;
  std::string loaderName;
private:
  CCAComponentDescription(const CCAComponentDescription&);
  CCAComponentDescription& operator=(const CCAComponentDescription&);
  std::string library;
};

} // end namespace SCIRun

#endif
