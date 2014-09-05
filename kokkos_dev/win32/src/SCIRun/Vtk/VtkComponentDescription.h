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
 *  VtkComponentDescription.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_Vtk_VtkComponentDescription_h
#define SCIRun_Vtk_VtkComponentDescription_h

#include <SCIRun/ComponentDescription.h>
#include <string>

namespace SCIRun
{
class ComponentModel;
class VtkComponentModel;

/**
 * \class VtkComponentDescription
 *
 * A container for information necessary to locate and instantiate a specific
 * Vtk component type in the SCIRun framework.  This class holds the type name of
 * the component and the VtkComponentModel instance to which it belongs.  The
 * name of the DLL containing the executable code for this component type is
 * also stored in this class.
 *
 */
class VtkComponentDescription : public ComponentDescription {
public:
  VtkComponentDescription(VtkComponentModel* model, const std::string& type, const std::string& library = "");
  virtual ~VtkComponentDescription();

  /** Returns the type name (a string) described by this class. */
  virtual std::string getType() const;

  /** Returns a pointer to the VtkComponentModel that holds this VtkComponentDescription.*/
  virtual const ComponentModel* getModel() const;

  /** Get/Set the name of the DLL for this component.  The loader will search
   *    the SIDL_DLL_PATH for a matching library name. */
  std::string getLibrary() const { return library; }
  void setLibrary(const std::string &l) { library = l; }
  
private:
  std::string library;
  
  VtkComponentDescription(const VtkComponentDescription&);
  VtkComponentDescription& operator=(const VtkComponentDescription&);
};
}

#endif
