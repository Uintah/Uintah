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
 *  CorbaComponentDescription.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_Corba_CorbaComponentDescription_h
#define SCIRun_Corba_CorbaComponentDescription_h

#include <SCIRun/ComponentDescription.h>
#include <string>

namespace SCIRun
{
  class ComponentModel;
  class CorbaComponentModel;

  /**
   * \class CorbaComponentDescription
   *
   * A container for information necessary to locate and instantiate a specific
   * Corba component type in the SCIRun framework.  This class holds the type name of
   * the component and the CorbaComponentModel instance to which it belongs.  The
   * name of the DLL containing the executable code for this component type is
   * also stored in this class.
   *
   */
  class CorbaComponentDescription : public ComponentDescription
    {
    public:
      CorbaComponentDescription(CorbaComponentModel* model, const std::string& type);
      virtual ~CorbaComponentDescription();

      /** Returns the type name (a string) described by this class. */
      virtual std::string getType() const;
      /** Returns a pointer to the CorbaComponentModel that holds this CorbaComponentDescription.*/
      virtual const ComponentModel* getModel() const;

      std::string type;
  
      /** Get/Set the path of the CORBA executable.*/
      std::string getExecPath() const{ 
	return exec_path; 
      }
      void setExecPath(const std::string &path){
	exec_path=path; 
      }
  
    private:
      CorbaComponentModel* model;
      std::string exec_path;
  
      CorbaComponentDescription(const CorbaComponentDescription&);
      CorbaComponentDescription& operator=(const CorbaComponentDescription&);
    };
}

#endif
