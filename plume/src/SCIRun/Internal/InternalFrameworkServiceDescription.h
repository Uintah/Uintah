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
 *  InternalFrameworkServiceDescription.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   August 2005
 *
 */

#ifndef SCIRun_Internal_InternalFrameworkServiceDescription_h
#define SCIRun_Internal_InternalFrameworkServiceDescription_h

#include <SCIRun/Internal/InternalServiceDescription.h>
#include <string>

namespace SCIRun {

  class InternalComponentModel;
  class InternalFrameworkServiceInstance;
  
  /** \class InternalFrameworkServiceDescription
   *
   *
   */
  class InternalFrameworkServiceDescription : public InternalServiceDescription
  {
  public:
    
    InternalFrameworkServiceDescription(
      InternalComponentModel* model,
      const std::string& serviceType,
      InternalFrameworkServiceInstance* (*create)(SCIRunFramework*));
    
    virtual ~InternalFrameworkServiceDescription();
    
    virtual InternalFrameworkServiceInstance *get(SCIRunFramework *);
    virtual void release(SCIRunFramework *);

  private:
    InternalFrameworkServiceInstance* (*create)(SCIRunFramework*);
    InternalFrameworkServiceInstance* singleton_instance;

    InternalFrameworkServiceDescription(const InternalFrameworkServiceDescription&);
    InternalFrameworkServiceDescription& operator=(const InternalFrameworkServiceDescription&);
  };

} // end namespace SCIRun

#endif
