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
 *  BabelComponentModel.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   July 2002
 *
 */

#ifndef SCIRun_Babel_BabelComponentModel_h
#define SCIRun_Babel_BabelComponentModel_h

#include <SCIRun/ComponentModel.h>
#include <SCIRun/ComponentInstance.h>
#include <SCIRun/Babel/gov_cca.hh>
#include <string>
#include <map>

namespace SCIRun {

class SCIRunFramework;
class BabelComponentDescription;
class BabelComponentInstance;

/**
 * \class BabelComponentModel
 *
 * A metacomponent model for Babel components.  This class handles the
 * allocation/deallocation of Babel components and maintains a database of
 * Babel components registered in the framework.  See ComponentModel for more
 * information.
 *
 * \sa ComponentModel CCAComponentModel InternalComponentModel VtkComponentModel
 *
 */
class BabelComponentModel : public ComponentModel
{
public:
  BabelComponentModel(SCIRunFramework* framework);
  virtual ~BabelComponentModel();

  /** ? */
  gov::cca::Services createServices(const std::string& instanceName,
                                    const std::string& className,
                                    const gov::cca::TypeMap& properties);

  /** Returns true if component type \em type has been registered with this
      component model.  In other words, returns true if this ComponentModel
      knows how to instantiate component \em type. */
  virtual bool haveComponent(const std::string& type);

  /** Allocates an instance of the component of type \em type.  The parameter
      \em name is assigned as the unique name of the newly created instance.
      Returns a smart pointer to the newly created instance, or a null pointer
      on failure. */
  virtual ComponentInstance* createInstance(const std::string& name,
                                            const std::string& type);

  /** ? */
  virtual std::string createComponent(const std::string& name,
					 const std::string& type);

 /** Deallocates the component instance \em ci.  Returns \code true on success and
     \code false on failure. */
  virtual bool destroyInstance(ComponentInstance *ci);

  /**  Returns the name (as a string) of this component model. */
  virtual std::string getName() const;

  /** Creates a list of all the available components (as ComponentDescriptions)
      registered in this ComponentModel. */
  virtual void listAllComponentTypes(std::vector<ComponentDescription*>&, bool);

  /** ? */
  virtual void destroyComponentList();

  /** ? */
  virtual void buildComponentList();

  static const std::string DEFAULT_PATH;
  
private:
  SCIRunFramework* framework;
  typedef std::map<std::string, BabelComponentDescription*> componentDB_type;
  componentDB_type components;
  
  void readComponentDescription(const std::string& file);

  BabelComponentModel(const BabelComponentModel&);
  BabelComponentModel& operator=(const BabelComponentModel&);
};

} //namespace SCIRun

#endif
