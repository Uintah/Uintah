/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#include <Core/CCA/spec/cca_sidl.h>
#include <Framework/Internal/InternalFrameworkServiceInstance.h>

namespace SCIRun {

class SCIRunFramework;

/**
 * \class FrameworkProxyService
 *
 */
class FrameworkProxyService : public sci::cca::ports::FrameworkProxyService,
                              public InternalFrameworkServiceInstance {
public:
    virtual ~FrameworkProxyService();

    /** Factory method for creating an instance of a FrameworkProxyService class.
        Returns a reference counted pointer to a newly-allocated BuilderService
        port.  The \em framework parameter is a pointer to the relevent framework
        and the \em name parameter will become the unique name for the new port.*/
    static InternalFrameworkServiceInstance* create(SCIRunFramework* framework);

    virtual sci::cca::ComponentID::pointer
    createInstance(const std::string& instanceName,
                   const std::string& className,
                   const sci::cca::TypeMap::pointer &properties);

    /** */
    int addComponentClasses(const std::string &loaderName);

    /** */
    int removeComponentClasses(const std::string &loaderName);

    /** */
    virtual int
    addLoader(const std::string &loaderName, const std::string &user,
                const std::string &domain, const std::string &loaderPath);

    /** */
    virtual int removeLoader(const std::string &name);

    /** */
    virtual sci::cca::Port::pointer
    getService(const std::string &);

  //virtual void registerFramework(const std::string &frameworkURL);
  //virtual void registerServices(const sci::cca::Services::pointer &svc);

private:
    FrameworkProxyService(SCIRunFramework* fwk);

    // used by registerFramework & registerServices methods
    //std::vector<sci::cca::Services::pointer> servicesList;
};

}
