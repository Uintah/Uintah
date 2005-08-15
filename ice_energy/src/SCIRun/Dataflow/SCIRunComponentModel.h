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
 *  SCIRunComponentModel.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_SCIRunComponentModel_h
#define SCIRun_Framework_SCIRunComponentModel_h

#include <SCIRun/ComponentModel.h>


namespace SCIRun {
class Network;
class SCIRunComponentDescription;
class SCIRunFramework;
class TCLInterface;

/**
 * \class SCIRunComponentModel
 *
 * Component model for SCIRun dataflow modules.
 *
 */
class SCIRunComponentModel : public ComponentModel
{
public:
    SCIRunComponentModel(SCIRunFramework* framework);
    virtual ~SCIRunComponentModel();

    /** Returns true if component type \em type has been registered with this
        component model.  In other words, returns true if this ComponentModel
        knows how to instantiate component \em type. */
    virtual bool
    haveComponent(const std::string& type);

    /** Allocates an instance of the component of type \em type.  The parameter
        \em name is assigned as the unique name of the newly created instance.
        Returns a smart pointer to the newly created instance, or a null pointer
        on failure. */
    virtual ComponentInstance*
    createInstance(const std::string& name,
                   const std::string& type,
                   const sci::cca::TypeMap::pointer& tm);

    /** Deallocates the component instance \em ci.
        Returns \code true on success and \code false on failure. */
    virtual bool
    destroyInstance(ComponentInstance * ic);

    /** Returns the name (as a string) of this component model. */ 
    virtual std::string getName() const;

    /** Creates a list of all the available components (as ComponentDescriptions)
        registered in this ComponentModel. */
    virtual void
    listAllComponentTypes(std::vector<ComponentDescription*>&, bool);

    /** Method not implemented by the SCIRunComponentModel. */
        virtual void destroyComponentList();

    /** Method not implemented by the SCIRunComponentModel. */
        virtual void buildComponentList();

    /** Initialize SCIRun GUI elements. */
    static void initGuiInterface();
    static TCLInterface* gui;
    static Network* net;

private:
    SCIRunFramework* framework;

    SCIRunComponentModel(const SCIRunComponentModel&);
    SCIRunComponentModel& operator=(const SCIRunComponentModel&);
};

} // end namespace SCIRun

#endif
