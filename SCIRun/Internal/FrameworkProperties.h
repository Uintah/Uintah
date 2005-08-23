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
 *  FrameworkProperties.h:  get and set CCA framework properties
 *
 *  Written by:
 *   Ayla Khan
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   September 2004
 *
 *   Copyright (C) 2004 SCI Institute
 */


#ifndef SCIRun_FrameworkProperties_h
#define SCIRun_FrameworkProperties_h

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/TypeMap.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalFrameworkServiceInstance.h>

namespace SCIRun {

class SCIRunFramework;

/**
 * An implementation of the sci::cca::ports::FrameworkProperties
 * interface.
 * Provides access to framework properties that need to be exposed to
 * components such as:
 *    key                    value          meaning
 *    url                    string         this framework's url
 *    network file           string         network file to be loaded by a builder
 */
class FrameworkProperties : public ::sci::cca::ports::FrameworkProperties,
			    public InternalFrameworkServiceInstance
{
public:
    virtual ~FrameworkProperties();

    /**
      * Factory method used to create a FrameworkProperties instance.
      * Returns a reference counted pointer to a newly-allocated FrameworkProperties
      * port.  The \em framework parameter is a pointer to the relevent framework
      * and the \em name parameter will become the unique name for the new port.
      */
  static InternalFrameworkServiceInstance* create(SCIRunFramework* framework);

    virtual sci::cca::Port::pointer getService(const std::string& name);

    /** Get a smart pointer to a TypeMap containing CCA framework properties. */
    virtual sci::cca::TypeMap::pointer getProperties();

    /** Set CCA framework properties from a TypeMap. */
    virtual void setProperties(const sci::cca::TypeMap::pointer& properties);

    static std::string CONFIG_DIR;
    static std::string CONFIG_FILE;
    static std::string CACHE_FILE;

private:
    FrameworkProperties(SCIRunFramework* framework);

    /** Gets user logged in on the controlling terminal of the process
        or a null pointer (see man getlogin(3)) */
    void getLogin();

    /**
     * Get SIDL file paths from (in order of processing):
     * the user's environment, CONFIG_FILE, component model defaults.
     *
     * Framework properties:
     * "sidl_xml_path": a ';' seperated list of directories where XML based
     *                  descriptions of components can be found.
     */
    void getSidlPaths();

    void parseEnvVariable(std::string& input, const char token,
                          SSIDL::array1<std::string>& stringArray);

    /** Persistent framework properties from file. */
    bool readPropertiesFromFile();

    /** Persistent framework properties from file. */
    bool writePropertiesToFile();

    sci::cca::TypeMap::pointer frameworkProperties;
};

}

#endif
