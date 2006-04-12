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
 *  FrameworkProperties.cc: get and set CCA framework properties
 *
 *  Written by:
 *   Ayla Khan
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   September 2004
 *
 *   Copyright (C) 2004 SCI Institute
 */

#include <sci_metacomponents.h>

#include <SCIRun/Internal/FrameworkProperties.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/PortInstance.h>

#include <Core/OS/Dir.h>
#include <Core/Util/Environment.h>

#include <iostream>
#include <unistd.h>

namespace SCIRun {

std::string FrameworkProperties::CONFIG_DIR("/.sr2");
std::string FrameworkProperties::CONFIG_FILE("sr2rc");
std::string FrameworkProperties::CACHE_FILE("sr2.cache");

FrameworkProperties::FrameworkProperties(SCIRunFramework* framework)
  : InternalFrameworkServiceInstance(framework, "internal:FrameworkProperties")
{
    frameworkProperties = sci::cca::TypeMap::pointer(new TypeMap);
    frameworkProperties->putString("url", framework->getURL().getString());
    getLogin();
    initSidlPaths();
}

FrameworkProperties::~FrameworkProperties()
{
}

InternalFrameworkServiceInstance*
FrameworkProperties::create(SCIRunFramework* framework)
{
    FrameworkProperties* fp = new FrameworkProperties(framework);
    return fp;
}

sci::cca::TypeMap::pointer
FrameworkProperties::getProperties()
{
    return frameworkProperties;
}

void
FrameworkProperties::setProperties(const sci::cca::TypeMap::pointer& properties)
{
    frameworkProperties = properties;
}

sci::cca::Port::pointer FrameworkProperties::getService(const std::string& name)
{
    return sci::cca::Port::pointer(this);
}

void FrameworkProperties::initSidlPaths()
{
    std::string srcDir(sci_getenv("SCIRUN_SRCDIR"));
    std::string fullCCAXMLPath(srcDir + CCAComponentModel::DEFAULT_PATH);
    SSIDL::array1<std::string> sArray;

    // ';' seperated list of directories where one can find SIDL xml files
    // getenv may return NULL if SIDL_XML_PATH was not set
    const char *component_path = getenv("SIDL_XML_PATH");
    if (component_path) {
        std::string s(component_path);
        parseEnvVariable(s, ';', sArray);
	// Check to make sure that the default CCA path is in the array.
	// If not found, add it.
	bool found = false;
	for (SSIDL::array1<std::string>::iterator iter = sArray.begin();
	     iter != sArray.end(); iter++) {
	  if (*iter == fullCCAXMLPath) {
	    found = true;
	    break;
	  }
	}
	if (! found) {
	  sArray.push_back(fullCCAXMLPath);
	}
        frameworkProperties->putStringArray("sidl_xml_path", sArray);
    } else if (readPropertiesFromFile()) {
        return;
    } else {
        sArray.push_back(fullCCAXMLPath);
#if HAVE_BABEL
        sArray.push_back(srcDir + BabelComponentModel::DEFAULT_PATH);
#endif
#if HAVE_VTK
        sArray.push_back(srcDir + VtkComponentModel::DEFAULT_PATH);
#endif

#if HAVE_TAO
        sArray.push_back(srcDir + CorbaComponentModel::DEFAULT_PATH);
        sArray.push_back(srcDir + TaoComponentModel::DEFAULT_PATH);
#endif
        frameworkProperties->putStringArray("sidl_xml_path", sArray);
    }
    // SIDL_DLL_PATH env. variable is read and parsed in VtkComponentModel etc.
}

void FrameworkProperties::parseEnvVariable(std::string& input,
                                           const char token,
                                           SSIDL::array1<std::string>& stringArray)
{
    std::string::size_type i = 0;
    while ( i != std::string::npos ) {
        i = input.find(token);
        if (i < input.size()) {
            stringArray.push_back(input.substr(0, i));
            input = input.substr(i + 1, input.size());
        } else {
            stringArray.push_back(input);
        }
    }
}

void FrameworkProperties::getLogin()
{
    char *login = getlogin();
    if (login) {
        frameworkProperties->putString("default_login", std::string(login));
    } else {
        frameworkProperties->putString("default_login", std::string());
    }
}

bool FrameworkProperties::readPropertiesFromFile()
{
    char *HOME = getenv("HOME");
    std::string name(HOME);
    sci_putenv("HOME", HOME);

    name += CONFIG_DIR + "/" + CONFIG_FILE;
    SSIDL::array1<std::string> sArray;
    if (parse_scirunrc(name)) {
        const char *dll_path = sci_getenv("SIDL_DLL_PATH");
        if (dll_path != 0) {
            frameworkProperties->putString("sidl_dll_path", dll_path);
        }

        const char *xml_path = sci_getenv("SIDL_XML_PATH");
        if (xml_path != 0) {
            std::string s(xml_path);
            parseEnvVariable(s, ';', sArray);
            frameworkProperties->putStringArray("sidl_xml_path", sArray);
        }
        return true;
    }
    return false;
}

bool FrameworkProperties::writePropertiesToFile()
{
    char *HOME = getenv("HOME");
    std::string name(HOME);
    name.append(CONFIG_DIR);
    Dir dir(name);
    if (! dir.exists()) {
        Dir::create(name);
    }
    // get file
    // write to file
    return false;
}

}
