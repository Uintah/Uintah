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


#include <SCIRun/Internal/FrameworkProperties.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/PortInstance.h>
#include <Core/OS/Dir.h>
#include <Core/Util/Environment.h>

#include <unistd.h>

namespace SCIRun {

const char* FrameworkProperties::CONFIG_DIR = "/.sr2";
const char* FrameworkProperties::CONFIG_FILE = "sr2.config";
const char* FrameworkProperties::CACHE_FILE = "sr2.cache";

FrameworkProperties::FrameworkProperties(SCIRunFramework* framework,
					    const std::string& name)
    : InternalComponentInstance(framework, name, "internal:FrameworkProperties")
{
    this->framework = framework;
    frameworkProperties = sci::cca::TypeMap::pointer(new TypeMap);
    frameworkProperties->putString("url", framework->getURL().getString());
    frameworkProperties->putString("default_login", getLogin());
    frameworkProperties->putString("sidl_xml_path", getSidlXMLPath());
}

FrameworkProperties::~FrameworkProperties()
{
}

InternalComponentInstance*
FrameworkProperties::create(SCIRunFramework* framework, const std::string& name)
{
    FrameworkProperties* fp =
	new FrameworkProperties(framework, name);
    fp->addReference();
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

void FrameworkProperties::readPropertiesFromFile()
{
  std::cerr << "FrameworkProperties::readPropertiesFromFile not implemented" << std::endl;
}

std::string FrameworkProperties::getSidlXMLPath()
{
    // ';' seperated list of directories where one can find SIDL xml files
    // getenv may return NULL if SIDL_XML_PATH was not set
    const char *component_path = getenv("SIDL_XML_PATH");
    if (component_path) {
	return std::string(component_path);
    }
    return std::string();
}

std::string FrameworkProperties::getLogin()
{
    char *login = getlogin();
    if (login) {
	return std::string(login);
    }
    return std::string();
}

void FrameworkProperties::writePropertiesToFile()
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
}

}
