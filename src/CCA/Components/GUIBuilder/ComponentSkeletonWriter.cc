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
 * ComponentSkeletonWriter.cc
 *
 * Written by:
 *  <author>
 *  Scientific Computing and Imaging Institute
 *  University of Utah
 *  <date>
 *
 */


#include<iostream>
#include<fstream>


#include <CCA/Components/GUIBuilder/ComponentSkeletonWriter.h>
#include <Core/Containers/StringUtil.h>
#include <Core/OS/Dir.h>
#include <Core/Util/Environment.h>
#include <SCIRun/Internal/FrameworkProperties.h>
#include <SCIRun/CCA/CCAComponentModel.h>

#ifndef DEBUG
#  define DEBUG 0
#endif

namespace GUIBuilder {

using namespace SCIRun;

const std::string ComponentSkeletonWriter::SP("  ");
const std::string ComponentSkeletonWriter::QT("\"");
const std::string ComponentSkeletonWriter::DIR_SEP("/");
const std::string ComponentSkeletonWriter::OPEN_C_COMMENT("/*");
const std::string ComponentSkeletonWriter::CLOSE_C_COMMENT("*/");
const std::string ComponentSkeletonWriter::UNIX_SHELL_COMMENT("#");
const std::string ComponentSkeletonWriter::NEWLINE("\n");

const std::string ComponentSkeletonWriter::DEFAULT_NAMESPACE("sci::cca::");
const std::string ComponentSkeletonWriter::DEFAULT_SIDL_NAMESPACE("sci.cca.");
const std::string ComponentSkeletonWriter::DEFAULT_PORT_NAMESPACE("sci::cca::ports::");
const std::string ComponentSkeletonWriter::DEFAULT_SIDL_PORT_NAMESPACE("sci.cca.ports.");

ComponentSkeletonWriter::ComponentSkeletonWriter(const std::string &cname,
                                                 const std::string& dir,
                                                 const std::vector<PortDescriptor*> pp,
                                                 const std::vector<PortDescriptor*> up)
  : SERVICES_POINTER(DEFAULT_NAMESPACE + "Services::pointer"),
    TYPEMAP_POINTER(DEFAULT_NAMESPACE + "TypeMap::pointer"),
    compName(cname),
    directory(dir),
    providesPortsList(pp),
    usesPortsList(up)
{
  Dir d(directory);
  if (! d.exists()) {
    // need to replace this with CCA exception?
    std::string msg("Directory " + directory + " does not exist.");
    throw InternalError(msg, __FILE__, __LINE__);
  }
}

void ComponentSkeletonWriter::GenerateCode()
{
  // Header file
  std::string sopf(directory + DIR_SEP + compName + ".h");
  componentHeaderFile.open(sopf.c_str());
  ComponentClassDefinitionCode(componentHeaderFile);
  componentHeaderFile.close();

  // Implementation file
  std::string sosf(directory + DIR_SEP + compName + ".cc");
  componentSourceFile.open(sosf.c_str());
  ComponentSourceFileCode(componentSourceFile);
  componentSourceFile.close();

  // Makefile fragment
  std::string somf(directory + DIR_SEP + "sub.mk");
  componentMakefile.open(somf.c_str());
  ComponentMakefileCode(componentMakefile);
  componentMakefile.close();
}

void ComponentSkeletonWriter::GenerateTempCode(const std::string& tempHeaderFile,
                                               const std::string& tempSourceFile,
                                               const std::string& tempSubmakeFile)
{
  // TODO: error checking for ofstream?

  // Header file
  std::string sopf(directory + DIR_SEP + tempHeaderFile);
  std::ofstream tempcomponentHeaderFile;
  tempcomponentHeaderFile.open(sopf.c_str());
  ComponentClassDefinitionCode(tempcomponentHeaderFile);
  tempcomponentHeaderFile.close();

  // Implementation file
  std::string sosf(directory + DIR_SEP + tempSourceFile);
  std::ofstream tempcomponentSourceFile;
  tempcomponentSourceFile.open(sosf.c_str());
  ComponentSourceFileCode(tempcomponentSourceFile);
  tempcomponentSourceFile.close();

  // Makefile fragment
  std::string somf(directory + DIR_SEP + tempSubmakeFile);
  std::ofstream tempcomponentMakefile;
  tempcomponentMakefile.open(somf.c_str());
  ComponentMakefileCode(tempcomponentMakefile);
  tempcomponentMakefile.close();

}
void ComponentSkeletonWriter::ComponentClassDefinitionCode(std::ofstream& fileStream)
{
  writeLicense(fileStream);
  writeHeaderInit(fileStream);
  writeComponentDefinitionCode(fileStream);
  writePortClassDefinitionCode(fileStream);
}

void ComponentSkeletonWriter::ComponentSourceFileCode(std::ofstream& fileStream)
{
  writeLicense(fileStream);
  writeSourceInit(fileStream);
  writeLibraryHandle(fileStream);
  writeSourceClassImpl(fileStream);
}


//////////////////////////////////////////////////////////////////////////
// private member functions

// Add license to file; the license text is identical for both the header
// and implementation files, so the file stream for each file is the
// function arguement.
// TODO: The license text should be read from file.
void ComponentSkeletonWriter::writeLicense(std::ofstream& fileStream)
{
  std::ifstream lt;
  std::string currPath(sci_getenv("SCIRUN_SRCDIR") + CCAComponentModel::DEFAULT_PATH + DIR_SEP + "license.txt");
  lt.open(currPath.c_str());
  if (! lt) {
#if DEBUG
    std::cout << "unable to read file" << std::endl;
#endif
  } else {
    std::string line;
    fileStream << OPEN_C_COMMENT << std::endl;
    while (!lt.eof()) {
      std::getline (lt,line);
      fileStream << SP << line << std::endl;
    }
    fileStream << CLOSE_C_COMMENT << std::endl;
    lt.close();
  }

}

void ComponentSkeletonWriter::writeMakefileLicense(std::ofstream& fileStream)
{
  std::ifstream lt;
  std::string currPath(sci_getenv("SCIRUN_SRCDIR") + CCAComponentModel::DEFAULT_PATH + DIR_SEP + "license.txt");
  lt.open(currPath.c_str());
  if (! lt) {
#if DEBUG
    std::cout << "unable to read file" << std::endl;
#endif
  } else {
    std::string line;
    while (!lt.eof()) {
      std::getline (lt,line);
      fileStream << UNIX_SHELL_COMMENT << SP << line << std::endl;
    }
    fileStream << std::endl;
    lt.close();
  }
}

void ComponentSkeletonWriter::writeHeaderInit(std::ofstream& fileStream)
{
  fileStream << std::endl;
  fileStream << std::endl;
  fileStream << "#ifndef SCIRun_Framework_" << compName << "_h" << std::endl;
  fileStream << "#define SCIRun_Framework_" << compName << "_h" << std::endl;
  fileStream << std::endl << "#include <Core/CCA/spec/cca_sidl.h>" << std::endl;

  fileStream << std::endl;
  fileStream << "using namespace SCIRun;" << std::endl;
}

void ComponentSkeletonWriter::writeComponentDefinitionCode(std::ofstream& fileStream)
{
  fileStream << std::endl;
  fileStream << "class " << compName << ": public " << DEFAULT_NAMESPACE << "Component {"
             << std::endl;
  // public members
  fileStream << "public:" << std::endl;
  fileStream << SP << compName << "();" << std::endl;
  fileStream << SP << "virtual ~"<< compName << "();" << std::endl;
  fileStream << SP << "virtual void setServices(const " << SERVICES_POINTER << "& svc);" << std::endl;
  // private members
  fileStream << std::endl;
  fileStream << "private:" << std::endl;
  fileStream << SP << compName << "(const " << compName << "&);" << std::endl;
  fileStream << SP << compName << "& operator=(const " << compName << "&);" << std::endl;
  // services handle
  fileStream << SP << SERVICES_POINTER << " services;" << std::endl;
  fileStream << "};" << std::endl;
  fileStream << std::endl;
}

void ComponentSkeletonWriter::writePortClassDefinitionCode(std::ofstream& fileStream)
{
  for (unsigned int i = 0; i < providesPortsList.size(); i++) {

    fileStream << std::endl;
    fileStream << "class " << (providesPortsList[i])->GetClassName() << " : public "
               << DEFAULT_PORT_NAMESPACE << (providesPortsList[i])->GetType() << " {"
               << std::endl;

    //public  members
    fileStream << "public:" << std::endl;
    fileStream << SP << "virtual ~"<< (providesPortsList[i])->GetClassName() << "() {}" << std::endl;
    fileStream << SP << "void setParent(" << compName << " *com)" << " { this->com = com; }";
    if ((providesPortsList[i])->GetType() == "GoPort") {
      fileStream <<std::endl<< SP << "virtual int go();";
    }

    if ((providesPortsList[i])->GetType() == "UIPort") {
      fileStream <<std::endl << SP << "virtual int ui();";
    }
    // private  members
    fileStream << std::endl;
    fileStream << std::endl << "private:" << std::endl;
    fileStream << SP << compName << " *com;" << std::endl;
    fileStream << std::endl << "};" << std::endl;
  }
  fileStream << std::endl;
  fileStream << "#endif" << std::endl;
}


void ComponentSkeletonWriter::writeSourceInit(std::ofstream& fileStream)
{
  fileStream << std::endl << std::endl;

  //Header files
  fileStream << "#include<CCA/Components" << DIR_SEP << compName << DIR_SEP << compName << ".h>" << std::endl;
  fileStream << "#include <SCIRun" << DIR_SEP << "TypeMap.h>" << std::endl;
  fileStream << std::endl;
  fileStream << "using namespace SCIRun;" << std::endl;
}

void ComponentSkeletonWriter::writeLibraryHandle(std::ofstream& fileStream)
{
  fileStream << std::endl;
  fileStream << "extern " << QT<< "C" << QT << " " << DEFAULT_NAMESPACE << "Component::pointer make_SCIRun_" << compName << "()" << std::endl;
  fileStream << "{" << std::endl;
  fileStream << SP << "return " << DEFAULT_NAMESPACE << "Component::pointer(new " << compName << "());" << std::endl;
  fileStream << "}" << std::endl;
  fileStream << std::endl;
}


void ComponentSkeletonWriter::writeSourceClassImpl(std::ofstream& fileStream)
{
  writeConstructorandDestructorCode(fileStream);
  writeSetServicesCode(fileStream);
  writeGoAndUIFunctionsCode(fileStream);
}


void ComponentSkeletonWriter::writeConstructorandDestructorCode(std::ofstream& fileStream)
{
  //Constructor code
  fileStream << std::endl << compName << "::" << compName << "()" << std::endl
             << "{" << std::endl
             << "}" << std::endl;
  //Destructor code
  fileStream << std::endl << compName << "::~" << compName << "()" << std::endl
             << "{" << std::endl;
  for (unsigned int i = 0; i < providesPortsList.size(); i++) {
    fileStream << SP << "services->removeProvidesPort("
               << QT << providesPortsList[i]->GetUniqueName() <<  QT << ");" << std::endl;
  }

  for (unsigned int i = 0; i < usesPortsList.size(); i++) {
    fileStream << SP << "services->unregisterUsesPort("
               << QT << usesPortsList[i]->GetUniqueName() << QT << ");" << std::endl;
  }
  fileStream << "}" << std::endl;
}


void ComponentSkeletonWriter::writeSetServicesCode(std::ofstream& fileStream)
{
  fileStream << std::endl
             << "void " << compName << "::setServices(const " << SERVICES_POINTER << "& svc)"<< std::endl;
  fileStream << "{" << std::endl;
  fileStream << SP << "services = svc;" << std::endl;
  //fileStream << std::endl << SP << "svc->registerForRelease(sci::cca::ComponentRelease::pointer(this)); ";

  for (unsigned int i = 0; i < providesPortsList.size(); i++) {
    std::string portName(providesPortsList[i]->GetClassName());
    std::string portType(providesPortsList[i]->GetType());
    std::string tempPortInstance, tempPortPtr, tempPortCategory;

    for (unsigned int j = 0; j < portType.length(); j++) {
      char tmp = portType.at(j);

      // TODO: Unicode safe?
      if ((tmp >= 'A') && (tmp <= 'Z')) {
        tempPortInstance.append(1, (char) tolower(tmp));
      }
    }

    tempPortInstance = "provides" + tempPortInstance;
    tempPortCategory = (std::string) providesPortsList[i]->GetUniqueName();
    tempPortPtr = portName + "::pointer(" + tempPortInstance + ")";

    fileStream << SP << portName << " *" << tempPortInstance
               << " = new " << portName << "();" << std::endl;
    fileStream << SP << tempPortInstance << "->setParent(this);" << std::endl;

    std::string propertiesMap("pProps");
    fileStream << SP << TYPEMAP_POINTER << " " << propertiesMap << i << " = svc->createTypeMap();" << std::endl;

    fileStream << SP << "svc->addProvidesPort(" << tempPortPtr
               << ", " << QT << tempPortCategory << QT
               << ", " << QT << DEFAULT_SIDL_PORT_NAMESPACE << portType << QT
               << ", " << propertiesMap << i << ");" << std::endl;

    fileStream << std::endl;
  }

  for (unsigned int i = 0; i < usesPortsList.size(); i++) {
    std::string portName(usesPortsList[i]->GetClassName());
    std::string portType(usesPortsList[i]->GetType());
    std::string tempPortCategory;

    char tmp = portType.at(0);
    tempPortCategory = portType.substr(1, portType.length());
    tempPortCategory = "uses" + tempPortCategory.insert(0, 1, (char) tolower(tmp));

    std::string propertiesMap("uProps");
    fileStream << SP << TYPEMAP_POINTER << " " << propertiesMap << i << " = svc->createTypeMap();" << std::endl;

    fileStream << SP << "svc->registerUsesPort(" << QT << tempPortCategory << QT
               << ", " << QT << DEFAULT_SIDL_PORT_NAMESPACE << portType << QT
               << ", " << propertiesMap << i << ");" << std::endl;
  }

  fileStream << "}" << std::endl;
}

//go() and ui() functions - if thers is a GoPort or UIPort among Provides ports
void ComponentSkeletonWriter::writeGoAndUIFunctionsCode(std::ofstream& fileStream)
{

  for (unsigned int i = 0; i < providesPortsList.size(); i++) {
    string portname(providesPortsList[i]->GetType());
    string porttype(providesPortsList[i]->GetType());

#if DEBUG
    std::cout << "\nhere in ckw " << porttype.c_str() << "\n";
#endif
    if (porttype.compare(string("UIPort")) == 0) {
      fileStream << std::endl <<"int " << providesPortsList[i]->GetClassName() << "::ui()"
                 << std::endl
                 << "{"
                 << std::endl << SP << "return 0;"
                 << std::endl
                 << "}" << std::endl;
    }

    if (porttype.compare(string("GoPort")) == 0) {
      fileStream << std::endl <<"int " << providesPortsList[i]->GetClassName() << "::go()"
                 << std::endl
                 << "{"
                 << std::endl << SP << "return 0;"
                 << std::endl
                 << "}" << std::endl;
    }
  }
  fileStream << std::endl;
}

void ComponentSkeletonWriter::ComponentMakefileCode(std::ofstream& fileStream)
{
  writeMakefileLicense(fileStream);
  fileStream << "include $(SCIRUN_SCRIPTS)/smallso_prologue.mk" << std::endl;
  fileStream << std::endl
             << "SRCDIR := CCA/Components/" << compName << std::endl;
  fileStream << std::endl
             << "SRCS += " << "$(SRCDIR)/" << compName << ".cc \\" << std::endl;
  fileStream << std::endl
             << "PSELIBS := Core/CCA/SSIDL Core/CCA/PIDL Core/CCA/Comm \\" << std::endl;
  fileStream << SP << "Core/CCA/spec Core/Thread Core/Containers Core/Exceptions" << std::endl;
  fileStream << std::endl
             << "CFLAGS += $(WX_CXXFLAGS)" << std::endl
             << "CXXFLAGS += $(WX_CXXFLAGS)" << std::endl
             << "LIBS := $(WX_LIBRARY)" << std::endl;
  fileStream << std::endl << "include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk" << std::endl;
  fileStream << std::endl << "$(SRCDIR)/" << compName << ".o: Core/CCA/spec/cca_sidl.h" << std::endl;
  fileStream << std::endl;
}

}
