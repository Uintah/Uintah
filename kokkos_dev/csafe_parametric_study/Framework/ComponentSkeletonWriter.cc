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


#include <iostream>
#include <fstream>
#include <sstream>

#include <Framework/ComponentSkeletonWriter.h>
#include <Core/Containers/StringUtil.h>
#include <Core/OS/Dir.h>
#include <Core/Util/Environment.h>
#include <Framework/Internal/FrameworkProperties.h>
#include <Framework/CCA/CCAComponentModel.h>


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

ComponentSkeletonWriter::ComponentSkeletonWriter()
{
}

ComponentSkeletonWriter::ComponentSkeletonWriter(const std::string &cname,
                                                 const std::string& dir,
                                                 const std::vector<PortDescriptor*> pp,
                                                 const std::vector<PortDescriptor*> up,
                                                 const bool& iws)
  : SERVICES_POINTER(DEFAULT_NAMESPACE + "Services::pointer"),
    TYPEMAP_POINTER(DEFAULT_NAMESPACE + "TypeMap::pointer"),
    compName(cname),
    directory(dir),
    providesPortsList(pp),
    usesPortsList(up),
    isWithSidl(iws)
{
  Dir d(directory);
  if (! d.exists()) {
    // need to replace this with CCA exception?
    std::string msg("Directory " + directory + " does not exist.");
    throw InternalError(msg, __FILE__, __LINE__);
  }
}


std::string ComponentSkeletonWriter::getTempDirName()
{
  std::string home (getenv("HOME"));
  std::string tmp(".sr2");
  std::string tmpDirName = std::string(home + DIR_SEP + tmp  + DIR_SEP + "ComponentGenerationWizard");
  return tmpDirName;
}

std::string ComponentSkeletonWriter::getCompDirName()
{
  return directory;
}

void ComponentSkeletonWriter::generate(std::string headerFilename, std::string sourceFilename, std::string makeFilename)
{
  // TODO: error checking for ofstream?
  // Header file
  std::ofstream headerFile;
  headerFile.open(headerFilename.c_str());
  ComponentClassDefinitionCode(headerFile);
  headerFile.close();

  // Implementation file
  std::ofstream sourceFile;
  sourceFile.open(sourceFilename.c_str());
  ComponentSourceFileCode(sourceFile);
  sourceFile.close();

  // Makefile fragment
  std::ofstream makeFile;
  makeFile.open(makeFilename.c_str());
  ComponentMakefileCode(makeFile);
  makeFile.close();
}

void ComponentSkeletonWriter::GenerateWithSidl(std::string headerFilename, std::string sourceFilename, std::string makeFilename, std::string sidlFilename)
{
  std::ofstream sidlFile;
  sidlFile.open(sidlFilename.c_str());
  writeSidlFile(sidlFile);
  sidlFile.close();
  generate(headerFilename,sourceFilename,makeFilename);
}

void ComponentSkeletonWriter::ComponentClassDefinitionCode(std::ofstream& fileStream)
{
  writeLicense(fileStream);
  writeHeaderInit(fileStream);
  writeComponentDefinitionCode(fileStream);
  if(!isWithSidl)
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
    fileStream << OPEN_C_COMMENT << NEWLINE;
    while (!lt.eof()) {
      std::getline (lt,line);
      fileStream << SP << line << NEWLINE;
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
      fileStream << UNIX_SHELL_COMMENT << SP << line << NEWLINE;
    }
    fileStream << NEWLINE;
    lt.close();
  }
}

void ComponentSkeletonWriter::writeHeaderInit(std::ofstream& fileStream)
{
  fileStream << NEWLINE << NEWLINE;
  fileStream << "#ifndef SCIRun_Framework_" << compName << "_h" << NEWLINE;
  fileStream << "#define SCIRun_Framework_" << compName << "_h" << NEWLINE;
  fileStream << NEWLINE << "#include <Core/CCA/spec/cca_sidl.h>" << NEWLINE;
  if(isWithSidl)
    fileStream  << "#include <CCA/Components/" << compName << DIR_SEP << compName << "_sidl.h>" << NEWLINE;
  fileStream << NEWLINE;
  fileStream << "using namespace SCIRun;" << std::endl;
}

void ComponentSkeletonWriter::writeComponentDefinitionCode(std::ofstream& fileStream)
{
  fileStream << NEWLINE;
  if (isWithSidl) {
    fileStream << "class " << compName << ": public " << DEFAULT_NAMESPACE << compName << " {" << NEWLINE;
  } else {
    fileStream << "class " << compName << ": public " << DEFAULT_NAMESPACE << "Component {" << NEWLINE;
  }
  // public members
  fileStream << "public:" << NEWLINE;
  fileStream << SP << compName << "();" << NEWLINE;
  fileStream << SP << "virtual ~"<< compName << "();" << NEWLINE;
  fileStream << SP << "virtual void setServices(const " << SERVICES_POINTER << "& svc);" << NEWLINE;
  fileStream << SP << SERVICES_POINTER << "&  getServices() { return services; }" << NEWLINE;

  //If With Sidl
  if (isWithSidl) {
    for (unsigned int i = 0; i < providesPortsList.size(); i++) {
      if ((providesPortsList[i])->GetType() == "GoPort") {
        fileStream << SP << "virtual int go();" << NEWLINE;
      }
    }
  }

  // private members
  fileStream << NEWLINE;
  fileStream << "private:" << NEWLINE;
  fileStream << SP << compName << "(const " << compName << "&);" << NEWLINE;
  fileStream << SP << compName << "& operator=(const " << compName << "&);" << NEWLINE;
  // services handle
  fileStream << SP << SERVICES_POINTER << " services;" << NEWLINE;
  fileStream << "};" << NEWLINE;
  if (isWithSidl) {
    fileStream << "#endif" << NEWLINE;
  }
  fileStream << std::endl;
}


void ComponentSkeletonWriter::writePortClassDefinitionCode(std::ofstream& fileStream)
{
  for (unsigned int i = 0; i < providesPortsList.size(); i++) {
    fileStream << NEWLINE;
    fileStream << "class " << (providesPortsList[i])->GetClassName() << " : public "
               << DEFAULT_PORT_NAMESPACE << (providesPortsList[i])->GetType() << " {"
               << NEWLINE;

    //public  members
    fileStream << "public:" << NEWLINE;
    fileStream << SP << "virtual ~"<< (providesPortsList[i])->GetClassName() << "() {}" << NEWLINE;
    fileStream << SP << "void setParent(" << compName << " *com)" << " { this->com = com; }";
    if ((providesPortsList[i])->GetType() == "GoPort") {
      fileStream <<NEWLINE<< SP << "virtual int go();";
    }

    if ((providesPortsList[i])->GetType() == "UIPort") {
      fileStream <<NEWLINE << SP << "virtual int ui();";
    }
    // private  members
    fileStream << NEWLINE;
    fileStream << NEWLINE << "private:" << NEWLINE;
    fileStream << SP << compName << " *com;" << NEWLINE;
    fileStream << NEWLINE << "};" << NEWLINE;
  }
  fileStream << NEWLINE;
  fileStream << "#endif" << std::endl;
}


void ComponentSkeletonWriter::writeSourceInit(std::ofstream& fileStream)
{
  fileStream << NEWLINE << NEWLINE;

  //Header files
  fileStream << "#include<CCA/Components" << DIR_SEP << compName << DIR_SEP << compName << ".h>" << NEWLINE;
  fileStream << "#include <SCIRun" << DIR_SEP << "TypeMap.h>" << NEWLINE;
  fileStream << NEWLINE;
  fileStream << "using namespace SCIRun;" << std::endl;
}

void ComponentSkeletonWriter::writeLibraryHandle(std::ofstream& fileStream)
{
  fileStream << NEWLINE;
  fileStream << "extern " << QT<< "C" << QT << " " << DEFAULT_NAMESPACE << "Component::pointer make_SCIRun_" << compName << "()" << NEWLINE;
  fileStream << "{" << NEWLINE;
  fileStream << SP << "return " << DEFAULT_NAMESPACE << "Component::pointer(new " << compName << "());" << NEWLINE;
  fileStream << "}" << NEWLINE;
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
  fileStream << NEWLINE << compName << "::" << compName << "()" << NEWLINE
             << "{" << NEWLINE
             << "}" << NEWLINE;
  //Destructor code
  fileStream << NEWLINE << compName << "::~" << compName << "()" << NEWLINE
             << "{" << NEWLINE;
  for (unsigned int i = 0; i < providesPortsList.size(); i++) {
    fileStream << SP << "services->removeProvidesPort("
               << QT << providesPortsList[i]->GetUniqueName() <<  QT << ");" << NEWLINE;
  }

  for (unsigned int i = 0; i < usesPortsList.size(); i++) {
    fileStream << SP << "services->unregisterUsesPort("
               << QT << usesPortsList[i]->GetUniqueName() << QT << ");" << NEWLINE;
  }
  fileStream << "}" << std::endl;
}


void ComponentSkeletonWriter::writeSetServicesCode(std::ofstream& fileStream)
{
  fileStream << NEWLINE
             << "void " << compName << "::setServices(const " << SERVICES_POINTER << "& svc)"<< NEWLINE;
  fileStream << "{" << NEWLINE;
  fileStream << SP << "services = svc;" << NEWLINE;
  //fileStream << std::endl << SP << "svc->registerForRelease(sci::cca::ComponentRelease::pointer(this)); ";

  for (unsigned int i = 0; i < providesPortsList.size(); i++) {
    std::string portName(providesPortsList[i]->GetClassName());
    std::string portType(providesPortsList[i]->GetType());
    std::string tempPortInstance, tempPortPtr, tempPortCategory;
    tempPortCategory = (std::string) providesPortsList[i]->GetUniqueName();
    if (!isWithSidl) {
      for (unsigned int j = 0; j < portType.length(); j++) {
        char tmp = portType.at(j);

        // TODO: Unicode safe?
        if ((tmp >= 'A') && (tmp <= 'Z')) {
          tempPortInstance.append(1, (char) tolower(tmp));
        }
      }
      std::stringstream appendCount;
      appendCount << i;
      tempPortInstance = "provides" + tempPortInstance + appendCount.str();

      tempPortPtr = portName + "::pointer(" + tempPortInstance + ")";

      fileStream << SP << portName << " *" << tempPortInstance
                 << " = new " << portName << "();" << NEWLINE;
      fileStream << SP << tempPortInstance << "->setParent(this);" << NEWLINE;

      std::string propertiesMap("pProps");
      fileStream << SP << TYPEMAP_POINTER << " " << propertiesMap << i << " = svc->createTypeMap();" << NEWLINE;

      fileStream << SP << "svc->addProvidesPort(" << tempPortPtr
                 << ", " << QT << tempPortCategory << QT
                 << ", " << QT << DEFAULT_SIDL_PORT_NAMESPACE << portType << QT
                 << ", " << propertiesMap << i << ");" << NEWLINE;

    } else {
      std::string propertiesMap("pProps");
      fileStream << SP << TYPEMAP_POINTER << " " << propertiesMap << i << " = svc->createTypeMap();" << NEWLINE;
      fileStream << SP << "svc->addProvidesPort(" << DEFAULT_PORT_NAMESPACE
                 << portType << "::pointer(this)"
                 << ", " << QT << tempPortCategory << QT
                 << ", " << QT << DEFAULT_SIDL_PORT_NAMESPACE << portType << QT
                 << ", " << propertiesMap << i << ");" << NEWLINE;
    }
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
    fileStream << SP << TYPEMAP_POINTER << " " << propertiesMap << i << " = svc->createTypeMap();" << NEWLINE;

    fileStream << SP << "svc->registerUsesPort(" << QT << tempPortCategory << QT
               << ", " << QT << DEFAULT_SIDL_PORT_NAMESPACE << portType << QT
               << ", " << propertiesMap << i << ");" << NEWLINE;
  }

  fileStream << "}" << std::endl;
}

//go() and ui() functions - if thers is a GoPort or UIPort among Provides ports
void ComponentSkeletonWriter::writeGoAndUIFunctionsCode(std::ofstream& fileStream)
{

  for (unsigned int i = 0; i < providesPortsList.size(); i++) {
    string portname(providesPortsList[i]->GetType());
    string porttype(providesPortsList[i]->GetType());
    if (porttype.compare(string("UIPort")) == 0) {
      fileStream << NEWLINE <<"int " << providesPortsList[i]->GetClassName() << "::ui()"
                 << NEWLINE
                 << "{"
                 << NEWLINE << SP << "return 0;"
                 << NEWLINE
                 << "}" << NEWLINE;
    }

    if (porttype.compare(string("GoPort")) == 0) {
      if(isWithSidl)
        fileStream << NEWLINE <<"int " << compName << "::go()";
      else
        fileStream << NEWLINE <<"int " << providesPortsList[i]->GetClassName() << "::go()";
      fileStream << NEWLINE
                 << "{"
                 << NEWLINE << SP << "return 0;"
                 << NEWLINE
                 << "}" << NEWLINE;
    }
  }
  fileStream << std::endl;
}

void ComponentSkeletonWriter::ComponentMakefileCode(std::ofstream& fileStream)
{
  writeMakefileLicense(fileStream);
  fileStream << "include $(SCIRUN_SCRIPTS)/smallso_prologue.mk" << NEWLINE;
  fileStream << NEWLINE
             << "SRCDIR := CCA/Components/" << compName << NEWLINE;
  fileStream << NEWLINE
             << "SRCS += " << "$(SRCDIR)/" << compName << ".cc \\" << NEWLINE;
  if (isWithSidl) {
    fileStream << SP << SP << "$(SRCDIR)/" << compName << "_sidl.cc" << NEWLINE;
    fileStream <<  "$(SRCDIR)/" << compName << "_sidl.o: $(SRCDIR)/" << compName << "_sidl.cc $(SRCDIR)/"
               <<  compName << "_sidl.h" << NEWLINE;
    fileStream << NEWLINE << "$(SRCDIR)/" << compName << "_sidl.cc: $(SRCDIR)/" << compName
               << ".sidl $(SIDL_EXE)" << NEWLINE;
    fileStream << "\t" << "\t" << "$(SIDL_EXE) -I $(SRCTOP_ABS)/Core/CCA/spec/cca.sidl -o $@ $<" << NEWLINE;
    fileStream << NEWLINE <<  "$(SRCDIR)/" << compName << "_sidl.h: $(SRCDIR)/" << compName
               << ".sidl $(SIDL_EXE)" << NEWLINE;
    fileStream << "\t" << "\t" << "$(SIDL_EXE) -I $(SRCTOP_ABS)/Core/CCA/spec/cca.sidl -h -o $@ $<" << NEWLINE;
    fileStream << NEWLINE << "GENHDRS := $(SRCDIR)/" << compName << "_sidl.h" << NEWLINE;
  }
  fileStream << NEWLINE
             << "PSELIBS := Core/CCA/SSIDL Core/CCA/PIDL Core/CCA/Comm \\" << NEWLINE;
  fileStream << SP << "Core/CCA/spec Core/Thread Core/Containers Core/Exceptions" << NEWLINE;
  fileStream << NEWLINE
             << "ifeq ($(HAVE_GUI),yes)" << NEWLINE
             << SP << "LIBS := $(WX_LIBRARY)" << NEWLINE
             << "endif" << NEWLINE;
  fileStream << NEWLINE << "include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk" << NEWLINE;
  fileStream << NEWLINE << "$(SRCDIR)/" << compName << ".o: Core/CCA/spec/cca_sidl.h" << NEWLINE;
  fileStream << std::endl;
}

void ComponentSkeletonWriter::writeSidlFile(std::ofstream& fileStream)
{
  writeLicense(fileStream);
  fileStream << "package sci {" << NEWLINE;
  fileStream << SP << "package cca {" << NEWLINE;
  fileStream << SP << SP << "class " << compName << " implements-all " << DEFAULT_SIDL_NAMESPACE << "Component";
  for (unsigned int i = 0; i < providesPortsList.size(); i++) {
    string porttype(providesPortsList[i]->GetType());
    fileStream << "," << DEFAULT_SIDL_PORT_NAMESPACE << porttype;
  }
  fileStream << "{" << NEWLINE;
  fileStream << SP << SP << "}" << NEWLINE;
  fileStream << SP << "}" << NEWLINE;
  fileStream << "}" << std::endl;
}

}
