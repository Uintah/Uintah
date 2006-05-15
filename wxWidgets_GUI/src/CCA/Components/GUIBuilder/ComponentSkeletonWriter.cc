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

#include<iostream>
#include<fstream>


#include <CCA/Components/GUIBuilder/ComponentSkeletonWriter.h>
#include <Core/Containers/StringUtil.h>
#include <Core/OS/dirent.h>
#include <Core/OS/Dir.h>
#include <Core/Util/Environment.h>

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

ComponentSkeletonWriter::ComponentSkeletonWriter(const std::string &cname, const std::vector<PortDescriptor*> pp, const std::vector<PortDescriptor*> up)
  : SERVICES_POINTER(DEFAULT_NAMESPACE + "Services::pointer"), TYPEMAP_POINTER(DEFAULT_NAMESPACE + "TypeMap::pointer"),
    compName(cname), providesPortsList(pp), usesPortsList(up)
{ }

void ComponentSkeletonWriter::GenerateCode()
{
   std::string srcDir(sci_getenv("SCIRUN_SRCDIR"));
   std::string compsDir("/CCA/Components/");

  Dir d1 = Dir(srcDir + compsDir + compName);
  if (!d1.exists()) {
    d1.create(srcDir + "/CCA/Components/" + compName);
  }

  // Header file
  std::string sopf(srcDir + compsDir + compName + DIR_SEP + compName + ".h");
  componentHeaderFile.open(sopf.c_str());
  ComponentClassDefinitionCode();
  componentHeaderFile.close();

  // Implementation file
  std::string sosf(srcDir + compsDir + compName + DIR_SEP + compName + ".cc");
  componentSourceFile.open(sosf.c_str());
  ComponentSourceFileCode();
  componentSourceFile.close();

  // Makefile fragment
  std::string somf(srcDir + compsDir + compName + DIR_SEP + "sub.mk");
  componentMakefile.open(somf.c_str());
  ComponentMakefileCode();
  componentMakefile.close();
}

void ComponentSkeletonWriter::ComponentClassDefinitionCode()
{
  writeLicense(componentHeaderFile);
  writeHeaderInit();
  writeComponentDefinitionCode();
  writePortClassDefinitionCode();
}

void ComponentSkeletonWriter::ComponentSourceFileCode()
{
  writeLicense(componentSourceFile);
  writeSourceInit();
  writeLibraryHandle();
  writeSourceClassImpl();
}


//////////////////////////////////////////////////////////////////////////
// private member functions

// Add license to file; the license text is identical for both the header
// and implementation files, so the file stream for each file is the
// function arguement.
// TODO: The license text should be read from file.
void ComponentSkeletonWriter::writeLicense(std::ofstream& fileStream)
{
  fileStream << OPEN_C_COMMENT << std::endl
             << SP << "For more information, please see: http://software.sci.utah.edu" << std::endl
             << std::endl
             << SP << "The MIT License" << std::endl
             << std::endl
             << SP << "Copyright (c) 2004 Scientific Computing and Imaging Institute," << std::endl
             << SP << "University of Utah." << std::endl
             << SP << "License for the specific language governing rights and limitations under" << std::endl
             << SP << "Permission is hereby granted, free of charge, to any person obtaining a" << std::endl
             << SP << "copy of this software and associated documentation files (the \"Software\")," << std::endl
             << SP << "to deal in the Software without restriction, including without limitation" << std::endl
             << SP << "the rights to use, copy, modify, merge, publish, distribute, sublicense," << std::endl
             << SP << "and/or sell copies of the Software, and to permit persons to whom the" << std::endl
             << SP << "Software is furnished to do so, subject to the following conditions:" << std::endl
             << std::endl
             << SP << "The above copyright notice and this permission notice shall be included" << std::endl
             << SP << "in all copies or substantial portions of the Software." << std::endl
             << std::endl
             << SP << "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS" << std::endl
             << SP << "OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY," << std::endl
             << SP << "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL" << std::endl
             << SP << "THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER" << std::endl
             << SP << "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING" << std::endl
             << SP << "FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER" << std::endl
             << SP << "DEALINGS IN THE SOFTWARE." << std::endl
             << CLOSE_C_COMMENT << std::endl;
}

void ComponentSkeletonWriter::writeMakefileLicense(std::ofstream& fileStream)
{
  fileStream << UNIX_SHELL_COMMENT << SP << "For more information, please see: http://software.sci.utah.edu" << std::endl
             << UNIX_SHELL_COMMENT << std::endl
             << UNIX_SHELL_COMMENT << SP << "The MIT License" << std::endl
             << UNIX_SHELL_COMMENT << std::endl
             << UNIX_SHELL_COMMENT << SP << "Copyright (c) 2004 Scientific Computing and Imaging Institute," << std::endl
             << UNIX_SHELL_COMMENT << SP << "University of Utah." << std::endl
             << UNIX_SHELL_COMMENT << SP << "License for the specific language governing rights and limitations under" << std::endl
             << UNIX_SHELL_COMMENT << SP << "Permission is hereby granted, free of charge, to any person obtaining a" << std::endl
             << UNIX_SHELL_COMMENT << SP << "copy of this software and associated documentation files (the \"Software\")," << std::endl
             << UNIX_SHELL_COMMENT << SP << "to deal in the Software without restriction, including without limitation" << std::endl
             << UNIX_SHELL_COMMENT << SP << "the rights to use, copy, modify, merge, publish, distribute, sublicense," << std::endl
             << UNIX_SHELL_COMMENT << SP << "and/or sell copies of the Software, and to permit persons to whom the" << std::endl
             << UNIX_SHELL_COMMENT << SP << "Software is furnished to do so, subject to the following conditions:" << std::endl
             << UNIX_SHELL_COMMENT << std::endl
             << UNIX_SHELL_COMMENT << SP << "The above copyright notice and this permission notice shall be included" << std::endl
             << UNIX_SHELL_COMMENT << SP << "in all copies or substantial portions of the Software." << std::endl
             << UNIX_SHELL_COMMENT << std::endl
             << UNIX_SHELL_COMMENT << SP << "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS" << std::endl
             << UNIX_SHELL_COMMENT << SP << "OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY," << std::endl
             << UNIX_SHELL_COMMENT << SP << "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL" << std::endl
             << UNIX_SHELL_COMMENT << SP << "THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER" << std::endl
             << UNIX_SHELL_COMMENT << SP << "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING" << std::endl
             << UNIX_SHELL_COMMENT << SP << "FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER" << std::endl
             << UNIX_SHELL_COMMENT << SP << "DEALINGS IN THE SOFTWARE." << std::endl
             << std::endl;
}

void ComponentSkeletonWriter::writeHeaderInit()
{
  componentHeaderFile << std::endl;
  componentHeaderFile << std::endl;
  componentHeaderFile << "#ifndef SCIRun_Framework_" << compName << "_h" << std::endl;
  componentHeaderFile << "#define SCIRun_Framework_" << compName << "_h" << std::endl;
  componentHeaderFile << std::endl << "#include <Core/CCA/spec/cca_sidl.h>" << std::endl;

  componentHeaderFile << std::endl;
  componentHeaderFile << "using namespace SCIRun;" << std::endl;
}

void ComponentSkeletonWriter::writeComponentDefinitionCode()
{
  componentHeaderFile << std::endl;
  componentHeaderFile << "class " << compName << ": public " << DEFAULT_NAMESPACE << "Component {"
          << std::endl;
  // public members
  componentHeaderFile << "public:" << std::endl;
  componentHeaderFile << SP << compName << "();" << std::endl;
  componentHeaderFile << SP << "virtual ~"<< compName << "();" << std::endl;
  componentHeaderFile << SP << "virtual void setServices(const " << SERVICES_POINTER << "& svc);" << std::endl;
  // private members
  componentHeaderFile << std::endl;
  componentHeaderFile << "private:" << std::endl;
  componentHeaderFile << SP << compName << "(const " << compName << "&);" << std::endl;
  componentHeaderFile << SP << compName << "& operator=(const " << compName << "&);" << std::endl;
  // services handle
  componentHeaderFile << SP << SERVICES_POINTER << " services;" << std::endl;
  componentHeaderFile << "};" << std::endl;
  componentHeaderFile << std::endl;
}

void ComponentSkeletonWriter::writePortClassDefinitionCode()
{
  for (unsigned int i = 0; i < providesPortsList.size(); i++) {

    componentHeaderFile << std::endl;
    componentHeaderFile << "class " << (providesPortsList[i])->GetClassName() << " : public "
                        << DEFAULT_PORT_NAMESPACE << (providesPortsList[i])->GetType() << " {"
                        << std::endl;

    //public  members
    componentHeaderFile << "public:" << std::endl;
    componentHeaderFile << SP << "virtual ~"<< (providesPortsList[i])->GetClassName() << "() {}" << std::endl;
    componentHeaderFile << SP << "void setParent(" << compName << " *com)" << " { this->com = com; }";
    if ((providesPortsList[i])->GetType() == "GoPort") {
      componentHeaderFile <<std::endl<< SP << "virtual int go();";
    }
    if ((providesPortsList[i])->GetType() == "UIPort") {
      componentHeaderFile <<std::endl << SP << "virtual int ui();";
    }
    // private  members
    componentHeaderFile << std::endl;
    componentHeaderFile << std::endl << "private:" << std::endl;
    componentHeaderFile << SP << compName << " *com;" << std::endl;
    componentHeaderFile << std::endl << "};" << std::endl;
  }
  componentHeaderFile << std::endl;
  componentHeaderFile << "#endif" << std::endl;
}


void ComponentSkeletonWriter::writeSourceInit()
{
  componentSourceFile << std::endl << std::endl;

  //Header files
  componentSourceFile << "#include<CCA/Components" << DIR_SEP << compName << DIR_SEP << compName << ".h>" << std::endl;
  componentSourceFile << "#include <SCIRun" << DIR_SEP << "TypeMap.h>" << std::endl;
  componentSourceFile << std::endl;
  componentSourceFile << "using namespace SCIRun;" << std::endl;
}

void ComponentSkeletonWriter::writeLibraryHandle()
{
  componentSourceFile << std::endl;
  componentSourceFile << "extern " << QT<< "C" << QT << " " << DEFAULT_NAMESPACE << "Component::pointer make_SCIRun_"
                      << compName << "()" << std::endl;
  componentSourceFile << "{" << std::endl;
  componentSourceFile << SP << "return " << DEFAULT_NAMESPACE << "Component::pointer(new " << compName << "());" << std::endl;
  componentSourceFile << "}" << std::endl;
  componentSourceFile << std::endl;
}


void ComponentSkeletonWriter::writeSourceClassImpl()
{
  writeConstructorandDestructorCode();
  writeSetServicesCode();
  writeGoAndUIFunctionsCode();
}


void ComponentSkeletonWriter::writeConstructorandDestructorCode()
{
  //Constructor code
  componentSourceFile << std::endl << compName << "::" << compName << "()" << std::endl
                      << "{" << std::endl
                      << "}" << std::endl;
  //Destructor code
  componentSourceFile << std::endl << compName << "::~" << compName << "()" << std::endl
                      << "{" << std::endl;
  for (unsigned int i = 0; i < providesPortsList.size(); i++) {
    componentSourceFile << SP << "services->removeProvidesPort("
                        << QT << providesPortsList[i]->GetUniqueName() <<  QT << ");" << std::endl;
  }

  for (unsigned int i = 0; i < usesPortsList.size(); i++) {
    componentSourceFile << SP << "services->unregisterUsesPort("
                        << QT << usesPortsList[i]->GetUniqueName() << QT << ");" << std::endl;
  }
  componentSourceFile << "}" << std::endl;
}


void ComponentSkeletonWriter::writeSetServicesCode()
{
  componentSourceFile << std::endl
                      << "void " << compName << "::setServices(const " << SERVICES_POINTER << "& svc)"<< std::endl;
  componentSourceFile << "{" << std::endl;
  componentSourceFile << SP << "services = svc;" << std::endl;
  //componentSourceFile << std::endl << SP << "svc->registerForRelease(sci::cca::ComponentRelease::pointer(this)); ";

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

    componentSourceFile << SP << portName << " *" << tempPortInstance
                        << " = new " << portName << "();" << std::endl;
    componentSourceFile << SP << tempPortInstance << "->setParent(this);" << std::endl;

    std::string propertiesMap("pProps");
    componentSourceFile << SP << TYPEMAP_POINTER << " " << propertiesMap << i << " = svc->createTypeMap();" << std::endl;

    componentSourceFile << SP << "svc->addProvidesPort(" << tempPortPtr
                        << ", " << QT << tempPortCategory << QT
                        << ", " << QT << DEFAULT_SIDL_PORT_NAMESPACE << portType << QT
                        << ", " << propertiesMap << i << ");" << std::endl;

    componentSourceFile << std::endl;
  }

  for (unsigned int i = 0; i < usesPortsList.size(); i++) {
    std::string portName(usesPortsList[i]->GetClassName());
    std::string portType(usesPortsList[i]->GetType());
    std::string tempPortCategory;

    char tmp = portType.at(0);
    tempPortCategory = portType.substr(1, portType.length());
    tempPortCategory = "uses" + tempPortCategory.insert(0, 1, (char) tolower(tmp));

    std::string propertiesMap("uProps");
    componentSourceFile << SP << TYPEMAP_POINTER << " " << propertiesMap << i << " = svc->createTypeMap();" << std::endl;

    componentSourceFile << SP << "svc->registerUsesPort(" << QT << tempPortCategory << QT
                        << ", " << QT << DEFAULT_SIDL_PORT_NAMESPACE << portType << QT
                        << ", " << propertiesMap << i << ");" << std::endl;
  }

  componentSourceFile << "}" << std::endl;
}

//go() and ui() functions - if thers is a GoPort or UIPort among Provides ports
void ComponentSkeletonWriter::writeGoAndUIFunctionsCode()
{

  for (unsigned int i = 0; i < providesPortsList.size(); i++) {
    string portname(providesPortsList[i]->GetType());
    string porttype(providesPortsList[i]->GetType());

#if DEBUG
    std::cout << "\nhere in ckw " << porttype.c_str() << "\n";
#endif
    if (porttype.compare(string("UIPort")) == 0) {
      componentSourceFile << std::endl <<"int " << providesPortsList[i]->GetClassName() << "::ui()"
                          << std::endl
                          << "{"
                          << std::endl << SP << "return 0;"
                          << std::endl
                          << "}" << std::endl;
    }
    if (porttype.compare(string("GoPort")) == 0) {
      componentSourceFile << std::endl <<"int " << providesPortsList[i]->GetClassName() << "::go()"
                          << std::endl
                          << "{"
                          << std::endl << SP << "return 0;"
                          << std::endl
                          << "}" << std::endl;
    }
  }
  componentSourceFile << std::endl;
}

void ComponentSkeletonWriter::ComponentMakefileCode()
{
  writeMakefileLicense(componentMakefile);
  componentMakefile << "include $(SCIRUN_SCRIPTS)/smallso_prologue.mk" << std::endl;
  componentMakefile << std::endl
                    << "SRCDIR := CCA/Components/" << compName << std::endl;
  componentMakefile << std::endl
                    << "SRCS += " << "$(SRCDIR)/" << compName << ".cc \\" << std::endl;
  componentMakefile << std::endl
                    << "PSELIBS := Core/CCA/SSIDL Core/CCA/PIDL Core/CCA/Comm \\" << std::endl;
  componentMakefile << SP << "Core/CCA/spec Core/Thread Core/Containers Core/Exceptions" << std::endl;
  componentMakefile << std::endl
                    << "CFLAGS += $(WX_CXXFLAGS)" << std::endl
                    << "CXXFLAGS += $(WX_CXXFLAGS)" << std::endl
                    << "LIBS := $(WX_LIBRARY)" << std::endl;
  componentMakefile << std::endl << "include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk" << std::endl;
  componentMakefile << std::endl << "$(SRCDIR)/" << compName << ".o: Core/CCA/spec/cca_sidl.h" << std::endl;
  componentMakefile << std::endl;
}

}
