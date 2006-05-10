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

const std::string ComponentSkeletonWriter::DEFAULT_NAMESPACE("sci::cca::");
const std::string ComponentSkeletonWriter::DEFAULT_SIDL_NAMESPACE("sci.cca.");
const std::string ComponentSkeletonWriter::DEFAULT_PORT_NAMESPACE("sci::cca::ports::");
const std::string ComponentSkeletonWriter::DEFAULT_SIDL_PORT_NAMESPACE("sci.cca.ports.");

ComponentSkeletonWriter::ComponentSkeletonWriter(const std::string &cname, const std::vector<PortDescriptor*> pp, const std::vector<PortDescriptor*> up) : SERVICES_POINTER(DEFAULT_NAMESPACE + "Services::pointer"), TYPEMAP_POINTER(DEFAULT_NAMESPACE + "TypeMap::pointer"),
               compName(cname), providesPortsList(pp), usesPortsList(up)
{ }

void ComponentSkeletonWriter::GenerateCode()
{
   std::string src(sci_getenv("SCIRUN_SRCDIR"));
   std::string compsDir("/CCA/Components/");

  Dir d1 = Dir(src + compsDir + compName);
  if (!d1.exists()) {
    d1.create(src + "/CCA/Components/" + compName);
  }

  std::string sopf(src + compsDir + compName + DIR_SEP + compName + ".h");
  std::string sosf(src + compsDir + compName + DIR_SEP + compName + ".cc");
  std::string somf(src + compsDir + compName + DIR_SEP + "sub.mk");

  componentHeaderFile.open(sopf.c_str());
  componentSourceFile.open(sosf.c_str());
  componentMakeFile.open(somf.c_str());

  ComponentClassDefinitionCode();
  ComponentSourceFileCode();
  ComponentMakeFileCode();

  componentHeaderFile.close();
  componentSourceFile.close();
  componentMakeFile.close();
}

void ComponentSkeletonWriter::ComponentClassDefinitionCode()
{
  writeHeaderLicense();
  writeHeaderInit();
  writeComponentDefinitionCode();
  writePortClassDefinitionCode();
}

void ComponentSkeletonWriter::ComponentSourceFileCode()
{
 
  writeSourceLicense();
  writeSourceInit();
  writeLibraryHandle();
  writeSourceClassImpl();
}


//////////////////////////////////////////////////////////////////////////
// private member functions

void ComponentSkeletonWriter::writeHeaderLicense()
{
  
  componentHeaderFile << "/*"<<std::endl<<SP<<"For more information, please see: http://software.sci.utah.edu"<<std::endl<<std::endl<<std::endl<<SP<<"The MIT License"<<std::endl<<std::endl<<SP<<"Copyright (c) 2004 Scientific Computing and Imaging Institute,"<<std::endl<<SP<<"University of Utah."<<std::endl<<SP<<"License for the specific language governing rights and limitations under"<<std::endl<<SP<<"Permission is hereby granted, free of charge, to any person obtaining a"<<std::endl<<SP<<"copy of this software and associated documentation files (the \"Software\"),"<<std::endl<<SP<<"to deal in the Software without restriction, including without limitation"<<std::endl<<SP<<"the rights to use, copy, modify, merge, publish, distribute, sublicense,"<<std::endl<<SP<<"and/or sell copies of the Software, and to permit persons to whom the"<<std::endl<<SP<<"Software is furnished to do so, subject to the following conditions:"<<std::endl<<std::endl<<SP<<"The above copyright notice and this permission notice shall be included"<<std::endl<<SP<<"in all copies or substantial portions of the Software."<<std::endl<<std::endl<<SP<<"THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS"<<std::endl<<SP<<"OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,"<<std::endl<<SP<<"FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL"<<std::endl<<SP<<"THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER"<<std::endl<<SP<<"LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING"<<std::endl<<SP<<"FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER"<<std::endl<<SP<<"DEALINGS IN THE SOFTWARE."<<std::endl<<"*/";
 
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
    componentHeaderFile << "class " << (providesPortsList[i])->GetName() << " : public "
			<< DEFAULT_PORT_NAMESPACE << (providesPortsList[i])->GetType() << " {"
			<< std::endl;

    //public  members
    componentHeaderFile << "public:" << std::endl;
    componentHeaderFile << SP << "virtual ~"<< (providesPortsList[i])->GetName() << "() {}" << std::endl;
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

void ComponentSkeletonWriter::writeSourceLicense()
{

  
  componentSourceFile << "/*"<<std::endl<<SP<<"For more information, please see: http://software.sci.utah.edu"<<std::endl<<std::endl<<std::endl<<SP<<"The MIT License"<<std::endl<<std::endl<<SP<<"Copyright (c) 2004 Scientific Computing and Imaging Institute,"<<std::endl<<SP<<"University of Utah."<<std::endl<<SP<<"License for the specific language governing rights and limitations under"<<std::endl<<SP<<"Permission is hereby granted, free of charge, to any person obtaining a"<<std::endl<<SP<<"copy of this software and associated documentation files (the \"Software\"),"<<std::endl<<SP<<"to deal in the Software without restriction, including without limitation"<<std::endl<<SP<<"the rights to use, copy, modify, merge, publish, distribute, sublicense,"<<std::endl<<SP<<"and/or sell copies of the Software, and to permit persons to whom the"<<std::endl<<SP<<"Software is furnished to do so, subject to the following conditions:"<<std::endl<<std::endl<<SP<<"The above copyright notice and this permission notice shall be included"<<std::endl<<SP<<"in all copies or substantial portions of the Software."<<std::endl<<std::endl<<SP<<"THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS"<<std::endl<<SP<<"OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,"<<std::endl<<SP<<"FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL"<<std::endl<<SP<<"THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER"<<std::endl<<SP<<"LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING"<<std::endl<<SP<<"FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER"<<std::endl<<SP<<"DEALINGS IN THE SOFTWARE."<<std::endl<<"*/";
  
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
   writeGoAndUiFunctionsCode();
}


void ComponentSkeletonWriter::writeConstructorandDestructorCode()
{
  componentSourceFile << std::endl << compName << "::" <<compName << "()" << std::endl
                      << "{" << std::endl
                      << "}" << std::endl;
  componentSourceFile << std::endl << compName << "::~" <<compName << "()" << std::endl
                      << "{" << std::endl;

  //Destructor code
  for (unsigned int i = 0; i < providesPortsList.size(); i++) {
    componentSourceFile << SP << "services->removeProvidesPort("
                        << QT << providesPortsList[i]->GetName() <<  QT << ");" << std::endl;
  }

  for (unsigned int i = 0; i < usesPortsList.size(); i++) {
    componentSourceFile << SP << "services->unregisterUsesPort("
                        << QT << usesPortsList[i]->GetName() << QT << ");" << std::endl;
  }
  componentSourceFile << "}" << std::endl;
}


void ComponentSkeletonWriter::writeSetServicesCode()
{
  componentSourceFile << std::endl
                      << "void " << compName << "::setServices(const " << SERVICES_POINTER << "& svc)"<< std::endl;
  componentSourceFile << "{" << std::endl;
  componentSourceFile << SP << "services = svc;" << std::endl;
  //componentSourceFile<<std::endl<<SP<<"svc->registerForRelease(sci::cca::ComponentRelease::pointer(this)); ";

  for (unsigned int i = 0; i < providesPortsList.size(); i++) {
    std::string portName(providesPortsList[i]->GetName());
    std::string portType(providesPortsList[i]->GetType());
    std::string tempPortInstance, tempPortPtr, tempPortCategory;

    for (unsigned int j = 0; j < portType.length(); j++) {
      char tmp = portType.at(j);
      
      // Unicode safe?
      if ((tmp >= 'A') && (tmp <= 'Z')) {
	tempPortInstance.append(1, (char) tolower(tmp));
      }
    }

    tempPortInstance = "provides" + tempPortInstance;
    // char tmp=portType.at(0);
    // tempPortCategory=portType.substr(1,portType.length());

    //tempPortCategory="provides"+tempPortCategory.insert(0,1,(char)tolower(tmp));
    tempPortCategory = (std::string) providesPortsList[i]->GetDesc();
    //tempPortPtr=tempPortCategory+"Ptr";
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
    std::string portName(usesPortsList[i]->GetName());
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

void ComponentSkeletonWriter::writeGoAndUiFunctionsCode()
{

 for (unsigned int i = 0; i < providesPortsList.size(); i++) {
   string portname(providesPortsList[i]->GetType());
   string porttype(providesPortsList[i]->GetType());

#if DEBUG
   std::cout << "\nhere in ckw " << porttype.c_str() << "\n";
#endif
   // if(strcmp(porttype.c_str(),"UIPort")==0);
   if (porttype.compare(string("UIPort")) == 0) {
     componentSourceFile << std::endl <<"int " << providesPortsList[i]->GetName() << "::ui()"
                         << std::endl
                         << "{"
                         << std::endl<<SP<<"return 0;"
                         << std::endl
                         << "}" << std::endl;
   }
   if (porttype.compare(string("GoPort")) == 0) {
     componentSourceFile << std::endl <<"int " << providesPortsList[i]->GetName() << "::go()"
                         << std::endl
                         << "{"
                         << std::endl<<SP<<"return 0;"
                         << std::endl
                         << "}" << std::endl;
   }
 }
 componentSourceFile << std::endl;
}

void ComponentSkeletonWriter::ComponentMakeFileCode()
{
  componentMakeFile << "#Sample Make File ";
  componentMakeFile << std::endl << std::endl;
  componentMakeFile << "include $(SCIRUN_SCRIPTS)/smallso_prologue.mk" << std::endl;
  componentMakeFile << std::endl << "SRCDIR := CCA/Components/" << compName << std::endl;
  componentMakeFile << std::endl << "SRCS += " << "$(SRCDIR)/" << compName << ".cc \\" << std::endl;
  componentMakeFile << std::endl << "PSELIBS := Core/CCA/SSIDL Core/CCA/PIDL Core/CCA/Comm \\" << std::endl;
  componentMakeFile << SP << "Core/CCA/spec Core/Thread Core/Containers Core/Exceptions" << std::endl;
  componentMakeFile << std::endl << "CFLAGS += $(WX_CXXFLAGS)" << std::endl
		    << "CXXFLAGS += $(WX_CXXFLAGS)" << std::endl
                    << "LIBS := $(WX_LIBRARY)" << std::endl;
  componentMakeFile << std::endl  << "include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk" << std::endl;
  componentMakeFile << std::endl << "$(SRCDIR)/" << compName << ".o: Core/CCA/spec/cca_sidl.h" << std::endl;
  componentMakeFile << std::endl;
}

}
