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

#include <Core/Containers/StringUtil.h>
#include <CCA/Components/Builder/ComponentSkeletonWriter.h>
#include <Core/OS/dirent.h>
#include <Core/OS/Dir.h>

namespace GUIBuilder {

using namespace SCIRun;

const std::string ComponentSkeletonWriter::SP("  ");
const std::string ComponentSkeletonWriter::QT("\"");
const std::string ComponentSkeletonWriter::DIR_SEP("/");

const std::string ComponentSkeletonWriter::DEFAULT_NAMESPACE("sci::cca::");
const std::string ComponentSkeletonWriter::DEFAULT_SIDL_NAMESPACE("sci.cca.");
const std::string ComponentSkeletonWriter::DEFAULT_PORT_NAMESPACE("sci::cca::ports::");
const std::string ComponentSkeletonWriter::DEFAULT_SIDL_PORT_NAMESPACE("sci.cca.ports");

ComponentSkeletonWriter::ComponentSkeletonWriter(const std::string &cname, const std::vector<PortDescriptor*> pp, const std::vector<PortDescriptor*> up) : SERVICES_POINTER(DEFAULT_NAMESPACE + "Services::pointer"), compName(cname), providesPortsList(pp), usesPortsList(up)
{ }

void ComponentSkeletonWriter::GenerateCode()
{

  std::cout<<"\nhere in ckw\n";
  Dir d;
  d.create("../src/CCA/Components/" + compName);
  std::string sopf("../src/CCA/Components/" + compName + "/" + compName + ".h");
  std::string sosf("../src/CCA/Components/" + compName + "/" + compName + ".cc");

  componentHeaderFile.open(sopf.c_str());
  componentSourceFile.open(sosf.c_str());


  ComponentClassDefinitionCode();
  ComponentSourceFileCode();

  componentHeaderFile.close();
  componentSourceFile.close();
}

void ComponentSkeletonWriter::ComponentClassDefinitionCode()
{
  writeHeaderInit();
  writeComponentDefinitionCode();
  writePortClassDefinitionCode();
}

void ComponentSkeletonWriter::ComponentSourceFileCode()
{
  writeSourceInit();
  writeLibraryHandle();
  writeSourceClassImpl();
}


//////////////////////////////////////////////////////////////////////////
// private member functions

void ComponentSkeletonWriter::writeHeaderInit()
{
  // license here
  componentHeaderFile << "//Sample header File" << std::endl;
  componentHeaderFile << std::endl;
  componentHeaderFile << std::endl;
  componentHeaderFile << "#ifndef SCIRun_Framework_" << compName << "_h" << std::endl;
  componentHeaderFile << "#define SCIRun_Framework_" << compName << "_h" << std::endl;
  componentHeaderFile << std::endl << "#include <Core/CCA/spec/cca_sidl.h>" << std::endl;
  //componentHeaderFile <<" \n#include <CCA/Components/"<<compName<<"/"<<compName<<"_sidl.h>
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
    // componentHeaderFile<<"\nclass "<<providesPortsList[i].name<<" : public sci::cca::ports::"
      //<<providesPortsList[i].type<<" {\n";

    componentHeaderFile << std::endl;
    componentHeaderFile << "class " << (providesPortsList[i])->GetName() << " : public "
                        << DEFAULT_PORT_NAMESPACE << (providesPortsList[i])->GetType() << " {"
                        << std::endl;

    // componentHeaderFile<<"public:\n"<<sp<<"virtual ~"<<providesPortsList[i].name<<"(){}";

    // public members
    componentHeaderFile << "public:" << std::endl;
    componentHeaderFile << SP << "virtual ~"<< (providesPortsList[i])->GetName() << "(){}" << std::endl;
    componentHeaderFile << SP << "void setParent(" << compName << " *com)" << "{ this->com = com; }" << std::endl;
    componentHeaderFile << SP << compName << " *com;" << std::endl;
    componentHeaderFile << "};" << std::endl;
  }
  componentHeaderFile << std::endl;
  componentHeaderFile << "#endif" << std::endl;
}

void ComponentSkeletonWriter::writeSourceInit()
{

  componentSourceFile << "//Sample Source code" << std::endl;
  componentSourceFile << std::endl;

  //Header files
  componentSourceFile << "#include<CCA/Components" << DIR_SEP << compName << DIR_SEP << compName << ".h>" << std::endl;
  componentSourceFile << "#include <SCIRun/TypeMap.h>" << std::endl;
  componentSourceFile << std::endl;
  componentSourceFile << "using namespace SCIRun;" << std::endl;

  // these headers are probably not necessary for the class skeleton
  //componentSourceFile << "#include <Core/Thread/Time.h>" << std::endl;
  //componentSourceFile << "#include <iostream>" << std::endl;
  //componentSourceFile << "#include <unistd.h>" << std::endl;

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

  componentSourceFile << std::endl;
  componentSourceFile << compName << "::" << compName << "()" << std::endl;
  componentSourceFile << "{" << std::endl;
  componentSourceFile << "}" << std::endl;
  //Destructor code
  componentSourceFile << std::endl;
  componentSourceFile << compName << "::~" << compName << "()" << std::endl;
  componentSourceFile << "{" << std::endl;

  // Remove provides ports
  for (int i = 0; i < providesPortsList.size(); i++) {
    componentSourceFile << SP << "services->removeProvidesPort("
                        << QT << providesPortsList[i]->GetName() << QT << ");" << std::endl;
  }

  componentSourceFile << std::endl;
  // Unregister uses ports
  for (int i = 0; i < usesPortsList.size(); i++) {
    componentSourceFile << SP << "services->unregisterUsesPort("
                        << QT << usesPortsList[i]->GetName() << QT << ");" << std::endl;
  }
  componentSourceFile << "}" << std::endl;

  // Set services code
  componentSourceFile << std::endl;
  componentSourceFile << "void " << compName << "::setServices(const " << SERVICES_POINTER << "& svc)" << std::endl;
  componentSourceFile << "{" << std::endl;
  componentSourceFile << SP << "services = svc;" << std::endl;
  //componentSourceFile << "\n"<<SP<<"svc->registerForRelease(sci::cca::ComponentRelease::pointer(this)); ";
  componentSourceFile << SP << DEFAULT_NAMESPACE << "TypeMap::pointer props = svc->createTypeMap();";

  // Add provides ports
  for (int i = 0; i < providesPortsList.size(); i++) {
      std::string portName(providesPortsList[i]->GetName());
      std::string portType(providesPortsList[i]->GetType());
      std::string tempPortInstance, tempPortPtr, tempPortCategory;

      if(strcmp(portType.c_str(),"UIPort")==0)
	{
	  tempPortInstance=std::string("uip");
	  tempPortPtr=std::string("uiPortPtr");
	  tempPortCategory=std::string("ui");
	}
      else if(strcmp(portType.c_str(),"GoPort")==0)
	{
	  tempPortInstance=std::string("gop");
	  tempPortPtr=std::string("goPortPtr");
	  tempPortCategory=std::string("go");
	}
      else if(strcmp(portType.c_str(),"StringPort")==0)
	{
	  tempPortInstance=std::string("sp");
	  tempPortPtr=std::string("stringPortPtr");
	  tempPortCategory=std::string("string");
	}

	  componentSourceFile<<"\n"<<SP<<portType<<" *"<<tempPortInstance<<" = new "
			 <<portName<<"();";
	  componentSourceFile<<"\n"<<SP<<tempPortInstance<<"->setParent(this);";
	  componentSourceFile<<"\n"<<SP<<portName<<"::pointer "<<tempPortPtr<<" = "
			     <<portName<<"::pointer("<<tempPortInstance<<");";
	  componentSourceFile<<"\n"<<SP<<"svc->addProvidesPort("<<tempPortPtr<<", \""<<tempPortCategory
			     <<"\",\"sci.cca.ports."<<portType<<"\",sci::cca::TypeMap::pointer(0))";



	  /*if(strcmp(portType.c_str(),"UIPort")==0)
	{
	  componentSourceFile<<"\n"<<sp<<providesPortsList[i]->GetType()<<" *uip = new "
			 <<portName<<"();";
	  componentSourceFile<<"\n"<<sp<<"uip->setParent(this);";
	  componentSourceFile<<"\n"<<sp<<portName<<"::pointer uiPortPtr = "<<portName<<"::pointer(uip);";
	  componentSourceFile<<"\n"<<sp
		      <<"svc->addProvidesPort(uiPortPtr, \"ui\",\"sci.cca.ports.UIPort\",sci::cca::TypeMap::pointer(0))";

	}
      if(strcmp(portType.c_str(),"GoPort")==0)
	{
	  componentSourceFile<<"\n"<<sp<<providesPortsList[i]->GetType()<<" *gop = new "
			 <<portName<<"();";
	  componentSourceFile<<"\n"<<sp<<"gop->setParent(this);";
	  componentSourceFile<<"\n"<<sp<<portName<<"::pointer GoPortPtr = "<<portName<<"::pointer(gop);";
	  componentSourceFile<<"\n"<<sp
		      <<"svc->addProvidesPort(goPortPtr, \"go\",\"sci.cca.ports.GoPort\",sci::cca::TypeMap::pointer(0));";
		      }*/

      componentSourceFile<<"\n";
    }
  for(int i=0;i<usesPortsList.size();i++)
    {
      std::string portName= std::string(usesPortsList[i]->GetName());
      std::string portType=std::string(usesPortsList[i]->GetType());
      std::string tempPortCategory;
      if(strcmp(portType.c_str(),"UIPort")==0)
	{

	  tempPortCategory=std::string("ui");
	}
      else if(strcmp(portType.c_str(),"GoPort")==0)
	{

	  tempPortCategory=std::string("go");
	}
      else if(strcmp(portType.c_str(),"StringPort")==0)
	{

	  tempPortCategory=std::string("string");
	}


	  componentSourceFile<<"\n"<<SP<<"svc->registerUsesPort(\""<<tempPortCategory;
	  componentSourceFile<<"\",\"sci.cca.ports."<<portType<<"\",props);";
    }


  componentSourceFile << std::endl;
  componentSourceFile << "}" << std::endl;

  //go() and ui() functions

  for(int i=0;i<providesPortsList.size();i++)
    {
      std::string portname=std::string(providesPortsList[i]->GetType());
      std::string porttype=std::string(providesPortsList[i]->GetType());
      if(strcmp(porttype.c_str(),"UIPort")==0)
	{
	  componentSourceFile<<"\n"<<providesPortsList[i]->GetName()<<"::ui()\n{\n}\n";
	}
      if(strcmp(porttype.c_str(),"GoPort")==0)
	{
	  componentSourceFile<<"\n"<<providesPortsList[i]->GetName()<<"::go()\n{\n}\n";
	}
    }
  componentSourceFile<<std::endl;

}


}
