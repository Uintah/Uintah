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


#include <Core/Containers/StringUtil.h>
#include <CCA/Components/Builder/ComponentSkeletonWriter.h>
#include <Core/OS/dirent.h>
#include <Core/OS/Dir.h>
#include <Core/Util/Environment.h>

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
   std::string src(sci_getenv("SCIRUN_SRCDIR"));
      
  Dir d1=Dir(src+"/CCA/Components/" + compName);
  if(!d1.exists())
    d1.create(src+"/CCA/Components/" + compName);
  
  std::string sopf(src+"/CCA/Components/" + compName + "/" + compName + ".h");
  std::string sosf(src+"/CCA/Components/" + compName + "/" + compName + ".cc");
  std::string somf(src+"/CCA/Components/" + compName + "/sub.mk");

  
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

   
  //componentHeaderFile << "//Sample header File" << std::endl;
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

   
      

    // public members
    componentHeaderFile << "public:" << std::endl;
    componentHeaderFile << SP << "virtual ~"<< (providesPortsList[i])->GetName() << "(){;}" << std::endl;
    componentHeaderFile << SP << "void setParent(" << compName << " *com)" << "{ this->com = com; }" << std::endl;
    componentHeaderFile << SP << compName << " *com;" << std::endl;
    
    if((providesPortsList[i])->GetType()=="GoPort")
        componentHeaderFile << "\n"<<SP<<"int go();";
    if((providesPortsList[i])->GetType()=="UIPort")
        componentHeaderFile << "\n"<<SP<<"int ui();";
    
    componentHeaderFile << "\n};" << std::endl;
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
  componentSourceFile<<"\n"<<compName<<"::"<<compName<<"()"<<"\n{\n}";
  componentSourceFile<<"\n"<<compName<<"::~"<<compName<<"()\n{";
  
  //Destructor code

  
  for(unsigned int i=0;i<providesPortsList.size();i++)
    {
        
	componentSourceFile << "\n"<<SP<<"services->removeProvidesPort(\""<<providesPortsList[i]->GetName()<<"\");";
    }

  for(unsigned int i=0;i<usesPortsList.size();i++)
    {
      componentSourceFile << "\n"<<SP<<"services->unregisterUsesPort(\""<<usesPortsList[i]->GetName()<<"\");";
    }

}
void ComponentSkeletonWriter::writeSetServicesCode()
{
  componentSourceFile<<"\n}\nvoid "<<compName<<"::setServices(const sci::cca::Services::pointer& svc)\n\{\n";
  componentSourceFile<<SP<<"services = svc;";
  //componentSourceFile<<"\n"<<SP<<"svc->registerForRelease(sci::cca::ComponentRelease::pointer(this)); ";
  componentSourceFile<<"\n"<<SP<<"sci::cca::TypeMap::pointer props = svc->createTypeMap(); \n\n";
 
  for(unsigned int i=0;i<providesPortsList.size();i++)
    {
      
      string portName= string(providesPortsList[i]->GetName());
      string portType=string(providesPortsList[i]->GetType());
      string tempPortInstance,tempPortPtr,tempPortCategory;
      tempPortInstance=tempPortPtr=tempPortCategory="";
      
      for(unsigned int i1=0;i1<portType.length();i1++)
	{
	  char tmp=portType.at(i1);
	  if((tmp>='A')&&(tmp<='Z'))
	   {
	     tempPortInstance.append(1,(char)tolower(tmp));
	     
	   }
	  
	}
        
        tempPortInstance="provides"+tempPortInstance;
	// char tmp=portType.at(0);
	// tempPortCategory=portType.substr(1,portType.length());
	
	//tempPortCategory="provides"+tempPortCategory.insert(0,1,(char)tolower(tmp));
	tempPortCategory=(string)providesPortsList[i]->GetDesc();
	//tempPortPtr=tempPortCategory+"Ptr";    
	tempPortPtr=portName+"::pointer("+tempPortInstance+")";

	  
	componentSourceFile<<"\n"<<SP<<portName<<" *"<<tempPortInstance<<" = new "
		       <<portName<<"();";
	componentSourceFile<<"\n"<<SP<<tempPortInstance<<"->setParent(this);";
	// componentSourceFile<<"\n"<<SP<<portName<<"::pointer "<<tempPortPtr<<" = "
// 			   <<portName<<"::pointer("<<tempPortInstance<<");";
	componentSourceFile<<"\n"<<SP<<"svc->addProvidesPort("<<tempPortPtr<<", \""<<tempPortCategory
			   <<"\",\"sci.cca.ports."<<portType<<"\",sci::cca::TypeMap::pointer(0));";

	componentSourceFile<<"\n";
	
    }
      
  for(unsigned int i=0;i<usesPortsList.size();i++)
    {
      string portName= string(usesPortsList[i]->GetName());
      string portType=string(usesPortsList[i]->GetType());  
      string tempPortCategory;

      char tmp=portType.at(0);
      tempPortCategory=portType.substr(1,portType.length());
      tempPortCategory="uses"+tempPortCategory.insert(0,1,(char)tolower(tmp));

      componentSourceFile<<"\n"<<SP<<"svc->registerUsesPort(\""<<tempPortCategory;
      componentSourceFile<<"\",\"sci.cca.ports."<<portType<<"\",props);";
    }
 
	       
  componentSourceFile<<"\n}";

    
}
void ComponentSkeletonWriter::writeGoAndUiFunctionsCode()
{
 
 for(unsigned int i=0;i<providesPortsList.size();i++)
    {
      
      string portname=string(providesPortsList[i]->GetType());
      string porttype=string(providesPortsList[i]->GetType());
      std::cout<<"\nhere in ckw "<<porttype.c_str()<<"\n";
      // if(strcmp(porttype.c_str(),"UIPort")==0);
      if(porttype.compare(string("UIPort"))==0)
      //if(porttype.c_str()=="UIPort")
      
	{
	  componentSourceFile<<"\nint "<<providesPortsList[i]->GetName()<<"::ui()\n{\n}\n";
	}
      //if(porttype.c_str()== "GoPort")
      if(porttype.compare(string("GoPort"))==0)
	//if(strcmp(porttype.c_str(),"GoPort")==0);
        {
	  	  
	   componentSourceFile<<"\nint "<<providesPortsList[i]->GetName()<<"::go()\n{\n}\n";
	}
    }
 componentSourceFile<<std::endl;
}

void ComponentSkeletonWriter::ComponentMakeFileCode()
{
    
  componentMakeFile<<"#Sample Make File ";
  componentMakeFile<<"\n\ninclude $(SCIRUN_SCRIPTS)/smallso_prologue.mk";
  componentMakeFile<<"\n\nSRCDIR := CCA/Components/"<<compName;
  componentMakeFile<<"\n\nSRCS += "<<SP<<"$(SRCDIR)/"<<compName<<".cc \\" ;
  componentMakeFile<<"\n\nPSELIBS := Core/CCA/SSIDL Core/CCA/PIDL Core/CCA/Comm \\";
  componentMakeFile<<"\n"<<SP<<"Core/CCA/spec Core/Thread Core/Containers Core/Exceptions";
  componentMakeFile<<"\n\nCFLAGS += $(WX_CXXFLAGS)\nCXXFLAGS += $(WX_CXXFLAGS)\nLIBS := $(WX_LIBRARY)"
		   <<"\n\ninclude $(SCIRUN_SCRIPTS)/smallso_epilogue.mk";
  componentMakeFile<<"\n\n$(SRCDIR)/"<<compName<<".o: Core/CCA/spec/cca_sidl.h";
}





}













