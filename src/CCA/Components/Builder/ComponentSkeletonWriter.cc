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

ComponentSkeletonWriter::
        ComponentSkeletonWriter(const std::string &cname, const std::vector<PortDescriptor*> pp, const std::vector<PortDescriptor*> up) : sp("    "), compName(cname), providesPortsList(pp), usesPortsList(up)
{ }

void ComponentSkeletonWriter::ComponentClassDefinitionCode()
{

  componentHeaderFile << "//Sample header File\n\n";
  
  componentHeaderFile <<"\n#ifndef SCIRun_Framework_"<<compName<<"_h";
  componentHeaderFile <<"\n#define SCIRun_Framework_"<<compName<<"_h";
  componentHeaderFile <<"\n\n#include <Core/CCA/spec/cca_sidl.h>";
  //componentHeaderFile <<" \n#include <CCA/Components/"<<compName<<"/"<<compName<<"_sidl.h>
  componentHeaderFile <<"\nusing namespace SCIRun;\n\n";

  

  componentHeaderFile <<"\nextern \"C\" sci::cca::Component::pointer make_SCIRun_"<<compName<<"()";
  componentHeaderFile <<"\n{\n"<<sp<<"return sci::cca::Component::pointer(new "<<compName<<"());\n}\n\n";



  componentHeaderFile << "\nclass " << compName << ": public sci::cca::Component { \npublic:";
  componentHeaderFile << "\n" << sp << compName << "();\n" << sp << "virtual ~"<< compName << "();";
  componentHeaderFile << "\n" << sp << "virtual void setServices(const sci::cca::Services::pointer& svc);";
  componentHeaderFile << "\nprivate:\n" << sp << compName << "(const " << compName << "&);";
  componentHeaderFile << "\n" << sp << compName<< "& operator=(const " 
		      << compName << "&);\n" << sp << "sci::cca::Services::pointer services;\n};" << std::endl;
}

void ComponentSkeletonWriter::PortClassDefinitionCode()
{
  for (unsigned int i = 0; i < providesPortsList.size(); i++) 
    {
    // componentHeaderFile<<"\nclass "<<providesPortsList[i].name<<" : public sci::cca::ports::"
      //<<providesPortsList[i].type<<" {\n";
    
    componentHeaderFile << "\nclass " << (providesPortsList[i])->GetName()
			<< " : public sci::cca::ports::" << (providesPortsList[i])->GetType() << " {\n";
    
    // componentHeaderFile<<"public:\n"<<sp<<"virtual ~"<<providesPortsList[i].name<<"(){}";
    componentHeaderFile << "public:\n" << sp << "virtual ~"<< (providesPortsList[i])->GetName() << "(){}";
    componentHeaderFile << "\n" << sp << "void setParent(" << compName << " *com)";
    componentHeaderFile << "{ this->com = com; }\n" << sp << compName << " *com;";
    componentHeaderFile << "\n};" << std::endl;
  }
  componentHeaderFile << "\n#endif" << std::endl;
}


void ComponentSkeletonWriter::ComponentSourceFileCode()
{
  componentSourceFile<<"//Sample Source code\n\n";
 
  //Header files

   componentSourceFile<<"\n#include <Core/Thread/Time.h>\n#include <SCIRun/TypeMap.h>\n#include <iostream>"
		      <<"\n#include <unistd.h>\n#include<CCA/Components/"<<compName
		      <<"/"<<compName<<".h>\nusing namespace SCIRun;\n\n";

  componentSourceFile<<"\n"<<compName<<"()::"<<compName<<"()"<<"\n{\n}";
  componentSourceFile<<"\n"<<compName<<"()::~"<<compName<<"()\n{";
  
  //Destructor code

  
  for(int i=0;i<providesPortsList.size();i++)
    {
        
	componentSourceFile << "\n"<<sp<<"services->removeProvidesPort(\""<<providesPortsList[i]->GetName()<<"\");";
    }

  for(int i=0;i<usesPortsList.size();i++)
    {
      componentSourceFile << "\n"<<sp<<"services->unregisterUsesPort(\""<<usesPortsList[i]->GetName()<<");";
    }

  //Set services code

  componentSourceFile<<"\n}\n"<<compName<<"::setServices(const sci::cca::Services::pointer& svc)\n\{\n";
  componentSourceFile<<sp<<"services = svc;";
  componentSourceFile<<"\n"<<sp<<"svc->registerForRelease(sci::cca::ComponentRelease::pointer(this)); ";
  componentSourceFile<<"\n"<<sp<<"sci::cca::TypeMap::pointer props = svc->createTypeMap(); \n\n";

  for(int i=0;i<providesPortsList.size();i++)
    {
      
      string portName= string(providesPortsList[i]->GetName());
      string portType=string(providesPortsList[i]->GetType());
      string tempPortInstance,tempPortPtr,tempPortCategory;
      
      if(strcmp(portType.c_str(),"UIPort")==0)
	{
	  tempPortInstance=string("uip");
	  tempPortPtr=string("uiPortPtr");
	  tempPortCategory=string("ui");
	}
      else if(strcmp(portType.c_str(),"GoPort")==0)
	{
	  tempPortInstance=string("gop");
	  tempPortPtr=string("goPortPtr");
	  tempPortCategory=string("go");
	}
      else if(strcmp(portType.c_str(),"StringPort")==0)
	{
	  tempPortInstance=string("sp");
	  tempPortPtr=string("stringPortPtr");
	  tempPortCategory=string("string");
	} 
         
        

          componentSourceFile<<"\n"<<sp<<portType<<" *"<<tempPortInstance<<" = new "
			 <<portName<<"();";
	  componentSourceFile<<"\n"<<sp<<tempPortInstance<<"->setParent(this);";
          componentSourceFile<<"\n"<<sp<<portName<<"::pointer "<<tempPortPtr<<" = "
			     <<portName<<"::pointer("<<tempPortInstance<<");";
	  componentSourceFile<<"\n"<<sp<<"svc->addProvidesPort("<<tempPortPtr<<", \""<<tempPortCategory
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
      string portName= string(usesPortsList[i]->GetName());
      string portType=string(usesPortsList[i]->GetType());  
      string tempPortCategory;
      if(strcmp(portType.c_str(),"UIPort")==0)
	{
	  
	  tempPortCategory=string("ui");
	}
      else if(strcmp(portType.c_str(),"GoPort")==0)
	{
	  
	  tempPortCategory=string("go");
	}
      else if(strcmp(portType.c_str(),"StringPort")==0)
	{
	  
	  tempPortCategory=string("string");
	} 
         
      
	  componentSourceFile<<"\n"<<sp<<"svc->registerUsesPort(\""<<tempPortCategory;
          componentSourceFile<<"\",\"sci.cca.ports."<<portType<<"\",props);";
    }
 
	       
  componentSourceFile<<"\n}";

  //go() and ui() functions

  for(int i=0;i<providesPortsList.size();i++)
    {
      string portname=string(providesPortsList[i]->GetType());
      string porttype=string(providesPortsList[i]->GetType());
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
  PortClassDefinitionCode();
  ComponentSourceFileCode();


  componentHeaderFile.close(); 
  componentSourceFile.close(); 
}

}
