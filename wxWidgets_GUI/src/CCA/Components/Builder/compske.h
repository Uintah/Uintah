
#ifndef compske_h
#define compske_h


#include<iostream>
#include<fstream>
#include<unistd.h>
class port;
class ske;

namespace GUIBuilder{

class port
{
public:
	std::string name;
	std::string type;
};
class ske
{
	public: 
		int a;	
		std::string compname;
		std::vector<port> pp;
		std::vector<port> up;
		std::ofstream myfile;
		void compcode();
		void  portcode();
		void gen();
};


void ske::compcode()
{
	
  	myfile << "//Sample header File\n\n";
	myfile << "\nclass "<<compname<<": public sci::cca::Component { \npublic:";
	myfile << "\n\t"<<compname<<"();\n\tvirtual ~"<<compname<<"();";
  	myfile << "\n\tvirtual void setServices(const sci::cca::Services::pointer& svc);";
	myfile << "\nprivate:\n\t"<<compname<<"(const "<<compname<<"&);";
	myfile << "\n\t"<<compname<<"& operator=(const "<<compname<<"&);\n\tsci::cca::Services::pointer services;\n};";
	
}
void ske::portcode()
{
	for(int i=0;i<pp.size();i++)
	{
		myfile<<"\nclass "<<pp[i].name<<" : public sci::cca::ports::"<<pp[i].type<<" {\n";
		myfile<<"public:\n\tvirtual ~"<<pp[i].name<<"(){}";
		myfile<<"\n\tvoid setParent("<<compname<<" *com)";
		myfile<<"{ this->com = com; }\n\t"<<compname<<" *com;";
		myfile<<"\n};";
	}
}

void ske::gen()
{
	std::string direct="mkdir "+compname;
	const char * dir=direct.c_str();
	system(dir);
	std::string sopf="./"+compname+"/"+compname+".h";
// 	cout<<sopf<<endl;
	const char *copf=sopf.c_str();
	myfile.open(copf);
	compcode();
	portcode();
	myfile.close();
}

}
#endif
