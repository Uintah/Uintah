//
// FileReader.cc,v 1.3 2002/01/29 20:21:07 okellogg Exp
//
#include "FileReader.h"
#include <iostream>
#include <fstream>

ACE_RCSID(FileReader, FileReader, "FileReader.cc,v 1.3 2002/01/29 20:21:07 okellogg Exp")

FileReader::FileReader (CORBA::ORB_ptr orb)
  : orb_ (CORBA::ORB::_duplicate (orb))
{
}

FileReader* _main (int argc, char *argv[]);

extern "C" SCIRun::tao::Component* make_Tao_FileReader()
{
  return _main(0,NULL);  
}

void FileReader::setServices(sci::cca::TaoServices::pointer services) 
{
  services->addProvidesPort("hello", "corba.FileReader");
}

CORBA::Long 
FileReader::getPDEdescription (
			       Test::FileReader::double_array_out nodes ,
			       Test::FileReader::long_array_out boundaries,
			       Test::FileReader::long_array_out dirichletNodes,
			       Test::FileReader::double_array_out dirichletValues
			       ACE_ENV_ARG_DECL_WITH_DEFAULTS
			       )
  ACE_THROW_SPEC ((CORBA::SystemException ))
{
  nodes = new Test::FileReader::double_array;
  boundaries=new Test::FileReader::long_array;
  dirichletNodes=new Test::FileReader::long_array;
  dirichletValues=new Test::FileReader::double_array;


  ifstream is("/scratch/ACE_wrappers/TAO/tests/FileReader/L.pde");


  while(true){
    std::string name;
    is>>name;
    if(name=="node"){
      unsigned int cnt;
      is>>cnt;
      nodes->length(cnt*2);
      for(unsigned int i=0; i<cnt; i++){
	double x, y;
	is>>x>>y;
	nodes[i+i]=x;
	nodes[i+i+1]=y;
      }
    }
    else if(name=="boundary"){
      unsigned  int cnt;
      is>>cnt;
      boundaries->length(cnt);

      for(unsigned int i=0; i<cnt; i++){
	int index;
	is>>index;
	boundaries[i]=index;
      }
    }
    else if(name=="dirichlet"){
      unsigned int cnt;
      is>>cnt;
      dirichletNodes->length(cnt);
      dirichletValues->length(cnt);

      for(unsigned int i=0; i<cnt; i++){
	int index;
	is>>index;
	dirichletNodes[i]=index;
      }
      for(unsigned int i=0; i<cnt; i++){
	double value;
	is>>value;
	dirichletValues[i]=value;
      }
    }
    else if(name=="end") break;  
  }

  return 0;

}

