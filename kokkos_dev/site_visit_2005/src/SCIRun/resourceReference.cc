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
 *  resourceReference.cc 
 *
 *  Written by:
 *   Kostadin Damevski & Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   April 2003 
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <SCIRun/resourceReference.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/TypeMap.h>
using namespace std;
using namespace SCIRun;

namespace SCIRun {

resourceReference::resourceReference(const std::string& name,
                       const ::SSIDL::array1< ::std::string>& URLs)
{
  this->name = name;
  this->size= URLs.size();
  for(int i=0; i < size ;i++)
    {
    this->URLs.push_back(URLs[i]);
    }
}

resourceReference::~resourceReference()
{
}

sci::cca::Loader::pointer resourceReference::getPtrToAll()
{
  std::cerr<<"calling resourceReference::getPtrToAll()... size="<<URLs.size()<<" ";
  Object::pointer obj=PIDL::objectFrom(URLs,1,0);
  std::cerr<<"Done" << std::endl;;
  std::cerr<<"calling pidl_cast...";
  sci::cca::Loader::pointer sc=pidl_cast<sci::cca::Loader::pointer>(obj);
  std::cerr<<"Done" << std::endl;
  return sc;
}

int resourceReference::getSize() 
{
  return size;
}

std::string resourceReference::getName() 
{
  return name;
}

void resourceReference::print(std::ostream& dbg)
{
  dbg << "********" << name << " (np=" << size << ")********\n";
  for(int i=0; i < size ;i++)
    {
    dbg << URLs[i].getString() << std::endl;
    }
  dbg << "*****************************************************************\n";
}
    

void resourceReference::listAllComponentTypes(::SSIDL::array1<std::string> &typeList)
{
  std::cerr<<"calling   node(0)->getAllComponentTypes(typeList)" << std::endl;
  node(0)->getAllComponentTypes(typeList);
}

sci::cca::Component::pointer resourceReference::createInstance(const std::string& name,
					      const std::string& type,
					      std::vector<int> nodes)
{
  //for all nodes
  SSIDL::array1<std::string> comURLs1;
  comURLs1.push_back("                                                                   ");
  comURLs1.push_back("                                                                   ");
  comURLs1.push_back("                                                                   ");

  //Inform everyone else of my distribution
  //(this is in correspondence with the instantiate() call)
  
  if(ploader.isNull())
    {
    Index* dr[1]; 
    //std::cerr<<"URLs.size="<< URLs.size() <<std::endl;
    dr[0] = new Index(0, URLs.size(), 1);  //first, last, stride
    MxNArrayRep* arrr = new MxNArrayRep(1,dr);
    //std::cerr<<"ploader->setCallerDistribution...";
    ploader=resourceReference::getPtrToAll();
    ploader->setCallerDistribution("dURL",arrr);   //client is caller
    delete dr[0];
    }
  std::cerr<<"Done" << std::endl;
 
//<<<<<<< resourceReference.cc
  cerr<<"comURLs1.size before create Instance="<<comURLs1.size()<<endl;
  cerr<<"ploader->createPInstance(name, type, comURLs1)...";

  SSIDL::array1<int> nodeSet;
  nodeSet.push_back(0);
  nodeSet.push_back(2);  //use nodes 0-2
  //  sci::cca::TypeMap::pointer properties(new TypeMap);
  //properties->putIntArray("nodes", nodeSet);
  //  ploader->createPInstance(name, type, properties, comURLs1);
  ploader->createPInstance(name, type, sci::cca::TypeMap::pointer(0), comURLs1);
//=======
//  std::cerr<<"comURLs1.size before create Instance="<<comURLs1.size()<<std::endl;
//  std::cerr<<"ploader->createPInstance(name, type, comURLs1)...";
//  ploader->createPInstance(name, type, sci::cca::TypeMap::pointer(0), comURLs1);
//>>>>>>> 1.8

  std::cerr << "Done" << std::endl;
  std::cerr<<"comURLs1.size="<<comURLs1.size()<<std::endl;


  std::vector<URL> comURLs;
  for(unsigned int i=0; i<comURLs1.size(); i++){
    cerr<<"comURLs["<<i<<"]="<<comURLs1[i]<<endl;
    if(comURLs1[i]!="") comURLs.push_back(comURLs1[i]);
  }

  Object::pointer obj=PIDL::objectFrom(comURLs);
  sci::cca::Component::pointer pcom=pidl_cast<sci::cca::Component::pointer>(obj);
  return pcom;


  //for sequential component
  //std::string comURL;
  //node(0)->createInstance(name, type, comURL);
  //return comURL;
}

int resourceReference::shutdown(float time)
{
  //ifstream f("loader.url");
  //std::string s;
  //f>>s;
  //f.close();
  std::string s=URLs[0].getString();
  std::cout<<"calling objectFrom $"<<s<<"$"<<std::endl;
  Object::pointer obj=PIDL::objectFrom(s);//.getString());
  if(obj.isNull()){
    std::cerr<<"Cannot get loader from url="<<URLs[0].getString()<<std::endl;
    return 0;
  }
  std::cout<<"Loader obj obtained"<<std::endl;
  sci::cca::Loader::pointer node0=pidl_cast<sci::cca::Loader::pointer>(obj);
  std::cout<<"Loader obj casted"<<std::endl;
  std::cout<<"Calling node0->listAllComponents"<<std::endl;
  node0->shutdown(time);
  return 0;
}

sci::cca::Loader::pointer 
resourceReference::node(int i)
{
  std::string s=URLs[i].getString();
  Object::pointer obj=PIDL::objectFrom(s);
  if(obj.isNull()){
    std::cerr<<"Cannot get loader from url="<<URLs[i].getString()<<std::endl;
    return sci::cca::Loader::pointer(0);
  }
  sci::cca::Loader::pointer nd=pidl_cast<sci::cca::Loader::pointer>(obj);
  return nd;
}

} // end namespace SCIRun





