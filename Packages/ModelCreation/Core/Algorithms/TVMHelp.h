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

#ifndef CORE_ALGORITHMS_TVMHELP_H
#define CORE_ALGORITHMS_TVMHELP_H 1

#include <sgi_stl_warnings_off.h>
#include <string>
#include <list>
#include <sgi_stl_warnings_on.h>

namespace TensorVectorMath {

class TVMHelpEntry {
  public:
    std::string functionname;
    std::string functionsyntax;
    std::string description;
};

inline bool operator<(const TVMHelpEntry& e1,const TVMHelpEntry& e2)
{
  return(e1.functionname<e2.functionname);
}

class TVMHelp {
  
  public:
  
    inline TVMHelp();
    inline std::string gethelp(bool addelem = false);
    
  private:
    inline void add(std::string functionname,std::string syntax,std::string description);
    void createhelplist();
    void createhelplist_element();
    
    std::string helphtml_;
    std::list<TVMHelpEntry> helplist_;
};

inline std::string TVMHelp::gethelp(bool addelem)
{
  if (helphtml_ == "")
  {
    createhelplist();
    if (addelem)
    {
      createhelplist_element();
    }
    
    helphtml_ = "<h4>Tensor/Vector/Scalar functions</h4>\n";

    helplist_.sort();
    
    std::list<TVMHelpEntry>::iterator it;
    std::string oldname;

    for (it = helplist_.begin(); it != helplist_.end(); it++)
    {
      std::string helptext;
      if (oldname != (*it).functionname) helptext = "<h5>"+(*it).functionname+"</h5>\n";
      helptext += "<p>"+(*it).functionsyntax+"</p>\n";
      helptext += "<p>"+(*it).description+"</p>\n";
      helphtml_ += helptext;    
      oldname = (*it).functionname;
    }  
  }

  return(helphtml_);
}

inline void TVMHelp::add(std::string functionname,std::string functionsyntax, std::string description)
{
  TVMHelpEntry entry;
  entry.functionname = functionname;
  entry.functionsyntax = functionsyntax;
  entry.description = description;
  helplist_.push_back(entry);
}

inline TVMHelp::TVMHelp() :
  helphtml_("")
{
}



} // end namespace

#endif
