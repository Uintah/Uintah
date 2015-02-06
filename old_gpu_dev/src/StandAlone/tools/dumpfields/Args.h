/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#ifndef DUMPFIELDS_ARGS_H
#define DUMPFIELDS_ARGS_H

#include <vector>
#include <string>

namespace SCIRun {
  
  class Args {
  public:
    Args(int argc, char ** argv);
    
    std::string              progname() const { return pname; }
    
    std::string              trailing();
    std::vector<std::string> trailing(int number);
    
    std::vector<std::string> allTrailing(unsigned int minnumber=1);
    
    // is this option present
    bool        getLogical(std::string name);
    
    // get typed argument, or raise error if not found
    std::string getString (std::string name);
    double      getReal   (std::string name);
    int         getInteger(std::string name);
    
    // get typed argument, or default if not found
    std::string getString (std::string name, std::string defval);
    double      getReal   (std::string name, double defval);
    int         getInteger(std::string name, int defval);
    
    bool hasUnused() const;
    std::vector<std::string> unusedArgs() const;
    
  private:
    std::string              pname;
    std::vector<std::string> args;
    std::vector<bool>        argused;
  };
}

#endif
