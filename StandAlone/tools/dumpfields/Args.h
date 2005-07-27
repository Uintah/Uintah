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
    
    std::vector<std::string> allTrailing(int minnumber=1);
    
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
