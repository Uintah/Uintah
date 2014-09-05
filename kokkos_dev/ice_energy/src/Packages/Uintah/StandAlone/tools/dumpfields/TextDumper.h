#ifndef DUMPFIELDS_TEXTDUMPER_H
#define DUMPFIELDS_TEXTDUMPER_H

#include "FieldDumper.h"
#include "Args.h"
#include "FieldSelection.h"

namespace Uintah {

  class TextOpts {
  public:
    TextOpts(Args & args);
    
    bool onedim;
    bool tseries;
    
    static std::string options() { 
      return \
        std::string("      -onedim                generate one dim plots") + "\n" +
        std::string("      -tseries               generate single time series file");
    }
  };
  
  class TextDumper : public FieldDumper 
  {
  public:
    TextDumper(DataArchive * da, string basedir, const TextOpts & opts, const FieldSelection & flds);
    
    string directoryExt() const { return "text"; }
    void addField(string fieldname, const Uintah::TypeDescription * /*type*/);
  
    class Step : public FieldDumper::Step {
    public:
      Step(DataArchive * da, string basedir, int index, double time, 
           const TextOpts & opts, const FieldSelection & flds);
      
      string infostr() const { return tsdir_; }
      void   storeGrid () {}
      void   storeField(string fieldname, const Uintah::TypeDescription * type);
    
    private:
      DataArchive *        da_;
      string               outdir_;
      TextOpts             opts_;
    const FieldSelection & flds_;
    };
  
    //
    Step * addStep(int index, double time, int iset);
    void   finishStep(FieldDumper::Step * s);
    
  private:
    ofstream               idxos_;
    TextOpts               opts_;
    const FieldSelection & flds_;
    FILE*                  filelist_;
  };
}

#endif
