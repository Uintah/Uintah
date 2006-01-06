#ifndef DUMPFIELDS_INFO_DUMPER_H
#define DUMPFIELDS_INFO_DUMPER_H

#include <Packages/Uintah/StandAlone/tools/dumpfields/FieldDumper.h>
#include <Packages/Uintah/StandAlone/tools/dumpfields/Args.h>
#include <Packages/Uintah/StandAlone/tools/dumpfields/FieldSelection.h>

#include <fstream>

namespace Uintah {
  
  class InfoOpts {
  public:
    InfoOpts(Args & args);
    
    bool  showeachmat;
    
    static std::string options() { 
      return \
        std::string("      -allmats               show material-by-material information");
    }
  };
  
  class InfoDumper : public FieldDumper 
  {
  public:
    InfoDumper(DataArchive* da, string basedir, 
               const InfoOpts & opts, const FieldSelection & fselect);
  
    string directoryExt() const { return "info"; }
    void addField(string /*fieldname*/, const Uintah::TypeDescription * /*theType*/) {}
  
    class Step : public FieldDumper::Step {
    public:
      Step(DataArchive * da, string tsdir, int index, double time, 
           const InfoOpts & opts, const FieldSelection & fselect);
      
      void storeGrid () {}
      void storeField(string fieldname, const Uintah::TypeDescription * type);
      
      string infostr() const { return info_; }
      
    private:
      string                 info_;
      DataArchive*           da_;
      InfoOpts               opts_;
      const FieldSelection & fselect_;
    };
    
    //
    Step * addStep(int index, double time, int iset);
    void   finishStep(FieldDumper::Step * s);
  
  private:
    std::ofstream idxos_;
    int      nbins_;
    double   range[2];
    FILE*    filelist_;
    
    InfoOpts               opts_;
    const FieldSelection & fselect_;
  };

}

#endif

