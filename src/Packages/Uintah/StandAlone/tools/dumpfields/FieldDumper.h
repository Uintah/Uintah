#ifndef DUMPFIELDS_FIELD_DUMPER_H
#define DUMPFIELDS_FIELD_DUMPER_H

#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <string>

namespace Uintah {
  using namespace SCIRun;
  using namespace std;
  
  class FieldDumper 
  {
  protected:
    FieldDumper(DataArchive * da, string basedir);
    
    string dirName(double tval, int iset) const;
    string createDirectory();
    
  public:
    virtual ~FieldDumper();
    
    // extension for the output directory
    virtual string directoryExt() const = 0;
    
    // add a new field
    virtual void addField(string fieldname, const Uintah::TypeDescription * type) = 0;
    
    // dump a single step
    class Step {
    protected:
      Step(string tsdir, int index, double time, bool nocreate=false);
      
      string fileName(string variable_name, string extension="") const;
      string fileName(string variable_name, int materialNum, string extension="") const;
      
    public:
      virtual ~Step();
      
      virtual string infostr() const = 0;
      virtual void storeGrid() = 0;
      virtual void storeField(string fieldname, const Uintah::TypeDescription * type) = 0;
      
    public: // FIXME: 
      string tsdir_;
      int    index_;
      double time_;
    };
    virtual Step * addStep(int index, double time, int iset) = 0;
    virtual void finishStep(Step * step) = 0;
    
    DataArchive * archive() const { return this->da_; }
    
  private:
    static string mat_string(int mval);
    static string time_string(double tval);
    static string step_string(int istep);
    
  private:
    DataArchive* da_;
    string       basedir_;
  };
}

#endif

