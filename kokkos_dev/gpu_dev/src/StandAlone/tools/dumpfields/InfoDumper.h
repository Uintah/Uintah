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


#ifndef DUMPFIELDS_INFO_DUMPER_H
#define DUMPFIELDS_INFO_DUMPER_H

#include <StandAlone/tools/dumpfields/FieldDumper.h>
#include <StandAlone/tools/dumpfields/Args.h>
#include <StandAlone/tools/dumpfields/FieldSelection.h>

#include <fstream>

namespace Uintah {
  
  class InfoOpts {
  public:
    InfoOpts(SCIRun::Args & args);
    
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
      Step(DataArchive * da, string tsdir, int timestep, double time, int index, 
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
    Step * addStep(int timestep, double time, int index);
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

