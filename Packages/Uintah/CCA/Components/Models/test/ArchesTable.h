
#ifndef Uintah_ArchesTable_h
#define Uintah_ArchesTable_h

#include <Packages/Uintah/CCA/Components/Models/test/TableInterface.h>

namespace Uintah {

/****************************************

CLASS
   ArchesTable
   
   Short description...

GENERAL INFORMATION

   ArchesTable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ArchesTable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class ArchesTable : public TableInterface {
  public:
    ArchesTable(ProblemSpecP& ps);
    virtual ~ArchesTable();

    virtual void addIndependentVariable(const string&);
    virtual int addDependentVariable(const string&);
    
    virtual void setup();
    
    virtual void interpolate(int index, CCVariable<double>& result,
			     vector<constCCVariable<double> >& independents);
  private:

    struct Ind {
      string name;
      vector<double> weights;
      vector<long> offset;

      bool uniform;
      double dx;
    };
    vector<Ind*> inds;

    struct Dep {
      string name;
      double* data;
    };
    vector<Dep*> deps;

    string filename;
    bool file_read;

    struct DefaultValue {
      string name;
      double value;
    };
    vector<DefaultValue*> defaults;
  };
} // End namespace Uintah
    
#endif
