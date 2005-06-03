
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
			     const CellIterator&,
			     vector<constCCVariable<double> >& independents);
    virtual double interpolate(int index, vector<double>& independents);
  private:

    struct InterpAxis {
      InterpAxis(int size, int stride);
      InterpAxis(const InterpAxis* copy, int newStride);
      vector<double> weights;
      vector<long> offset;
      bool uniform;
      double dx;
      int useCount;
      void finalize();
      bool sameAs(const InterpAxis* b) const;
    };

    struct Ind {
      string name;
    };
    vector<Ind*> inds;

    struct Expr {
      char op;
      Expr(char op, Expr* child1, Expr* child2)
        : op(op), child1(child1), child2(child2)
      {
      }
      Expr(int var)
        : op('d'), child1(0), child2(0), var(var)
      {
      }
      Expr(const string& id)
        : op('i'), child1(0), child2(0), id(id)
      {
      }
      Expr(double constant)
        : op('c'), child1(0), child2(0), constant(constant)
      {
      }
      Expr* child1;
      Expr* child2;
      int var;
      double constant;
      string id;
      ~Expr() {
        if(child1)
          delete child1;
        if(child2)
          delete child2;
      }
    };

    struct Dep {
      string name;
      enum Type {
        ConstantValue,
        DerivedValue,
        TableValue
      } type;
      double constantValue;
      double* data;
      Expr* expression;
      vector<Ind*> myinds;
      vector<InterpAxis*> axes;
      Dep(Type type) : type(type) { data = 0; expression = 0; }
      ~Dep();
      void addAxis(InterpAxis*);
    };
    vector<Dep*> deps;

    Expr* parse_addsub(string::iterator&  begin, string::iterator& end);
    Expr* parse_muldiv(string::iterator&  begin, string::iterator& end);
    Expr* parse_sign(string::iterator&  begin, string::iterator& end);
    Expr* parse_idorconstant(string::iterator&  begin, string::iterator& end);
    void evaluate(Expr* expr, vector<InterpAxis*>& out_axes,
                  double* data, int size);
    void checkAxes(const vector<InterpAxis*>& a, const vector<InterpAxis*>& b,
                   vector<InterpAxis*>& out_axes);

    string filename;
    bool file_read;

    struct DefaultValue {
      string name;
      double value;
    };
    vector<DefaultValue*> defaults;

    int getInt(istream&);
    double getDouble(istream&);
    string getString(istream&);
    string getLine(istream&);

    bool startline;
    void error(istream& in);
    void skipComments(istream& in);
    void eatWhite(istream& in);
  };
} // End namespace Uintah
    
#endif
