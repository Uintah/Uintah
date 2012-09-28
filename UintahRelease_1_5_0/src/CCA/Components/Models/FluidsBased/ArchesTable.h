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



#ifndef Uintah_ArchesTable_h
#define Uintah_ArchesTable_h

#include <CCA/Components/Models/FluidsBased/TableInterface.h>
#include <sstream>

namespace Uintah {

  using std::stringstream;

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

    virtual void outputProblemSpec(ProblemSpecP& ps);
    virtual void addIndependentVariable(const string&);
    virtual int addDependentVariable(const string&);
    
    virtual void setup(const bool cerrSwitch);
    
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
      void outputProblemSpec(ProblemSpecP& ps);
      string name;
      enum Type {
        ConstantValue,
        DerivedValue,
        TableValue
      } type;
      double constantValue;
      double* data;
      string expr_string;
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

    string filename_;
    bool   file_read_;

    struct DefaultValue {
      void outputProblemSpec(ProblemSpecP& ps) {
        stringstream ss;
        ss << value;
        ProblemSpecP dv_ps = ps->appendElement("defaultValue",ss.str());
        dv_ps->setAttribute("name",name);
      };
      string name;
      double value;
    };
    vector<DefaultValue*> defaults;

  };
} // End namespace Uintah
    
#endif
