/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef Uintah_ArchesTable_h
#define Uintah_ArchesTable_h

#include <CCA/Components/Models/FluidsBased/TableInterface.h>
#include <sstream>

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
    virtual void addIndependentVariable(const std::string&);
    virtual int addDependentVariable(const std::string&);
    
    virtual void setup(const bool cerrSwitch);
    
    virtual void interpolate(int index, CCVariable<double>& result,
                             const CellIterator&,
                             std::vector<constCCVariable<double> >& independents);
    virtual double interpolate(int index, std::vector<double>& independents);

  private:

    struct InterpAxis {
      InterpAxis(int size, int stride);
      InterpAxis(const InterpAxis* copy, int newStride);
      std::vector<double> weights;
      std::vector<long> offset;
      bool uniform;
      double dx;
      int useCount;
      void finalize();
      bool sameAs(const InterpAxis* b) const;
    };

    struct Ind {
      std::string name;
    };
    std::vector<Ind*> inds;

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
      Expr(const std::string& id)
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
      std::string id;
      ~Expr() {
        if(child1)
          delete child1;
        if(child2)
          delete child2;
      }
    };

    struct Dep {
      void outputProblemSpec(ProblemSpecP& ps);
      std::string name;
      enum Type {
        ConstantValue,
        DerivedValue,
        TableValue
      } type;
      double constantValue;
      double* data;
      std::string expr_string;
      Expr* expression;
      std::vector<Ind*> myinds;
      std::vector<InterpAxis*> axes;
      Dep(Type type) : type(type) { data = 0; expression = 0; }
      ~Dep();
      void addAxis(InterpAxis*);
    };
    std::vector<Dep*> deps;

    Expr* parse_addsub(std::string::iterator&  begin, std::string::iterator& end);
    Expr* parse_muldiv(std::string::iterator&  begin, std::string::iterator& end);
    Expr* parse_sign(std::string::iterator&  begin, std::string::iterator& end);
    Expr* parse_idorconstant(std::string::iterator&  begin, std::string::iterator& end);
    void evaluate(Expr* expr, std::vector<InterpAxis*>& out_axes,
                  double* data, int size);
    void checkAxes(const std::vector<InterpAxis*>& a, const std::vector<InterpAxis*>& b,
                   std::vector<InterpAxis*>& out_axes);

    std::string filename_;
    bool   file_read_;

    struct DefaultValue {
      void outputProblemSpec(ProblemSpecP& ps) {
        std::stringstream ss;
        ss << value;
        ProblemSpecP dv_ps = ps->appendElement("defaultValue",ss.str());
        dv_ps->setAttribute("name",name);
      };
      std::string name;
      double value;
    };
    std::vector<DefaultValue*> defaults;

  };
} // End namespace Uintah
    
#endif
