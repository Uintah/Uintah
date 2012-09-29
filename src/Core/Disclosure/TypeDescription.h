/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_TypeDescription_H
#define UINTAH_HOMEBREW_TypeDescription_H

#include   <string>

#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI


namespace Uintah {

using std::string;

class Variable;

/**************************************
     
     CLASS
       TypeDescription
      
       Short Description...
      
     GENERAL INFORMATION
      
       TypeDescription.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
      
     KEYWORDS
       TypeDescription
      
     DESCRIPTION
       Long description...
      
     WARNING
      
****************************************/
    
class TypeDescription {
public:
  enum Type {
    CCVariable,
    NCVariable,
    SFCXVariable,
    SFCYVariable,
    SFCZVariable,
    ParticleVariable,
    PerPatch,
    Point,
    Vector,
    Matrix3,
    ReductionVariable,
    SoleVariable,
    double_type,
    float_type,
    bool_type,
    int_type,
    short_int_type,
    long_type,
    long64_type,
    Short27,   // for Fracture
    Unknown,
    Other
  };

  TypeDescription(Type type, const string& name,
                  bool isFlat, MPI_Datatype (*make_mpitype)());
  TypeDescription(Type type, const string& name,
                  bool isFlat, MPI_Datatype mpitype);
  TypeDescription(Type type, const string& name,
                  Variable* (*maker)(),
                  const TypeDescription* subtype);
     
  bool isReductionVariable() const {
    return d_type == ReductionVariable;
  }
  Type getType() const {
    return d_type;
  }
  const TypeDescription* getSubType() const {
    return d_subtype;
  }
  string getName() const;
  string getFileName() const;

  bool isFlat() const {
    return d_isFlat;
  }

  MPI_Datatype getMPIType() const;

  struct  Register {
    Register(const TypeDescription*);
    ~Register();
  };
  static const TypeDescription* lookupType(const string&);

  Variable* createInstance() const;

  ~TypeDescription();

  static void deleteAll();
     
private:
  Type d_type;
  const TypeDescription* d_subtype;
  string d_name;
  bool d_isFlat;
  mutable MPI_Datatype d_mpitype;
  MPI_Datatype (*d_mpitypemaker)();
  Variable* (*d_maker)();

  TypeDescription(const TypeDescription&);
  TypeDescription& operator=(const TypeDescription&);

  void register_type();
};

} // End namespace Uintah


#endif

