#ifndef CORE_DISCLOSURE_TYPEDESCRIPTION_H
#define CORE_DISCLOSURE_TYPEDESCRIPTION_H

/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <Core/Parallel/UintahMPI.h>

#include <string>

namespace Uintah {

class Variable;

/**************************************
     
     CLASS
       TypeDescription
      
      
     GENERAL INFORMATION
      
       TypeDescription.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
      
     KEYWORDS
       TypeDescription
      
     DESCRIPTION
      
****************************************/
    
class TypeDescription {

public:

  enum Type
  {  CCVariable
  ,  NCVariable
  ,  SFCXVariable
  ,  SFCYVariable
  ,  SFCZVariable
  ,  ParticleVariable
  ,  PerPatch
  ,  Point
  ,  Vector
  ,  Matrix3
  ,  ReductionVariable
  ,  SoleVariable
  ,  double_type
  ,  float_type
  ,  bool_type
  ,  int_type
  ,  short_int_type
  ,  long_type
  ,  long64_type
  ,  Short27   // for Fracture
  ,  Int130 
  ,  Stencil4
  ,  Stencil7
  ,  IntVector
  ,  Unknown
  ,  Other
  };

  // Coverts the 'type' enum to a string.
  static std::string toString( Type type );

  TypeDescription( Type type, const std::string& name, bool isFlat, MPI_Datatype (*make_mpitype)() );

  TypeDescription( Type type, const std::string& name, bool isFlat, MPI_Datatype mpitype );

  TypeDescription( Type type, const std::string& name, Variable* (*maker)(), const TypeDescription* subtype );

  ~TypeDescription(){};

  std::string getName() const;

  std::string getFileName() const;

  static const TypeDescription* lookupType( const std::string& );

  Variable* createInstance() const;

  static void deleteAll();
     
  bool isReductionVariable() const {
    return d_type == ReductionVariable;
  }

  Type getType() const {
    return d_type;
  }

  const TypeDescription* getSubType() const {
    return d_subtype;
  }

  bool isFlat() const {
    return d_isFlat;
  }

  MPI_Datatype getMPIType() const;

  // Our main variables (CCVariables, etc) create a static variable of 
  // this type.  This is used to 'register' the variable type (eg: NCVariable<int>)
  // with the TypeDescription system when the Variable classes are originally
  // loaded (usually at program start up).
  struct Register {
    Register( const TypeDescription* )
    {
      // Actual registration of Variable Type happens when the 'td' variable is originally created.
    }
    ~Register(){};
  };

  // These are for uniquely identifying the Uintah::CrowdMonitors<Tag>
  // used to protect multi-threaded access to global data structures
  struct register_tag{}; // used in register_type()
  struct lookup_tag{};   // used in lookup_type()


private:

  // disable copy, assignment, and move
  TypeDescription( const TypeDescription & )            = delete;
  TypeDescription& operator=( const TypeDescription & ) = delete;
  TypeDescription( TypeDescription && )                 = delete;
  TypeDescription& operator=( TypeDescription && )      = delete;

  void register_type();

  Type                    d_type{};
  const TypeDescription * d_subtype{nullptr};
  std::string             d_name{};
  bool                    d_isFlat{false};
  mutable MPI_Datatype    d_mpitype{};

  MPI_Datatype (*d_mpitypemaker)(){nullptr};
  Variable* (*d_maker)(){nullptr};

};

} // End namespace Uintah

#endif // CORE_DISCLOSURE_TYPEDESCRIPTION_H
