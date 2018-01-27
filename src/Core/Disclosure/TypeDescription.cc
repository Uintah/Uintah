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

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/CrowdMonitor.hpp>
#include <Core/Parallel/MasterLock.h>
#include <Core/Util/Assert.h>

#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <vector>

using namespace Uintah;

namespace {

using register_monitor = Uintah::CrowdMonitor<TypeDescription::register_tag>;
using lookup_monitor   = Uintah::CrowdMonitor<TypeDescription::lookup_tag>;

Uintah::MasterLock get_mpi_type_lock{};
std::map<std::string, const TypeDescription*> * types_g     = nullptr;
std::vector<const TypeDescription*>           * typelist_g  = nullptr;
bool killed = false;

}


TypeDescription::TypeDescription(        Type          type
                                ,  const std::string & name
                                ,        bool          isFlat
                                ,        MPI_Datatype( *mpitypemaker )()
                                )
  : d_type{type}
  , d_name{name}
  , d_isFlat{isFlat}
  , d_mpitype{MPI_Datatype(-1)}
  , d_mpitypemaker{mpitypemaker}
{
  register_type();
}

TypeDescription::TypeDescription(        Type          type
                                ,  const std::string & name
                                ,        bool          isFlat
                                ,        MPI_Datatype  mpitype
                                )
  : d_type{type}
  , d_name{name}
  , d_isFlat{isFlat}
  , d_mpitype{mpitype}
{
  register_type();
}

TypeDescription::TypeDescription(        Type              type
                                ,  const std::string     & name
                                ,        Variable        * (*maker)()
                                ,  const TypeDescription * subtype
                                )
  : d_type{type}
  , d_subtype{subtype}
  , d_name{name}
  , d_mpitype{MPI_Datatype(-2)}
  , d_maker{maker}
{
  register_type();
}

void
TypeDescription::deleteAll()
{
  if( !types_g ) {
    ASSERT( !killed );
    ASSERT( !typelist_g );
    return;
  }

  killed = true;

  for(auto iter = typelist_g->begin(); iter != typelist_g->end();iter++) {
    delete *iter;
  }

  delete types_g;
  delete typelist_g;

  types_g    = nullptr;
  typelist_g = nullptr;
}

void
TypeDescription::register_type()
{
  {
    register_monitor register_write_lock{ Uintah::CrowdMonitor<TypeDescription::register_tag>::WRITER };

    if (!types_g) {
      ASSERT(!killed);
      ASSERT(!typelist_g)

      types_g    = scinew std::map<std::string, const TypeDescription*>;
      typelist_g = scinew std::vector<const TypeDescription*>;
    }

    auto iter = types_g->find(getName());
    if (iter == types_g->end()) {
      (*types_g)[getName()] = this;
    }
    typelist_g->push_back(this);
  }
}

std::string
TypeDescription::getName() const
{
  if (d_subtype) {
    return d_name + "<" + d_subtype->getName() + ">";
  } else {
    return d_name;
  }
}

std::string
TypeDescription::getFileName() const
{
  if (d_subtype) {
    return d_name + d_subtype->getFileName();
  } else {
    return d_name;
  }
}

const TypeDescription *
TypeDescription::lookupType( const std::string & t )
{
  {
    lookup_monitor lookup_read_lock{ Uintah::CrowdMonitor<TypeDescription::lookup_tag>::READER };

    if (!types_g) {
      return 0;
    }

    auto iter = types_g->find(t);
    if (iter == types_g->end()) {
      return 0;
    }

    return iter->second;
  }
}

MPI_Datatype
TypeDescription::getMPIType() const
{
  if (d_mpitype == MPI_Datatype(-1)) {
    // scope the lock_guard
    {
      std::lock_guard<Uintah::MasterLock> guard(get_mpi_type_lock);
      if (d_mpitype == MPI_Datatype(-1)) {
        if (d_mpitypemaker) {
          d_mpitype = (*d_mpitypemaker)();
        }
        else {
          throw InternalError( "MPI Datatype requested, but do not know how to make it", __FILE__, __LINE__ );
        }
      }
    }

  }

  ASSERT(d_mpitype != MPI_Datatype(-2));

  return d_mpitype;
}

Variable *
TypeDescription::createInstance() const
{
  if (!d_maker) {
    throw InternalError( "Do not know how to create instance for type: " + getName(), __FILE__, __LINE__ );
  }

  return (*d_maker)();
}

std::string
TypeDescription::toString( Type type )
{
  switch( type ) {
    case CCVariable:          return "CCVariable";
    case NCVariable:          return "NCVariable";
    case SFCXVariable:        return "SFCXVariable";
    case SFCYVariable:        return "SFCYVariable";
    case SFCZVariable:        return "SFCZVariable";
    case ParticleVariable:    return "ParticleVariable";
    case PerPatch:            return "PerPatch";
    case Point:               return "Point";
    case Vector:              return "Vector";
    case Matrix3:             return "Matrix3";
    case ReductionVariable:   return "ReductionVariable";
    case SoleVariable:        return "SoleVariable";
    case double_type:         return "double_type";
    case float_type:          return "float_type";
    case bool_type:           return "bool_type";
    case int_type:            return "int_type";
    case short_int_type:      return "short_int_type";
    case long_type:           return "long_type";
    case long64_type:         return "long64_type";
    case Short27:             return "Short27";
    case Stencil4:            return "Stencil4";
    case Stencil7:            return "Stencil7";
    case IntVector:           return "IntVector";
    case Unknown:             return "Unknown";
    case Other:               return "Other";
    default:
      std::stringstream msg;
      msg << "Invalid type: " << type;
      throw InternalError( msg.str(), __FILE__, __LINE__);
  }
}
