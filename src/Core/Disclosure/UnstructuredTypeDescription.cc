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

#include <Core/Disclosure/UnstructuredTypeDescription.h>
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

using register_monitor = Uintah::CrowdMonitor<UnstructuredTypeDescription::register_tag>;
using lookup_monitor   = Uintah::CrowdMonitor<UnstructuredTypeDescription::lookup_tag>;

Uintah::MasterLock get_mpi_type_lock{};
std::map<std::string, const UnstructuredTypeDescription*> * types_g     = nullptr;
std::vector<const UnstructuredTypeDescription*>           * typelist_g  = nullptr;
bool killed = false;

}


UnstructuredTypeDescription::UnstructuredTypeDescription(        UnstructuredType          type
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

UnstructuredTypeDescription::UnstructuredTypeDescription(        UnstructuredType          type
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

UnstructuredTypeDescription::UnstructuredTypeDescription(        UnstructuredType              type
                                ,  const std::string     & name
                                ,        UnstructuredVariable        * (*maker)()
                                ,  const UnstructuredTypeDescription * subtype
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
UnstructuredTypeDescription::deleteAll()
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
UnstructuredTypeDescription::register_type()
{
  {
    register_monitor register_write_lock{ Uintah::CrowdMonitor<UnstructuredTypeDescription::register_tag>::WRITER };

    if (!types_g) {
      ASSERT(!killed);
      ASSERT(!typelist_g)

      types_g    = scinew std::map<std::string, const UnstructuredTypeDescription*>;
      typelist_g = scinew std::vector<const UnstructuredTypeDescription*>;
    }

    auto iter = types_g->find(getName());
    if (iter == types_g->end()) {
      (*types_g)[getName()] = this;
    }
    typelist_g->push_back(this);
  }
}

std::string
UnstructuredTypeDescription::getName() const
{
  if (d_subtype) {
    return d_name + "<" + d_subtype->getName() + ">";
  } else {
    return d_name;
  }
}

std::string
UnstructuredTypeDescription::getFileName() const
{
  if (d_subtype) {
    return d_name + d_subtype->getFileName();
  } else {
    return d_name;
  }
}

const UnstructuredTypeDescription *
UnstructuredTypeDescription::lookupType( const std::string & t )
{
  {
    lookup_monitor lookup_read_lock{ Uintah::CrowdMonitor<UnstructuredTypeDescription::lookup_tag>::READER };

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
UnstructuredTypeDescription::getMPIType() const
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

UnstructuredVariable *
UnstructuredTypeDescription::createInstance() const
{
  if (!d_maker) {
    throw InternalError( "Do not know how to create instance for type: " + getName(), __FILE__, __LINE__ );
  }

  return (*d_maker)();
}

std::string
UnstructuredTypeDescription::toString( UnstructuredType type )
{
  switch( type ) {
    case UnstructuredCCVariable:          return "UnstructuredCCVariable";
    case UnstructuredNCVariable:          return "UnstructuredNCVariable";
    case UnstructuredSFCXVariable:        return "UnstructuredSFCXVariable";
    case UnstructuredSFCYVariable:        return "UnstructuredSFCYVariable";
    case UnstructuredSFCZVariable:        return "UnstructuredSFCZVariable";
    case ParticleVariable:    return "ParticleVariable";
    case UnstructuredParticleVariable:    return "UnstructuredParticleVariable";
    case UnstructuredPerPatch:            return "UnstructuredPerPatch";
    case Point:               return "Point";
    case Vector:              return "Vector";
    case Matrix3:             return "Matrix3";
    case UnstructuredReductionVariable:   return "UnstructuredReductionVariable";
    case UnstructuredSoleVariable:        return "SoleVariable";
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
