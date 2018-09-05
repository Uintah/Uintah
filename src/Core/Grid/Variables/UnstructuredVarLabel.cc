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


#include <Core/Grid/Variables/UnstructuredVarLabel.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/UnstructuredPatch.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Util/DOUT.hpp>

#include <iostream>
#include <map>
#include <sstream>

using namespace Uintah;

namespace {
  MasterLock g_label_mutex{};

  Dout g_varlabel_dbg( "UnstructuredVarLabel", "UnstructuredVarLabel", "report when a UnstructuredVarLabel is created and deleted", false );
}

//______________________________________________________________________
// Initialize class static variables:

std::string UnstructuredVarLabel::s_particle_position_name   = "p.x";
std::string UnstructuredVarLabel::s_default_compression_mode = "none";

std::map<std::string, UnstructuredVarLabel*> UnstructuredVarLabel::g_all_labels;

//______________________________________________________________________
//

UnstructuredVarLabel*
UnstructuredVarLabel::create( const std::string     & name
                , const UnstructuredTypeDescription * td
                , const IntVector       & boundaryLayer /* = IntVector(0,0,0) */
                ,       VarType           vartype       /* = Normal */
                )
{
  UnstructuredVarLabel* label = nullptr;

  g_label_mutex.lock();
  {
    auto iter = g_all_labels.find(name);
    if (iter != g_all_labels.end()) {
      // two labels with the same name -- make sure they are the same type
      UnstructuredVarLabel* dup = iter->second;
      if (boundaryLayer != dup->m_boundary_layer) {
        SCI_THROW(InternalError(std::string("Multiple UnstructuredVarLabels for " + dup->getName() + " defined with different # of boundary layers"), __FILE__, __LINE__));
      }
    
#if !defined(_AIX) && !defined(__APPLE__)
      // AIX uses lib.a's, therefore the "same" var labels are different...
      // Need to look into fixing this in a better way...
      // And I am not sure why we have to do this on the mac or windows...
      if (td != dup->m_td || vartype != dup->m_var_type) {
        SCI_THROW(InternalError(std::string("UnstructuredVarLabel with same name exists, '" + name + "', but with different type"), __FILE__, __LINE__));
      }
#endif

      label = dup;
    }
    else {
      label = scinew UnstructuredVarLabel(name, td, boundaryLayer, vartype);
      g_all_labels[name]=label;
      DOUT(g_varlabel_dbg, "Created UnstructuredVarLabel: " << label->m_name << " [address = " << label);
    }
    label->addReference();
    }
  g_label_mutex.unlock(); 
  
  return label;
}

bool
UnstructuredVarLabel::destroy(const UnstructuredVarLabel* label)
{
  if (label == nullptr) {
    return false;
  }

  if (label->removeReference()) {
    g_label_mutex.lock();
    {
      auto iter = g_all_labels.find(label->m_name);
      if (iter != g_all_labels.end() && iter->second == label) {
        g_all_labels.erase(iter);
      }
      DOUT(g_varlabel_dbg, "Deleting UnstructuredVarLabel: " << label->m_name);
    }
    g_label_mutex.unlock();

    delete label;

    return true;
  }

  return false;
}

UnstructuredVarLabel::UnstructuredVarLabel( const std::string             & name
                  , const Uintah::UnstructuredTypeDescription * td
                  , const IntVector               & boundaryLayer
                  ,       VarType                   vartype
                  )
  : m_name(name)
  , m_td(td)
  , m_boundary_layer(boundaryLayer)
  , m_var_type(vartype)
{
}

void
UnstructuredVarLabel::printAll()
{
  auto iter = g_all_labels.begin();

  for (; iter != g_all_labels.end(); iter++) {
    std::cout << (*iter).second->m_name << std::endl;
  }
}

UnstructuredVarLabel*
UnstructuredVarLabel::find( const std::string &  name )
{
  auto found = g_all_labels.find( name );

   if( found == g_all_labels.end() ) {
      return nullptr;
   }
   else {
      return found->second;
   }
}

UnstructuredVarLabel*
UnstructuredVarLabel::particlePositionLabel()
{
  return find(s_particle_position_name);
}

std::string
UnstructuredVarLabel::getFullName( int matlIndex, const UnstructuredPatch * patch ) const
{
  std::ostringstream out;
  out << m_name << "(matl=" << matlIndex;

  if( patch ) {
    out << ", patch=" << patch->getID();
  }
  else {
    out << ", no patch";
  }
  out << ")";

  return out.str();
}                             

void
UnstructuredVarLabel::allowMultipleComputes()
{
   if (!m_td->isReductionVariable()) {
     SCI_THROW(InternalError(std::string("Only reduction variables may allow multiple computes.\n'" + m_name + "' is not a reduction variable."), __FILE__, __LINE__));
   }
   m_allow_multiple_computes = true;
}

namespace Uintah {
std::ostream &
  operator<<( std::ostream & out, const Uintah::UnstructuredVarLabel & vl )
  {
    out << vl.getName();
    return out;
  }
}

