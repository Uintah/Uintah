/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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


#include <Core/Grid/Variables/VarLabel.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/Patch.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Util/DOUT.hpp>

#include <iostream>
#include <map>
#include <sstream>

using namespace Uintah;

namespace {
  MasterLock g_label_mutex{};

  Dout g_varlabel_dbg( "VarLabel", "VarLabel", "report when a VarLabel is created and deleted", false );
}

//______________________________________________________________________
// Initialize class static variables:

std::string VarLabel::s_particle_position_name   = "p.x";
std::string VarLabel::s_default_compression_mode = "none";

std::map<std::string, VarLabel*> VarLabel::g_all_labels;

//______________________________________________________________________
//
VarLabel*
VarLabel::create( const std::string     & name
                , const TypeDescription * td
                , const IntVector       & boundaryLayer /* = IntVector(0,0,0) */
                ,       VarType           vartype       /* = Normal */
                )
{
  VarLabel* label = nullptr;

  g_label_mutex.lock();
  {
    auto iter = g_all_labels.find(name);
    if (iter != g_all_labels.end()) {
      // two labels with the same name -- make sure they are the same type
      VarLabel* dup = iter->second;
      if (boundaryLayer != dup->m_boundary_layer) {
        SCI_THROW(InternalError(std::string("Multiple VarLabels for " + dup->getName() + " defined with different # of boundary layers"), __FILE__, __LINE__));
      }

#if !defined(_AIX) && !defined(__APPLE__)
      // AIX uses lib.a's, therefore the "same" var labels are different...
      // Need to look into fixing this in a better way...
      // And I am not sure why we have to do this on the mac or windows...
      if (td != dup->m_td || vartype != dup->m_var_type) {
        SCI_THROW(InternalError(std::string("VarLabel with same name exists, '" + name + "', but with different type"), __FILE__, __LINE__));
      }
#endif

      label = dup;
    }
    else {
      label = scinew VarLabel(name, td, boundaryLayer, vartype);
      g_all_labels[name]=label;
      DOUT(g_varlabel_dbg, "Created VarLabel: "
           << std::setw(24) << label->m_name << "  "
           // << "address: " << std::setw(12) << label << "  "
           << "Type description: "
           << std::setw(30) << label->m_td->getName() << "  "
           << "Var type: "
           << std::setw(0) << (label->m_var_type ? "PositionVariable" :
                                                   "Normal"));
    }
    label->addReference();
  }
  g_label_mutex.unlock();

  return label;
}
//______________________________________________________________________
//
bool
VarLabel::destroy(const VarLabel* label)
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
      DOUT(g_varlabel_dbg, "Deleting VarLabel: " << label->m_name);
    }
    g_label_mutex.unlock();

    delete label;

    return true;
  }

  return false;
}
//______________________________________________________________________
//
VarLabel::VarLabel( const std::string             & name
                  , const Uintah::TypeDescription * td
                  , const IntVector               & boundaryLayer
                  ,       VarType                   vartype
                  )
  : m_name(name)
  , m_td(td)
  , m_boundary_layer(boundaryLayer)
  , m_var_type(vartype)
{
}
//______________________________________________________________________
//
void
VarLabel::printAll()
{
  for (auto & iter : g_all_labels) {
    proc0cout << "VarLabel: "
              << std::setw(24) << iter.second->m_name << "  "
              << "Type description: "
              << std::setw(30) << iter.second->m_td->getName() << "  "
              << "Var type: "
              << std::setw(0) << (iter.second->m_var_type ? "PositionVariable" :
                                                            "Normal")
              << std::endl;
  }
}
//______________________________________________________________________
//
VarLabel*
VarLabel::find( const std::string &  name )
{
  auto found = g_all_labels.find( name );

  if( found == g_all_labels.end() ) {
    return nullptr;
  }
  else {
    return found->second;
  }
}

//______________________________________________________________________
//    If the varLabel isn't found throw an exception
VarLabel*
VarLabel::find( const std::string & name,
                const std::string & mesg )
{
  VarLabel* label = VarLabel::find(name);

  if( label == nullptr ) {
    std::ostringstream warn;
    warn << mesg;
    warn << " --- Could not find the VarLabel (" << name << " ).";
    throw InternalError(warn.str(), __FILE__, __LINE__);
  }
  return label;
}

//______________________________________________________________________
//
VarLabel*
VarLabel::particlePositionLabel()
{
  return find(s_particle_position_name);
}


//______________________________________________________________________
//
std::string
VarLabel::getFullName( int matlIndex, const Patch * patch ) const
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


//______________________________________________________________________
//
void
VarLabel::schedReductionTask( bool input)
{
   if (!m_td->isReductionVariable()) {
     SCI_THROW(InternalError(std::string("Only reduction variables may set this flag.\n'" + m_name + "' is not a reduction variable."), __FILE__, __LINE__));
   }
   m_sched_reduction_task = input;
}

namespace Uintah {
std::ostream &
  operator<<( std::ostream & out, const Uintah::VarLabel & vl )
  {
    out << vl.getName();
    return out;
  }
}

