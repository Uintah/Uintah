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


#ifndef UINTAH_HOMEBREW_VarLabelMatl_H
#define UINTAH_HOMEBREW_VarLabelMatl_H

#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>


namespace Uintah {


/**************************************

  struct
    VarLabelMatl

    VarLabel, Material, and Domain


  GENERAL INFORMATION

    VarLabelMatl.h

    Wayne Witzel
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


  KEYWORDS
    VarLabel, Material, Patch

  DESCRIPTION

    Specifies a VarLabel on a specific material and a specific
    patch or level and optionally with another domain (data
    warehouse) with an operator< defined so this can be used as a
    key in a map.


****************************************/

template<class DomainType0, class DomainType1 = void>
struct VarLabelMatl
{

  VarLabelMatl( const VarLabel    * label
              ,       int           matlIndex
              , const DomainType0 * domain0
              , const DomainType1 * domain1
              )
    : m_label{label}
    , m_matl_index{matlIndex}
    , m_domain_0{domain0}
    , m_domain_1{domain1}
  {}

  VarLabelMatl( const VarLabelMatl<DomainType0, DomainType1> & copy )
    : m_label{copy.m_label}
    , m_matl_index{copy.m_matl_index}
    , m_domain_0{copy.m_domain_0}
    , m_domain_1{copy.m_domain_1}
  {}

  VarLabelMatl<DomainType0, DomainType1>& operator=( const VarLabelMatl<DomainType0, DomainType1>& copy )
  {
    m_label     = copy.m_label;
    m_matl_index= copy.m_matl_index;
    m_domain_0  = copy.m_domain_0;
    m_domain_1  = copy.m_domain_1;

    return *this;
  }
  
  bool operator<( const VarLabelMatl<DomainType0, DomainType1>& other ) const
  {
    if (m_label->equals(other.m_label)) {
      if (m_matl_index == other.m_matl_index) {
        if (m_domain_0 == other.m_domain_0) {
          return m_domain_1 < other.m_domain_1;
        }
        else {
          return m_domain_0 < other.m_domain_0;
        }
      }
      else {
        return m_matl_index < other.m_matl_index;
      }
    }
    else {
      VarLabel::Compare comp;
      return comp(m_label, other.m_label);
    }
  }
  
  bool operator==( const VarLabelMatl<DomainType0, DomainType1>& other ) const
  {
    return ((m_label->equals(other.m_label)) && (m_matl_index == other.m_matl_index) && (m_domain_0 == other.m_domain_0) && (m_domain_1 == other.m_domain_1));
  }

  const VarLabel    * m_label{nullptr};
        int           m_matl_index{};
  const DomainType0 * m_domain_0{nullptr};
  const DomainType1 * m_domain_1{nullptr};
};  


template<class DomainType>
struct VarLabelMatl<DomainType, void>
{

  VarLabelMatl( const VarLabel   * label
              ,       int          matlIndex
              , const DomainType * domain
              )
    : m_label{label}
    , m_matl_index{matlIndex}
    , m_domain{domain}
  {}

  VarLabelMatl( const VarLabelMatl<DomainType>& copy )
    : m_label(copy.m_label), m_matl_index(copy.m_matl_index), m_domain(copy.m_domain)
  {}

  VarLabelMatl<DomainType>& operator=( const VarLabelMatl<DomainType>& copy )
  {
    m_label      = copy.m_label;
    m_matl_index = copy.m_matl_index;
    m_domain     = copy.m_domain;
    return *this;
  }
  
  bool operator<( const VarLabelMatl<DomainType>& other ) const
  {
    if (m_label->equals(other.m_label)) {
      if (m_matl_index == other.m_matl_index) {
        return m_domain < other.m_domain;
      }
      else {
        return m_matl_index < other.m_matl_index;
      }
    }
    else {
      VarLabel::Compare comp;
      return comp(m_label, other.m_label);
    }
  }
  
  bool operator==( const VarLabelMatl<DomainType>& other ) const
  {
    return ((m_label->equals(other.m_label)) && (m_matl_index == other.m_matl_index) && (m_domain == other.m_domain));
  }

  const VarLabel   * m_label{nullptr};
        int          m_matl_index{};
  const DomainType * m_domain{nullptr};
};  


} // End namespace Uintah

#endif
