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

#include <Core/Grid/BoundaryConditions/BCData.h>
#include <Core/Grid/BoundaryConditions/BoundCondBase.h>
#include <Core/Util/DebugStream.h>

#include <algorithm>
#include <iostream>
#include <typeinfo> // for typeid

using namespace Uintah;

namespace {
  DebugStream BCData_dbg("BCDATA_DBG", "Grid_BoundaryConditions", "", false);
}


BCData::BCData() 
{
}

BCData::~BCData()
{
  std::vector<BoundCondBase*>::const_iterator itr;
  for (itr = d_BCData.begin(); itr != d_BCData.end(); itr++) {
    delete (*itr);
  }
  d_BCData.clear();
}

BCData::BCData(const BCData& rhs)
{
  std::vector<BoundCondBase*>::const_iterator i;
  for (i = rhs.d_BCData.begin(); i != rhs.d_BCData.end(); ++i) {
    d_BCData.push_back((*i)->clone());
  }
}

BCData& BCData::operator=(const BCData& rhs)
{
  if (this == &rhs)
    return *this;

  // Delete the lhs
  std::vector<BoundCondBase*>::const_iterator itr;
  for (itr = d_BCData.begin(); itr != d_BCData.end(); ++itr) 
    delete (*itr);

  d_BCData.clear();
  // Copy the rhs to the lhs
  
  std::vector<BoundCondBase*>::const_iterator i;
  for (i = rhs.d_BCData.begin(); i != rhs.d_BCData.end(); ++i) {
    d_BCData.push_back((*i)->clone());
  }
    
  return *this;
}

bool BCData::operator==(const BCData& rhs)
{
  if (d_BCData.size() != rhs.d_BCData.size())
    return false;

  std::vector<BoundCondBase*>::const_iterator itr;
  for (itr = d_BCData.begin(); itr != d_BCData.end();) {
    if (rhs.find((*itr)->getBCVariable()) == false)
      return false;
    else
      ++itr;
  }
  return true;
}

bool BCData::operator<(const BCData& rhs) const
{
  std::vector<BoundCondBase*>::const_iterator itr;
  if (d_BCData.size() < rhs.d_BCData.size())
    return true;
  else 
    return false;
}

void 
BCData::setBCValues(BoundCondBase* bc)
{
  if (!find(bc->getBCVariable()))
    d_BCData.push_back(bc->clone());
}

const BoundCondBase*
BCData::getBCValues(const std::string& var_name) const
{
  // The default location for BCs defined for all materials is mat_id = -1.
  // Need to first check the actual mat_id specified.  If this is not found,
  // then will check mat_id = -1 case.  If it isn't found, then return 0.
  std::vector<BoundCondBase*>::const_iterator itr;
  for (itr = d_BCData.begin(); itr != d_BCData.end(); ++itr) {
    if ((*itr)->getBCVariable() == var_name)
      return (*itr);
  }
  return 0;
}


const BoundCondBase*
BCData::cloneBCValues(const std::string& var_name) const
{
  // The default location for BCs defined for all materials is mat_id = -1.
  // Need to first check the actual mat_id specified.  If this is not found,
  // then will check mat_id = -1 case.  If it isn't found, then return 0.
  std::vector<BoundCondBase*>::const_iterator itr;
  for (itr = d_BCData.begin(); itr != d_BCData.end(); ++itr) {
    if ((*itr)->getBCVariable() == var_name)
      return (*itr)->clone();
  }
  return 0;

}

const std::vector<BoundCondBase*>&
BCData::getBCData() const
{
  return d_BCData;
}

bool BCData::find(const std::string& var_name) const
{
  std::vector<BoundCondBase*>::const_iterator itr;

  for (itr = d_BCData.begin(); itr != d_BCData.end(); ++itr) {
    if ((*itr)->getBCVariable() == var_name) {
      return true;
    }
  }
  return false;
}

bool BCData::find(const std::string& bc_type, const std::string& bc_variable) const
{
  const BoundCondBase* bc = cloneBCValues(bc_variable);

  if (bc) {
    if (bc->getBCType() == bc_type) {
      delete bc;
      return true;
    }
  }
  delete bc;
  return false; 
      
}

void BCData::combine(BCData& from)
{
  std::vector<BoundCondBase*>::const_iterator itr;
  for (itr = from.d_BCData.begin(); itr != from.d_BCData.end(); ++itr) {
    // std::cerr << "bc = " << itr->first << " address = " << itr->second << std::endl;
    setBCValues(*itr);
  }

}


void BCData::print() const
{
  std::vector<BoundCondBase*>::const_iterator itr;
  BCData_dbg << "size of d_BCData = " << d_BCData.size() << std::endl;
  for (itr = d_BCData.begin(); itr != d_BCData.end(); itr++) {
    BCData_dbg << "BC = " << (*itr)->getBCVariable() << " type = " 
               << (*itr)->getBCType() << std::endl;
  }
  
}
