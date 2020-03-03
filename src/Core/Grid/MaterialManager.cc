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

#include <Core/Grid/MaterialManager.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/MaterialSetP.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <sci_defs/uintah_defs.h>

using namespace Uintah;

int MaterialManager::count = 0;  

MaterialManager::MaterialManager()
{
  if (count++ >= 1) {
    throw ProblemSetupException("Allocated multiple MaterialManagers", __FILE__, __LINE__);
  }
}

//__________________________________
//
MaterialManager::~MaterialManager()
{
  clearMaterials();

  for (unsigned i = 0; i < m_old_matls.size(); i++){
    delete m_old_matls[i];
  }

  if(m_all_orig_matls && m_all_orig_matls->removeReference()){
    delete m_all_orig_matls;
  }
}

//__________________________________
//
void
MaterialManager::registerMaterial( Material* matl )
{
  matl->setDWIndex((int)m_all_matls.size());      

  m_all_matls.push_back(matl);                    

  if(matl->hasName()) {                    
    m_named_matls[matl->getName()] = matl;
  }
}

//__________________________________
//
void
MaterialManager::registerMaterial( Material* matl, unsigned int index )
{
  matl->setDWIndex(index);

  if (m_all_matls.size() <= index) {
    m_all_matls.resize(index + 1);
  }

  m_all_matls[index] = matl;

  if (matl->hasName()) {
    m_named_matls[matl->getName()] = matl;
  }
}

//__________________________________
//
void
MaterialManager::registerSimpleMaterial( SimpleMaterial* matl )
{
  m_simple_matls.push_back(matl);
  registerMaterial(matl);
}

//__________________________________
//
const MaterialSet*
MaterialManager::allMaterials() const
{
  ASSERT(m_all_matl_set != nullptr);
  return m_all_matl_set;
}

//__________________________________
//
const MaterialSet*
MaterialManager::allOriginalMaterials() const
{
  ASSERT(m_all_orig_matls != nullptr);
  return m_all_orig_matls;
}

//__________________________________
//
void
MaterialManager::setOriginalMatlsFromRestart( MaterialSet* matls )
{
  if (m_all_orig_matls && m_all_orig_matls->removeReference())
    delete m_all_orig_matls;
  m_all_orig_matls = matls;
}

//__________________________________
//
Material*
MaterialManager::getMaterialByName( const std::string& name ) const
{
  std::map<std::string, Material*>::const_iterator iter =
    m_named_matls.find(name);
  
  if(iter == m_named_matls.end()){
    return nullptr;
  }
  return iter->second;
}

//__________________________________
//
Material*
MaterialManager::parseAndLookupMaterial( ProblemSpecP& params,
                                         const std::string& name ) const
{
  // for single material problems return matl 0
  Material* result = getMaterial(0);

  if (getNumMatls() > 1) {
    std::string matlname;
    if (!params->get(name, matlname)) {
      throw ProblemSetupException("Cannot find material section", __FILE__, __LINE__);
    }

    result = getMaterialByName(matlname);
    if (!result) {
      throw ProblemSetupException("Cannot find a material named:" + matlname, __FILE__, __LINE__);
    }
  }
  return result;
}

//__________________________________
// These register and return materials from an app.

void
MaterialManager::registerMaterial( std::string name, Material* matl )
{
  m_app_matls[name].push_back(matl);
  registerMaterial(matl);
}

void
MaterialManager::registerMaterial( std::string name, Material* matl,
                                      unsigned int index )
{
  m_app_matls[name].push_back(matl);
  registerMaterial(matl, index);
}

const MaterialSet*
MaterialManager::allMaterials( std::string name ) const
{
  if( m_app_matl_set.find(name) != m_app_matl_set.end() )
    return m_app_matl_set.at(name);
  else
    return nullptr;
}

unsigned int
MaterialManager::getNumMatls( std::string name ) const
{
  if( m_app_matls.find(name) != m_app_matls.end() )
    return (unsigned int) m_app_matls.at(name).size();
  else
    return 0;
}
  
Material*
MaterialManager::getMaterial(std::string name, int idx) const
{
  if( m_app_matls.find(name) != m_app_matls.end() )
    return m_app_matls.at(name)[idx];
  else
    return nullptr;
}

//__________________________________
//
void
MaterialManager::finalizeMaterials()
{
  // All Matls
  if (m_all_matl_set && m_all_matl_set->removeReference()) {
    delete m_all_matl_set;
  }
  m_all_matl_set = scinew MaterialSet();
  m_all_matl_set->addReference();
  std::vector<int> tmp_matls(m_all_matls.size());

  for (size_t i = 0; i < m_all_matls.size(); i++) {
    tmp_matls[i] = m_all_matls[i]->getDWIndex();
  }
  m_all_matl_set->addAll(tmp_matls);

  // All Original Matls
  if (m_all_orig_matls == nullptr) {
    m_all_orig_matls = scinew MaterialSet();
    m_all_orig_matls->addReference();
    m_all_orig_matls->addAll(tmp_matls);
  }

  // All In One Matls
  if (m_all_in_one_matls && m_all_in_one_matls->removeReference()) {
    delete m_all_in_one_matls;
  }
  m_all_in_one_matls = scinew MaterialSubset();
  m_all_in_one_matls->addReference();
  // a material that represents all materials
  // (i.e. summed over all materials -- the whole enchilada)
  m_all_in_one_matls->add((int)m_all_matls.size());
  
  for ( auto & var : m_app_matl_set )
  {
    if (var.second && var.second->removeReference()) {
      delete var.second;
    }
  }
  
  for ( auto & var : m_app_matls )
  {
    std::string name = var.first;

    m_app_matl_set[name] = scinew MaterialSet();
    m_app_matl_set[name]->addReference();

    std::vector<int> tmp_app_matls(m_app_matls[name].size());

    for (size_t i = 0; i < m_app_matls[name].size(); i++) {
      tmp_app_matls[i] = m_app_matls[name][i]->getDWIndex();
    }

    m_app_matl_set[name]->addAll(tmp_app_matls);
  }
}

//__________________________________
//
void
MaterialManager::clearMaterials()
{
  for (size_t i = 0; i < m_all_matls.size(); i++){
    m_old_matls.push_back(m_all_matls[i]);
  }

  if(m_all_matl_set && m_all_matl_set->removeReference()){
    delete m_all_matl_set;
  }
  m_all_matl_set = nullptr;
  
  if (m_all_in_one_matls && m_all_in_one_matls->removeReference()) {
    delete m_all_in_one_matls;
  }
  m_all_in_one_matls = nullptr;

  m_all_matls.clear();
  m_named_matls.clear();
  m_simple_matls.clear();

  for ( auto & var : m_app_matls )
  {
    std::string name = var.first;

    if (m_app_matl_set[name] && m_app_matl_set[name]->removeReference()) {
      delete m_app_matl_set[name];
    }
    
    m_app_matls[name].clear();
    
    m_app_matl_set[name] = nullptr;
  }
}
