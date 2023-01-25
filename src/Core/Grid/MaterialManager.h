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

#ifndef UINTAH_CORE_GRID_MaterialManager_H
#define UINTAH_CORE_GRID_MaterialManager_H

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/RefCounted.h>
#include <Core/Util/DOUT.hpp>

#include <sci_defs/uintah_defs.h>

#include <map>
#include <vector>

namespace Uintah {

class VarLabel;
class Material; 
class SimpleMaterial;

/**************************************
      
    CLASS
      MaterialManager
      
      Short Description...
      
    GENERAL INFORMATION
      
      MaterialManager.h
      
      Steven G. Parker
      Department of Computer Science
      University of Utah
      
      Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
      
    KEYWORDS
      MaterialManager
      
    DESCRIPTION
      A data structure to manage materials
      
    WARNING
      
****************************************/

class MaterialManager : public RefCounted {

public:

  MaterialManager();
  ~MaterialManager();

  void clearMaterials();
  void finalizeMaterials();

  // These register and return materials from an app.
  void registerMaterial(std::string name, Material * mat);
  void registerMaterial(std::string name, Material * mat, unsigned int index);

  const MaterialSet* allMaterials( std::string name ) const;

  unsigned int getNumMatls( std::string name ) const;
  
  Material* getMaterial(std::string name, int idx) const;

  // These register return materials from all apps.
  void registerSimpleMaterial(SimpleMaterial*);
  
  const MaterialSet* allMaterials() const;

  unsigned int getNumMatls() const {
    return (unsigned int) m_all_matls.size();
  }

  Material* getMaterial(int idx) const {
    return m_all_matls[idx];
  }

  Material* getMaterialByName(const std::string& name) const;

  // 
  MaterialSubset* getAllInOneMatls() {
    return m_all_in_one_matls;
  }

  const MaterialSet* allOriginalMaterials() const;

  void setOriginalMatlsFromRestart(MaterialSet* matls);
  
  Material* parseAndLookupMaterial(ProblemSpecP& params,
                                   const std::string& name) const;

private:

  MaterialManager( const MaterialManager& );
  MaterialManager& operator=( const MaterialManager& );
      
  void registerMaterial( Material* );
  void registerMaterial( Material*, unsigned int index );

  // Named materials
  std::map<std::string, Material*> m_named_matls;

  // All simple materials
  std::vector<SimpleMaterial*> m_simple_matls;

  // All materials from all apps
  std::vector<Material*> m_all_matls;
  MaterialSet *          m_all_matl_set{nullptr};

  // Materials from each app
  std::map< std::string, std::vector<Material*> > m_app_matls;
  std::map< std::string, MaterialSet* >           m_app_matl_set;
  
  // The switcher needs to clear the materials, but don't 
  // delete them or there mihgt be VarLabel problems when 
  // CMs are destroyed.  Store them here until the end.
  std::vector<Material*> m_old_matls;

  // Keep track of all the original materials if switching
  MaterialSet    * m_all_orig_matls{nullptr};
  MaterialSubset * m_all_in_one_matls{nullptr};

  static int count;
};

} // End namespace Uintah

#endif
