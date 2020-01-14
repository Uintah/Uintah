/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef UINTAH_VISIT_LIBSIM_DATABASE_H
#define UINTAH_VISIT_LIBSIM_DATABASE_H

#include "VisItDataInterface_V2.h"

#include <set>

/**************************************
        
CLASS
   visit_libsim_database
        
   Short description...
        
GENERAL INFORMATION
        
   visit_init
        
   Allen R. Sanderson
   Scientific Computing and Imaging Institute
   University of Utah
        
KEYWORDS
   VisIt, libsim, in-situ
        
DESCRIPTION
   Long description...
        
WARNING
        

****************************************/

namespace Uintah {

void visit_SimGetCustomUIData    (void *cbdata);
visit_handle visit_SimGetMetaData(void *cbdata);
visit_handle visit_SimGetMesh    (int domain, const char *name, void *cbdata);
visit_handle visit_SimGetVariable(int domain, const char *name, void *cbdata);
visit_handle visit_SimGetDomainList      (const char *name, void *cbdata);
visit_handle visit_SimGetDomainBoundaries(const char *name, void *cbdata);
visit_handle visit_SimGetDomainNesting   (const char *name, void *cbdata);

void addRectilinearMesh( visit_handle md, std::set<std::string> &meshes_added,
                         std::string meshName, visit_simulation_data *sim );

void addParticleMesh( visit_handle md, std::set<std::string> &meshes_added,
                      std::string meshName, visit_simulation_data *sim );

void addMeshNodeRankSIL( visit_handle md, std::string mesh,
                         visit_simulation_data *sim );
  
void addMeshVariable( visit_handle md, std::set<std::string> &mesh_vars_added,
                      std::string varName, std::string varType,
                      std::string meshName, VisIt_VarCentering cent );

void nameCleanup( std::string &str );
  
template< class ENUM, class T >
void addReductionStats( visit_handle md, ReductionInfoMapper< ENUM, T > stats,
                        std::string statName, std::string meshName,
                        std::string meshLayout )
{
  // There is performance on a per rank and per node basis.
  const unsigned int nProcLevels = 3;
  std::string proc_level[nProcLevels] = {"/Rank", "/Node/Average", "/Node/Sum"};

  for( unsigned j=0; j<nProcLevels; ++j )
  {
    for( unsigned int i=0; i<stats.size(); ++i )
    {
      visit_handle vmd = VISIT_INVALID_HANDLE;
      
      if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
      {
        std::string tmp_name = statName + 
          stats.getName( i ) + proc_level[j];
        
        tmp_name += meshLayout;

        std::string units = stats.getUnits( i );
        
        nameCleanup( tmp_name );

        VisIt_VariableMetaData_setName(vmd, tmp_name.c_str());
        VisIt_VariableMetaData_setMeshName(vmd, meshName.c_str());
        VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);
        VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
        VisIt_VariableMetaData_setNumComponents(vmd, 1);
        VisIt_VariableMetaData_setUnits(vmd, units.c_str());
      
        // ARS - FIXME
        //      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
        VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
        
        VisIt_SimulationMetaData_addVariable(md, vmd);
      }
    }
  }
}

template< class ENUM, class T >
void addVectorStats( visit_handle md, VectorInfoMapper< ENUM, T > stats,
                     std::string statName, std::string meshName )
{
  if(stats.size() == 0 )
    return;
  
  for( unsigned int i=0; i<stats[0].size(); ++i )
  {
    visit_handle vmd = VISIT_INVALID_HANDLE;
    
    if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
    {
      std::string tmp_name = statName + stats[0].getName( i );
      std::string units = stats[0].getUnits( i );

      nameCleanup( tmp_name );
      
      VisIt_VariableMetaData_setName(vmd, tmp_name.c_str());
      VisIt_VariableMetaData_setMeshName(vmd, meshName.c_str());
      VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);
      VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
      VisIt_VariableMetaData_setNumComponents(vmd, 1);
      VisIt_VariableMetaData_setUnits(vmd, units.c_str());
      
      // ARS - FIXME
      //      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
      VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
      
      VisIt_SimulationMetaData_addVariable(md, vmd);
    }
  }
}

template< class ENUM, class T >
void addVectorReductionStats( visit_handle md, VectorInfoMapper< ENUM, T > stats,
                              std::string statName, std::string meshName )
{
  if(stats.size() == 0 )
    return;

  const unsigned int nReductions = 6;
  std::string reduction[nReductions] =
    {"/Size", "/Sum", "/Average", "/Minimum", "/Maximum", "/StdDev"};

  for( unsigned j=0; j<nReductions; ++j )
  {
    if( (j == 0) ||
        (j == 1 && stats.calculateSum()) ||
        (j == 2 && stats.calculateAverage()) ||
        (j == 3 && stats.calculateMinimum()) ||
        (j == 4 && stats.calculateMaximum()) ||
        (j == 5 && stats.calculateStdDev() ) )
    {  
      for( unsigned int i=0; i<stats[0].size(); ++i )
      {
        visit_handle vmd = VISIT_INVALID_HANDLE;
        
        if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
        {
          std::string tmp_name =
            statName + stats[0].getName( i ) + reduction[j];

          std::string units = (j ? stats[0].getUnits( i ) : "" );
      
          nameCleanup( tmp_name );
          
          VisIt_VariableMetaData_setName(vmd, tmp_name.c_str());
          VisIt_VariableMetaData_setMeshName(vmd, meshName.c_str());
          VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);
          VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
          VisIt_VariableMetaData_setNumComponents(vmd, 1);
          VisIt_VariableMetaData_setUnits(vmd, units.c_str());
      
          // ARS - FIXME
          //      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
          VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
      
          VisIt_SimulationMetaData_addVariable(md, vmd);
        }
      }
    }
  }
}

template< class KEY, class ENUM, class T >
void addMapAllStats( const visit_handle md, MapInfoMapper< KEY, ENUM, T > stats,
                     std::string statName, std::string statIndex,
                     std::string meshName )
{
  if( stats.size() == 0 )
    return;
  
  KEY key = stats.getKey(0);

  for( unsigned int i=0; i<stats[key].size(); ++i )
  {
    visit_handle vmd = VISIT_INVALID_HANDLE;
    
    if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
    {
      std::string tmp_name = statName + stats[key].getName( i );

      if( statIndex.size() )
        tmp_name += statIndex;

      std::string units = stats[key].getUnits( i );
      
      nameCleanup( tmp_name );
      
      VisIt_VariableMetaData_setName(vmd, tmp_name.c_str());
      VisIt_VariableMetaData_setMeshName(vmd, meshName.c_str());
      VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);
      VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
      VisIt_VariableMetaData_setNumComponents(vmd, 1);
      VisIt_VariableMetaData_setUnits(vmd, units.c_str());
      
      // ARS - FIXME
      //      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
      VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
      
      VisIt_SimulationMetaData_addVariable(md, vmd);
    }
  }
}

template< class KEY, class ENUM, class T >
void addMapIndividualStats( const visit_handle md, MapInfoMapper< KEY, ENUM, T > stats,
                            std::string statName, std::string statIndex,
                            std::string meshName )
{
  if( stats.size() == 0 )
    return;
  
  for( unsigned int k=0; k<stats.size(); ++k )
  {
    KEY key = stats.getKey(k);

    std::ostringstream keyName;
    keyName << key;
    
    for( unsigned int i=0; i<stats[key].size(); ++i )
    {
      visit_handle vmd = VISIT_INVALID_HANDLE;
      
      if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
      {
        std::string tmp_name = statName +
          keyName.str() + "/" + stats[key].getName( i );
        
        if( statIndex.size() )
          tmp_name += statIndex;

        std::string units = stats[key].getUnits( i );
        
        nameCleanup( tmp_name );
        
        VisIt_VariableMetaData_setName(vmd, tmp_name.c_str());
        VisIt_VariableMetaData_setMeshName(vmd, meshName.c_str());
        VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);
        VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
        VisIt_VariableMetaData_setNumComponents(vmd, 1);
        VisIt_VariableMetaData_setUnits(vmd, units.c_str());
        
        // ARS - FIXME
        //      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
        VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
        
        VisIt_SimulationMetaData_addVariable(md, vmd);
      }
    }
  }
}

template< class KEY, class ENUM, class T >
void addMapReductionStats( visit_handle md, MapInfoMapper< KEY, ENUM, T > stats,
                           std::string statName, std::string statIndex,
                           std::string meshName )
{
  if( stats.size() == 0 )
    return;
  
  KEY key = stats.getKey(0);

  const unsigned int nReductions = 6;
  std::string reduction[nReductions] =
    {"/Size", "/Sum", "/Average", "/Minimum", "/Maximum", "/StdDev"};

  for( unsigned j=0; j<nReductions; ++j )
  {
    if( (j == 0) ||
        (j == 1 && stats.calculateSum()) ||
        (j == 2 && stats.calculateAverage()) ||
        (j == 3 && stats.calculateMinimum()) ||
        (j == 4 && stats.calculateMaximum()) ||
        (j == 5 && stats.calculateStdDev() ) )
    {  
      for( unsigned int i=0; i<stats[key].size(); ++i )
      {
        visit_handle vmd = VISIT_INVALID_HANDLE;
        
        if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
        {
          std::string tmp_name =
            statName + stats[key].getName( i ) + reduction[j];

          if( statIndex.size() )
            tmp_name += statIndex;

          std::string units = (j ? stats[key].getUnits( i ) : "" );

          nameCleanup( tmp_name );
          // std::cerr << __FUNCTION__ << "  " << __LINE__ << "  " << tmp_name << std::endl;
          
          VisIt_VariableMetaData_setName(vmd, tmp_name.c_str());
          VisIt_VariableMetaData_setMeshName(vmd, meshName.c_str());
          VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);
          VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
          VisIt_VariableMetaData_setNumComponents(vmd, 1);
          VisIt_VariableMetaData_setUnits(vmd, units.c_str());
      
          // ARS - FIXME
          //      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
          VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
      
          VisIt_SimulationMetaData_addVariable(md, vmd);
        }
      }
    }
  }
}

} // End namespace Uintah

#endif
