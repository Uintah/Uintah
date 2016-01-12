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

#include "VisItControlInterface_V2.h"

#include "visit_libsim.h"

#include <sci_defs/mpi_defs.h>
#include <sci_defs/visit_defs.h>

#include <CCA/Components/SimulationController/SimulationController.h>

#include <CCA/Components/OnTheFlyAnalysis/MinMax.h>
#include <Core/Grid/Material.h>
#define ALL_LEVELS 99

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>

static SCIRun::DebugStream visitdbg( "VisItLibSim", true );

namespace Uintah {

//---------------------------------------------------------------------
// GetOutputIntervals
//    Get the output checkpoints so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_GetOutputIntervals( visit_simulation_data *sim )
{
  if( !VisItIsConnected() ) //|| sim->isProc0 == false )
    return;
  
  sim->outputIntervals.clear();

  SimulationStateP simStateP = sim->simController->getSimulationStateP();
  Output *output = sim->simController->getOutput();
  
  // Add in the output and checkout intervals.
  if( output )
  {
    std::string name;
    double val;
    
    // Output interval based on time.
    if( output->getOutputInterval() > 0 )
    {
      name = simStateP->get_outputInterval_label()->getName();
      val = output->getOutputInterval();
    }
    // Output interval based on timestep.
    else
    {
      name = simStateP->get_outputTimestepInterval_label()->getName();
      val = output->getOutputTimestepInterval();
    }
    
    uintah_variable_data var;
    
    var.name = name;
    var.value = val;
    var.modified = false;
    sim->outputIntervals.push_back(var);
    
    // Checkpoint interval based on times.
    if( output->getCheckpointInterval() > 0 )
    {
      name = simStateP->get_checkpointInterval_label()->getName();
      val = output->getCheckpointInterval();
    }
    // Checkpoint interval based on timestep.
    else
    {
      name = simStateP->get_checkpointTimestepInterval_label()->getName();
      val = output->getCheckpointTimestepInterval();
    }

    var.name = name;
    var.value = val;
    var.modified = false;
    sim->outputIntervals.push_back(var);
  }
}


//---------------------------------------------------------------------
// getAnalysisVars
//    Get the Min/Max analysys vars. 
//---------------------------------------------------------------------
std::vector<MinMax::varProperties>
getAnalysisVars(ProblemSpecP d_prob_spec, SimulationStateP d_sharedState)
{
  static std::vector<MinMax::varProperties> d_analyzeVars;

  if( d_analyzeVars.size() )
    return d_analyzeVars;
  
  ProblemSpecP vars_ps = d_prob_spec->findBlock("Variables");

  // find the material to extract data from.  Default is matl 0.
  // The user can use either 
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>
  const Material* d_matl;
  if(d_prob_spec->findBlock("material") ){
    d_matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");
  } else if (d_prob_spec->findBlock("materialIndex") ){
    int indx;
    d_prob_spec->get("materialIndex", indx);
    d_matl = d_sharedState->getMaterial(indx);
  } else {
    d_matl = d_sharedState->getMaterial(0);
  }
  
  int defaultMatl = d_matl->getDWIndex();
  
  //__________________________________
  //  Now loop over all the variables to be analyzed  
    
  for (ProblemSpecP var_spec = vars_ps->findBlock("analyze");
       var_spec != 0; 
       var_spec = var_spec->findNextBlock("analyze")) {

    std::map< std::string, std::string > attribute;
    var_spec->getAttributes(attribute);
    
    //  Read in the optional material index from the variables that
    //  may be different from the default index and construct the
    //  material set
    int matl = defaultMatl;
    if (attribute["matl"].empty() == false) {
      matl = atoi(attribute["matl"].c_str());
    }

    // Read in the optional level index
    int level = ALL_LEVELS;
    if (attribute["level"].empty() == false) {
      level = atoi(attribute["level"].c_str());
    }
    
    // Read in the variable name
    std::string labelName = attribute["label"];
    VarLabel* label = VarLabel::find(labelName);
    
    const TypeDescription* td = label->typeDescription();
    const TypeDescription* subtype = td->getSubType();
    const int subType  = subtype->getType();
    
    //__________________________________
    //  Create the min and max VarLabels for the reduction variables
    
    std::string VLmax = labelName + "_max";
    std::string VLmin = labelName + "_min";
    VarLabel* meMax = NULL;
    VarLabel* meMin = NULL;
    
    // double
    if( subType == TypeDescription::double_type ) {
      meMax = VarLabel::create( VLmax, max_vartype::getTypeDescription() );
      meMin = VarLabel::create( VLmin, min_vartype::getTypeDescription() );
    }
    // Vectors
    else if( subType == TypeDescription::Vector ) {
      meMax = VarLabel::create( VLmax, maxvec_vartype::getTypeDescription() );
      meMin = VarLabel::create( VLmin, minvec_vartype::getTypeDescription() );
    }

    MinMax::varProperties me;
    me.label = label;
    me.matl = matl;
    me.level = level;
    me.reductionMaxLabel = meMax;
    me.reductionMinLabel = meMin;
    
    d_analyzeVars.push_back(me);
  }

  return d_analyzeVars;
}
    

//---------------------------------------------------------------------
// GetAnalysisVars
//    Get the min/max analysis vars so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_GetAnalysisVars( visit_simulation_data *sim )
{
  if( !VisItIsConnected() ) //|| sim->isProc0 == false )
    return;
  
  sim->minMaxVariables.clear();

  GridP gridP = sim->gridP;
  SimulationStateP simStateP = sim->simController->getSimulationStateP();
  ProblemSpecP d_ups = sim->simController->getProblemSpecP();
  DataWarehouse* newDW = sim->simController->getSchedulerP()->getLastDW();
  
  int numLevels = gridP->numLevels();
  
  // Add in the data analysis variable.
  ProblemSpecP da_ps = d_ups->findBlock("DataAnalysis");

  if( da_ps )
  {
    for(ProblemSpecP module_ps = da_ps->findBlock("Module");
        module_ps != 0;
        module_ps = module_ps->findNextBlock("Module"))
    {
      if( module_ps )
      {
        std::map<std::string,std::string> attributes;
        module_ps->getAttributes(attributes);
        
        // Add in the min max variables.
        if( attributes["name"] == "minMax" )
        {
          ProblemSpecP variables_ps = module_ps->findBlock("Variables");

          std::vector<Uintah::MinMax::varProperties> minMaxVars =
            getAnalysisVars(module_ps, simStateP);
          
          for( int i=0; i<minMaxVars.size(); ++i )
          {
            MinMax::varProperties minMaxVar = minMaxVars[i];

            double varMin = 0, varMax = 0;

            // Get level info
            for (int l=0; l<numLevels; ++l)
            {
              if( minMaxVar.level != ALL_LEVELS &&
                  minMaxVar.level != l )
                continue;
                 
              LevelP levelP = gridP->getLevel(l);
              Level *level = levelP.get_rep();

              // double
              if( minMaxVar.label->typeDescription()->getSubType()->getType() ==
                  TypeDescription::double_type )
              {
                min_vartype var_min;
                max_vartype var_max;
                
                newDW->get(var_min, minMaxVar.reductionMinLabel, level );
                newDW->get(var_max, minMaxVar.reductionMaxLabel, level );
                varMin = var_min;
                varMax = var_max;
              }
              
              // Vector
              else if( minMaxVar.label->typeDescription()->getSubType()->getType() ==
                       TypeDescription::Vector )
              {
                minvec_vartype var_min;
                maxvec_vartype var_max;
                
                newDW->get(var_min, minMaxVar.reductionMinLabel, level );
                newDW->get(var_max, minMaxVar.reductionMaxLabel, level );
                
                varMin = ((Vector) var_min).length();
                varMax = ((Vector) var_max).length();
              }

              uintah_min_max_data var;

              std::stringstream name;

              name << "L-" << l << "/"
                   << minMaxVar.label->getName()
                   << "/" << minMaxVar.matl;
              
              // var.name  = minMaxVar.label->getName();
              var.name  = name.str();
              var.matl  = minMaxVar.matl; // Ignored
              var.level = l;              // Ignored
              var.min   = varMin;
              var.max   = varMax;
              sim->minMaxVariables.push_back(var);
            }
          }
        }
      }
    }
  }
}

//---------------------------------------------------------------------
// GetUPSVars
//    Get the UPS vars that the user can modify in the Custon UI
//---------------------------------------------------------------------
void visit_GetUPSVars( visit_simulation_data *sim )
{
  if( !VisItIsConnected() ) //|| sim->isProc0 == false )
    return;
  
  sim->upsVariables.clear();
}

  
//---------------------------------------------------------------------
// UpdateUPSVars
//    Update the UPS that were changed in the Custon UI
//---------------------------------------------------------------------
void visit_UpdateUPSVars( visit_simulation_data *sim )
{
  // Loop through the user viewable/settable variables and update
  // those that have been modified.
  std::vector< uintah_variable_data > &vars = sim->upsVariables;

  for( int i=0; i<vars.size(); ++i )
  {
    uintah_variable_data &var = vars[i];
    
    if( var.modified )
    {
      // if( var.name == SOMEVAR )
      // {
      // }
    }
  }
}

} // End namespace Uintah
