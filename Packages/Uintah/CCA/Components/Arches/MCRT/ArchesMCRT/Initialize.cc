// ****************************************************************************
// Actual read
// ****************************************************************************
void
Arches::radInitialCondition(const ProcessorGroup* ,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse* ,
                            DataWarehouse* new_dw)
{

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    
    // initialize control volume (media)'s radiative properties
    CCVariable<double> pressure;
    CCVariable<double> scalar;
    CCVariable<double> temperature;
    CCVariable<double> absorbCoeff; // absorption coefficient, 1/m
    CCVariable<double> scatterCoeff; // scattering coefficient, 1/m

    // initialize surface ( boundary)'s radiative properties
    // when transmittance tau = 0; we have rho + absorbCoeff = 1;
    // and rho = rhoReflect + rhoDiffuse;
    // emiss = absorbCoeff;
    SFCXVariable<double> temperatureX;
    SFCXVariable<double> absorbCoeffX;
    SFCXVariable<double> rhoSpecularX;  // specular reflection
    SFCXVariable<double> rhoDiffuseX; // diffusive reflection
    SFCXVariable<double> emissX;

    SFCYVariable<double> temperatureY;
    SFCYVariable<double> absorbCoeffY;
    SFCYVariable<double> rhoSpecularY;
    SFCYVariable<double> rhoDiffuseY;
    SFCYVariable<double> emissY;

    SFCZVariable<double> temperatureZ;
    SFCZVariable<double> absorbCoeffZ;
    SFCZVariable<double> rhoSpecularZ;
    SFCZVariable<double> rhoDiffuseZ;
    SFCZVariable<double> emissZ;    


    // whats the label for?
    
    // d_lab from ArchesLabel.cc 
      // for radiation
      const VarLabel* d_fvtfiveINLabel;
      const VarLabel* d_tfourINLabel;
      const VarLabel* d_tfiveINLabel;
      const VarLabel* d_tnineINLabel;
      const VarLabel* d_qrgINLabel;
      const VarLabel* d_qrsINLabel;
      const VarLabel* d_absorpINLabel;
      const VarLabel* d_sootFVINLabel;
      const VarLabel* d_abskgINLabel;
      const VarLabel* d_radiationSRCINLabel;
      const VarLabel* d_radiationFluxEINLabel;
      const VarLabel* d_radiationFluxWINLabel;
      const VarLabel* d_radiationFluxNINLabel;
      const VarLabel* d_radiationFluxSINLabel;
      const VarLabel* d_radiationFluxTINLabel;
      const VarLabel* d_radiationFluxBINLabel;

      
//     new_dw->getModifiable(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch);
//     new_dw->getModifiable(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch);
//     new_dw->getModifiable(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch);
//     new_dw->getModifiable(pressure,  d_lab->d_pressurePSLabel,    indx, patch);
//     new_dw->getModifiable(scalar,    d_lab->d_scalarSPLabel,      indx, patch);




    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, indx, patch)){ 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    }else{ 
      throw VariableNotFoundInGrid("cellInformation"," ", __FILE__, __LINE__);
    }
    
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    double pi = acos(-1.0);

    // absorption coefficient initialization for control volumes
    // cell centered variables: xx, yy, zz
    // benchmark case absorption coefficient
    //Note: the origin has to be right at the center of the domain.
    // otherwise the xx, yy, zz have to be modified to get the same absorbCoeff
    
    for ( CellIterator iter=patch->getCellIterator_New(); !iter.done(); iter++){
      IntVector currCell = *iter;

      absorbCoeff[*iter] = 0.9 * ( 1 - 2 * abs (cellinfo->xx[currCell.x()]) )
      * ( 1 - 2 * abs ( cellinfo->yy[currCell.y()] ) )
      * ( 1 - 2 * abs ( cellinfo->zz[currCell.z()] ) ) + 0.1;
      
    }

    
    //CELL centered variables
    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
      IntVector currCell = *iter; 
    
      if (d_mms == "constantMMS") { 
        pressure[*iter] = d_cp;
        scalar[*iter]   = d_phi0;
        if (d_calcExtraScalars) {
      
          for (int i=0; i < static_cast<int>(d_extraScalars.size()); i++) {
           CCVariable<double> extra_scalar;
           new_dw->allocateAndPut(extra_scalar,
                                  d_extraScalars[i]->getScalarLabel(),indx, patch);
           extra_scalar.initialize(d_esphi0);
         }
        }
      } else if (d_mms == "almgrenMMS") {         
        pressure[*iter] = -d_amp*d_amp/4 * (cos(4.0*pi*cellinfo->xx[currCell.x()])
                          + cos(4.0*pi*cellinfo->yy[currCell.y()]));
        scalar[*iter]   = 0.0;
      }
    }

    //X-FACE centered variables 
    for (CellIterator iter=patch->getSFCXIterator__New(); !iter.done(); iter++){
      IntVector currCell = *iter; 

      if (d_mms == "constantMMS") { 
        uVelocity[*iter] = d_cu; 
      } else if (d_mms == "almgrenMMS") { 
        // for mms in x-y plane
        uVelocity[*iter] = 1 - d_amp * cos(2.0*pi*cellinfo->xu[currCell.x()])
                           * sin(2.0*pi*cellinfo->yy[currCell.y()]);       
      }
    }

    //Y-FACE centered variables 
    for (CellIterator iter=patch->getSFCYIterator__New(); !iter.done(); iter++){
      IntVector currCell = *iter; 

      if (d_mms == "constantMMS") { 
        vVelocity[*iter] = d_cv; 
      } else if (d_mms == "almgrenMMS") { 
        // for mms in x-y plane
        vVelocity[*iter] = 1 + d_amp * sin(2.0*pi*cellinfo->xx[currCell.x()])
                              * cos(2.0*pi*cellinfo->yv[currCell.y()]); 
        
      }
    }

    //Z-FACE centered variables 
    for (CellIterator iter=patch->getSFCZIterator__New(); !iter.done(); iter++){

      if (d_mms == "constantMMS") { 
        wVelocity[*iter] = d_cw; 
      } else if (d_mms == "almgrenMMS") { 
        // for mms in x-y plane
        wVelocity[*iter] =  0.0;
      }
    }

    // Previously, we had the boundaries initialized here (below this comment).  I have removed
    // this since a) it seemed incorrect and b) because it would fit better
    // where BC's were applied.  Note that b) implies that we have a better
    // BC abstraction.
    // -Jeremy

  }
}
