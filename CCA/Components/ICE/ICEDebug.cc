#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/UnknownVariable.h>
#include <iostream>
#include <fstream>

using std::ifstream;
using std::cerr;
using namespace SCIRun;
using namespace Uintah;

/* 
 ======================================================================*
 Function:  printData--
 Purpose:  Print to stderr a cell-centered, single material
_______________________________________________________________________ */
void    ICE::printData(const Patch* patch, int include_GC,
        const string&    message1,         /* message1                     */
        const string&    message2,         /* message to user              */
        const CCVariable<double>& q_CC)
{
 //__________________________________
 // Limit when we dump
  d_dbgTime= dataArchiver->getCurrentTime();    
  if ( d_dbgTime >= d_dbgStartTime && 
       d_dbgTime <= d_dbgStopTime  &&
       d_dbgTime >= d_dbgNextDumpTime) {
    d_dbgOldTime = d_dbgTime;      
    
    IntVector low, high; 

    cerr << "______________________________________________\n";
    cerr << "$" << message1 << "\n";
    cerr << "$" << message2 << "\n";

    if (include_GC == 1)  { 
      low   = patch->getCellLowIndex();
      high  = patch->getCellHighIndex();
    }
    if (include_GC == 0) {
      low   = patch->getInteriorCellLowIndex();
      high  = patch->getInteriorCellHighIndex();
    }

    cerr.setf(ios::scientific,ios::floatfield);
    cerr.precision(16);  
    for(int k = low.z(); k < high.z(); k++)  {
      for(int j = low.y(); j < high.y(); j++) {
        for(int i = low.x(); i < high.x(); i++) {
         IntVector idx(i, j, k);
         cerr << "[" << i << "," << j << "," << k << "]~ " 
              << q_CC[idx] << "  ";

         /*  cerr << "\n"; */
        }
       cerr << "\n";
      }
      cerr << "\n";
    }
    cerr <<" ______________________________________________\n";
    cerr.setf(ios::scientific ,ios::floatfield);
  }
}

/* 
 ======================================================================*
 Function:  printData--
 Purpose:  Print to stderr a cell-centered, single material
_______________________________________________________________________ */
void    ICE::printData(const Patch* patch, int include_GC,
        const string&    message1,       /* message1                     */
        const string&    message2,       /* message to user              */
        const CCVariable<int>& q_CC)
{
 //__________________________________
 // Limit when we dump
  d_dbgTime= dataArchiver->getCurrentTime();
  if ( d_dbgTime >= d_dbgStartTime && 
       d_dbgTime <= d_dbgStopTime  &&
       d_dbgTime >= d_dbgNextDumpTime) {
    d_dbgOldTime = d_dbgTime;      
    IntVector low, high; 

    cerr << "______________________________________________\n";
    cerr << "$" << message1 << "\n";
    cerr << "$" << message2 << "\n";

    if (include_GC == 1)  { 
      low   = patch->getCellLowIndex();
      high  = patch->getCellHighIndex();
    }
    if (include_GC == 0) {
      low   = patch->getInteriorCellLowIndex();
      high  = patch->getInteriorCellHighIndex();
    }

    for(int k = low.z(); k < high.z(); k++)  {
      for(int j = low.y(); j < high.y(); j++) {
        for(int i = low.x(); i < high.x(); i++) {
         IntVector idx(i, j, k);
         cerr << "[" << i << "," << j << "," << k << "]~ " 
              << q_CC[idx] << " ";

         /*  cerr << "\n"; */
        }
       cerr << "\n";
      }
      cerr << "\n";
    }
    cerr << " ______________________________________________\n";
  }
}
/* 
 ======================================================================*
 Function:  printVector--
 Purpose:  Print to stderr a cell-centered, single material
_______________________________________________________________________ */
void    ICE::printVector(const Patch* patch, int include_GC,
        const string&    message1,       /* message1                     */
        const string&    message2,       /* message to user              */
        int     component,              /*  x = 0,y = 1, z = 1          */
        const CCVariable<Vector>& q_CC)
{

 //__________________________________
 // Limit when we dump
  d_dbgTime= dataArchiver->getCurrentTime();
  if ( d_dbgTime >= d_dbgStartTime && 
       d_dbgTime <= d_dbgStopTime  &&
       d_dbgTime >= d_dbgNextDumpTime) {
    d_dbgOldTime = d_dbgTime;      
    IntVector low, high; 

    cerr << "______________________________________________\n";
    cerr << "$" << message1 << "\n";
    cerr << "$" << message2 << "\n";

    if (include_GC == 1)  { 
      low   = patch->getCellLowIndex();
      high  = patch->getCellHighIndex();
    }
    if (include_GC == 0) {
      low   = patch->getInteriorCellLowIndex();
      high  = patch->getInteriorCellHighIndex();
    }

    cerr.setf(ios::scientific,ios::floatfield);
    cerr.precision(16);  
    for(int k = low.z(); k < high.z(); k++)  {
      for(int j = low.y(); j < high.y(); j++) {
        for(int i = low.x(); i < high.x(); i++) {
         IntVector idx(i, j, k);
         cerr << "[" << i << "," << j << "," << k << "]~ " 
              <<  q_CC[idx](component) << "  ";

         /*  cerr << "\n"; */
        }
       cerr << "\n";
      }
      cerr << "\n";
    }
    cerr << " ______________________________________________\n";
    cerr.setf(ios::scientific, ios::floatfield);
  }
}


/* 
 ======================================================================*
 Function:  printData_FC--
 Purpose:  Print left face
_______________________________________________________________________ */
void    ICE::printData_FC(const Patch* patch, int include_GC,
        const string&    message1,        /* message1                     */
        const string&    message2,        /* message to user              */
        const SFCXVariable<double>& q_FC)
{
 //__________________________________
 // Limit when we dump
  d_dbgTime= dataArchiver->getCurrentTime();
  if ( d_dbgTime >= d_dbgStartTime && 
       d_dbgTime <= d_dbgStopTime  &&
       d_dbgTime >= d_dbgNextDumpTime) {
    d_dbgOldTime = d_dbgTime;
    IntVector low, high; 

    cerr << "______________________________________________\n";
    cerr << "$" << message1 << "\n";
    cerr << "$" << message2 << "\n";

    if (include_GC == 1)  { 
      low   = patch->getSFCXLowIndex();
      high  = patch->getSFCXHighIndex();
    }
    if (include_GC == 0) {
      low   = patch->getInteriorCellLowIndex();
      high  = patch->getInteriorCellHighIndex();
    }

    cerr.setf(ios::scientific,ios::floatfield);
    cerr.precision(16);  
    for(int k = low.z(); k < high.z(); k++)  {
      for(int j = low.y(); j < high.y(); j++) {
      //for(int j = high.y()-1; j >= low.y(); j--) {
        for(int i = low.x(); i < high.x(); i++) {
         IntVector idx(i, j, k);
         cerr << "[" << i << "," << j << "," << k << "]~ " <<
           q_FC[idx] << "  ";

         /* cerr <<"\n"; */
        }
        cerr << "\n";
      }
      cerr <<"\n";
    }
    cerr << " ______________________________________________\n";
    cerr.setf(ios::scientific, ios::floatfield);
  }
}
/* 
 ======================================================================*
 Function:  printData_FC--
 Purpose:   Prints bottom Face
_______________________________________________________________________ */
void    ICE::printData_FC(const Patch* patch, int include_GC,
        const string&    message1,         /* message1                     */
        const string&    message2,         /* message to user              */
        const SFCYVariable<double>& q_FC)
{
 //__________________________________
 // Limit when we dump
  d_dbgTime= dataArchiver->getCurrentTime();
  if ( d_dbgTime >= d_dbgStartTime && 
       d_dbgTime <= d_dbgStopTime  &&
       d_dbgTime >= d_dbgNextDumpTime) {
    d_dbgOldTime = d_dbgTime;      
    IntVector low, high; 

    cerr << "______________________________________________\n";
    cerr << "$" << message1 << "\n";
    cerr << "$" << message2 << "\n";

    if (include_GC == 1)  { 
      low   = patch->getSFCYLowIndex();
      high  = patch->getSFCYHighIndex();
    }
    if (include_GC == 0) {
      low   = patch->getInteriorCellLowIndex();
      high  = patch->getInteriorCellHighIndex();
    }

    cerr.setf(ios::scientific,ios::floatfield);
    cerr.precision(16);
    for(int k = low.z(); k < high.z(); k++)  {
      for(int j = low.y(); j < high.y(); j++) {
      //for(int j = high.y()-1; j >= low.y(); j--) {
        for(int i = low.x(); i < high.x(); i++) {
         IntVector idx(i, j, k);
         cerr << "[" << i << "," << j << "," << k << "]~ " <<  
           q_FC[idx] << "  ";

         /*  cerr << "\n"; */
        }
        cerr << "\n";
      }
      cerr << "\n";
    }
    cerr << " ______________________________________________\n";
    cerr.setf(ios::scientific, ios::floatfield);
  }
}

/* 
 ======================================================================*
 Function:  printData_FC--
 Purpose:  Piints back face
_______________________________________________________________________ */
void    ICE::printData_FC(const Patch* patch, int include_GC,
        const string&    message1,        /* message1                     */
        const string&    message2,        /* message to user              */
        const SFCZVariable<double>& q_FC)
{

 //__________________________________
 // Limit when we dump
  d_dbgTime= dataArchiver->getCurrentTime();
  if ( d_dbgTime >= d_dbgStartTime && 
       d_dbgTime <= d_dbgStopTime  &&
       d_dbgTime >= d_dbgNextDumpTime) {
    d_dbgOldTime = d_dbgTime;      
    IntVector low, high; 

    cerr << "______________________________________________\n";
    cerr << "$" << message1 << "\n";
    cerr << "$" << message2 << "\n";

    if (include_GC == 1)  { 
      low   = patch->getSFCZLowIndex();
      high  = patch->getSFCZHighIndex();
    }
    if (include_GC == 0) {
      low   = patch->getInteriorCellLowIndex();
      high  = patch->getInteriorCellHighIndex();
    }

    cerr.setf(ios::scientific,ios::floatfield);
    cerr.precision(16);   
    for(int k = low.z(); k < high.z(); k++)  {
      for(int j = low.y(); j < high.y(); j++) {
      //for(int j = high.y()-1; j >= low.y(); j--) {
        for(int i = low.x(); i < high.x(); i++) {
         IntVector idx(i, j, k);
         cerr << "[" << i << "," << j << "," << k << "]~ " << 
           q_FC[idx] << "  ";

         /*  cerr << "\n"; */
        }
       cerr << "\n";
      }
      cerr << "\n";
    }
    cerr << " ______________________________________________\n";
    cerr.setf(ios::scientific, ios::floatfield);
  }
}

/* 
 ======================================================================*
 Function:  readData--
 Purpose:  Print to stderr a cell-centered, single material
_______________________________________________________________________ */
void    ICE::readData(const Patch* patch, int include_GC,
        const string&    filename,        /* message1                     */
        const string&    var_name,        /* var_name              */
        CCVariable<double>& q_CC)
{
  int i, j, k,xLo, yLo, zLo, xHi, yHi, zHi;
  IntVector lowIndex, hiIndex; 
  string text;
  double number;
  
  ifstream fp(filename.c_str());
  if (!fp)
    throw UnknownVariable("Couldn't open the file with hardwired variables","");
        
  //  fscanf(fp,"______________________________________________\n");
  fp >> text;  // scan over the "______"
  fp >> text;
  fp >> text;
  
  if (var_name != text)
    throw UnknownVariable("You're trying to read in apples and orangs " + var_name + " " +  text,"");
  
  if (include_GC == 1)  { 
    lowIndex = patch->getCellLowIndex();
    hiIndex  = patch->getCellHighIndex();
  }
  if (include_GC == 0) {
    lowIndex = patch->getInteriorCellLowIndex();
    hiIndex  = patch->getInteriorCellHighIndex();
  }
  xLo = lowIndex.x();
  yLo = lowIndex.y();
  zLo = lowIndex.z();
  
  xHi = hiIndex.x();
  yHi = hiIndex.y();
  zHi = hiIndex.z();
  
  for(k = zLo; k < zHi; k++)  {
    for(j = yLo; j < yHi; j++) {
      for(i = xLo; i < xHi; i++) {
       IntVector idx(i, j, k);
       
       char c;
       fp.get(c);
       while ( c != '~') {         
         fp.get(c);
         // cerr << c;
       }
       
       fp >> number;
       if (!fp.good())       
	 throw UnknownVariable("Having problem reading " + var_name,"");
              
      // cerr << number;
       q_CC[idx] = number;
      }
      char c;
      fp >> c;
    }
    char c;
    fp >> c;
  }
  fp >> text;
}

/* 
 ======================================================================
 Function~  ICE::Message:
 Purpose~  Output an error message and stop the program if requested. 
 _______________________________________________________________________ */
void    ICE::Message(
        int     abort,          /* =1 then abort                            */
        const string&    message1,   
        const string&    message2,   
        const string&    message3) 
{        
  cerr << "\n\n ______________________________________________\n";
  cerr << message1 << "\n";
  cerr << message2 << "\n";
  cerr << message3 << "\n";
  cerr << "\n\n ______________________________________________\n";
  char* exitMode = getenv("ICE_DEBUGGER_ON_EXIT");

  if(!exitMode)
    exitMode = "no";    //default exit mode
  //______________________________
  // Now aborting program
  string mode(exitMode);
  if(abort == 1) {
    if(mode == "yes") {
      string c;
      cerr << "\n";
      cerr << "<c> = cvd\n";
      cin >> c;
      system("date");
      if(c == "c") system("cvd -P sus");
    }
    exit(1); 
  }
}

/* 
 ======================================================================*
 Function:  printConservedQuantities--
 If the switch is turned on then print out the conserved quantities.
_______________________________________________________________________ */
void ICE::printConservedQuantities(const ProcessorGroup*,  
                                   const PatchSubset* patches,
                                   const MaterialSubset* /*matls*/,
                                   DataWarehouse* /*old_dw*/,
                                   DataWarehouse* new_dw)
{
  
  int numICEmatls = d_sharedState->getNumICEMatls();
  int flag = -9;
  double mass;
  vector<Vector> mat_mom_xyz(numICEmatls,Vector(0.,0.,0.));
  vector<double> mat_mass(numICEmatls,0.);
  vector<double> mat_total_mom(numICEmatls,0.);
  vector<double> mat_total_eng(numICEmatls,0.);
  vector<double> mat_int_eng(numICEmatls,0.);
  vector<double> mat_KE(numICEmatls,0.);
  Vector total_mom_xyz(0.0, 0.0, 0.0);
  
  double total_momentum = 0.0;
  double total_energy   = 0.0;
  double total_mass     = 0.0;
  double total_KE       = 0.0;
  double total_int_eng  = 0.0; 
  
  static double initial_total_eng = 0.0;
  static double initial_total_mom = 0.0;
  static int n_passes;
  
  //__________________________________
  //  Loop over all the patches
  for(int p=0; p<patches->size(); p++)  {
    const Patch* patch = patches->get(p);
    cout << "Doing printConservedQuantities on patch " << patch->getID()
     << "\t\t ICE" << endl;
    constCCVariable<Vector> vel_CC;
    constCCVariable<double> rho_CC;
    constCCVariable<double> Temp_CC;
    constCCVariable<double> delP_Dilatate;
    Vector dx       = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    new_dw->get(delP_Dilatate,lb->delP_DilatateLabel, 0, patch,Ghost::None, 0);
    
    //__________________________________
    // Loop over all the ICE matls
    for (int m = 0; m < numICEmatls; m++ ) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      new_dw->get(vel_CC, lb->vel_CCLabel, indx, patch,  Ghost::None, 0);
      new_dw->get(rho_CC, lb->rho_CCLabel, indx, patch,  Ghost::None, 0);
      new_dw->get(Temp_CC,lb->temp_CCLabel,indx, patch,  Ghost::None, 0);
      double cv = ice_matl->getSpecificHeat();   
      
      //__________________________________
      // Accumulate the momenta and energy
      for (CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
       mass            = rho_CC[*iter] * cell_vol;
       mat_mom_xyz[m] += vel_CC[*iter]*rho_CC[*iter] * mass;
       double vel_sq = vel_CC[*iter].length() * vel_CC[*iter].length();
       mat_KE[m]      += 0.5 * mass * vel_sq;
       mat_int_eng[m] += mass * cv * Temp_CC[*iter];
       mat_mass[m]    += mass;
      }
    }  // numICEmatls loop

    if (switchTestConservation) {
      //__________________________________
      // This grossness checks to see if delPress
      // near a ghost cell is > 0  
      IntVector low, hi;
      
      low = delP_Dilatate.getLowIndex();
      hi  = delP_Dilatate.getHighIndex();
      // x_plus
      for (int j = low.y(); j<hi.y(); j++) {
       for (int k = low.z(); k<hi.z(); k++) {
         if( fabs(delP_Dilatate[IntVector(hi.x()-2,j,k)]) > 0.0 )  {
           flag = 1;
         }
       }
      }
      // x_minus
      for (int j = low.y(); j<hi.y(); j++) {
       for (int k = low.z(); k<hi.z(); k++) {
         if( fabs(delP_Dilatate[IntVector(low.x()+1,j,k)]) > 0.0 )  {
           flag = 1;
         }
       }
      }
      // y_plus
      for (int i = low.x(); i<hi.x(); i++) {
       for (int k = low.z(); k<hi.z(); k++) {
         if( fabs(delP_Dilatate[IntVector(i,hi.y()-2,k)]) > 0.0 )  {
           flag = 1;
         }
       }
      }
      // y_minus
      for (int i = low.x(); i<hi.x(); i++) {
       for (int k = low.z(); k<hi.z(); k++) {
         if( fabs(delP_Dilatate[IntVector(i,low.y()+1,k)]) > 0.0 )  {
           flag = 1;
         }
       }
      }
      // z_plus
      for (int i = low.x(); i<hi.x(); i++) {
       for (int j = low.y(); j<hi.y(); j++) {
         if( fabs(delP_Dilatate[IntVector(i,j,hi.z()-2)]) > 0.0 )   {
           flag = 1;
         }
       }
      }
      // z_minus
      for (int i = low.x(); i<hi.x(); i++) {
       for (int j = low.y(); j<hi.y(); j++) {
         if( fabs(delP_Dilatate[IntVector(i,j,low.z()+1)]) > 0.0 )   {
           flag = 1;
         }
       }
      }
    } // end switchTestConservation
  }  // patch loop
  
  //__________________________________
  //  Now compute totals and the change in quantities
  for (int m = 0; m < numICEmatls; m++ ) {
    mat_total_mom[m]= mat_mom_xyz[m].x() + mat_mom_xyz[m].y() + mat_mom_xyz[m].z();
    mat_total_eng[m]= mat_int_eng[m] + mat_KE[m];
    total_momentum += mat_total_mom[m];
    total_energy   += mat_total_eng[m];
    total_KE       += mat_KE[m];
    total_int_eng  += mat_int_eng[m];
    total_mass     += mat_mass[m];
    total_mom_xyz  += mat_mom_xyz[m];
    if ( n_passes < numICEmatls) {
      initial_total_eng += mat_total_eng[m];
      initial_total_mom += mat_total_mom[m];
      n_passes ++;
    } 
    
    cerr.setf(ios::scientific,ios::floatfield);
    cerr.precision(4);
    cerr << m << "Fluid mass " <<  mat_mass[m] << "\n";
    cerr.setf(ios::fixed,ios::floatfield);
    cerr << m << "Fluid momentum[ " << mat_mom_xyz[m].x() << ", " << 
      mat_mom_xyz[m].y() << ", " << mat_mom_xyz[m].z() << "]\t";
    cerr << "Components Sum: " << mat_total_mom[m] << "\n";
    cerr.setf(ios::scientific,ios::floatfield);
    cerr << m << "Fluid eng[internal " << mat_int_eng[m] <<  ", Kinetic: " 
        << mat_KE[m] << "]: " << mat_total_eng[m] << "\n";
  }
  double change_total_mom =
              100.0 * (total_momentum - initial_total_mom)/
              (initial_total_mom + d_SMALL_NUM);
  double change_total_eng =
              100.0 * (total_energy - initial_total_eng)/
              (initial_total_eng + d_SMALL_NUM);

  cerr.setf(ios::scientific, ios::floatfield);
  cerr.precision(4);
  cerr << "Totals: \t mass " << total_mass << " \t\tmomentum " << 
    total_momentum << " \t\t energy " << total_energy << "\n";
  cerr.setf(ios::fixed,ios::floatfield);
  cerr << "Percent change in total fluid mom.: " << change_total_mom <<
    " \t fluid total eng: " << change_total_eng << "\n";
  cerr.setf(ios::scientific, ios::floatfield);

  if (flag == 1)  {
    cout<< " D E L P R E S S   >   0   O N   B O U N D A R Y"<<endl;
    cout<< "******* N O   L O N G E R   C O N S E R V I N G *******\n"<<endl;
  }
  new_dw->put(sum_vartype(total_mass),      lb->TotalMassLabel);
  new_dw->put(sum_vartype(total_KE),        lb->KineticEnergyLabel);
  new_dw->put(sum_vartype(total_int_eng),   lb->TotalIntEngLabel);
  new_dw->put(sumvec_vartype(total_mom_xyz),  lb->CenterOfMassVelocityLabel);
}
