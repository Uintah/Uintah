#ifndef Packages_Uintah_CCA_Components_ICE_Rieman_h
#define Packages_Uintah_CCA_Components_ICE_Rieman_h

#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/CircleBCData.h>
#include <Packages/Uintah/Core/Grid/BCDataArray.h>
#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Math/MiscMath.h>

using namespace Uintah;
namespace Uintah {

void computeStagnationProperties(double &stag_press,
                                 double &stag_temp,
                                 double &stag_density,
                                 double &time,
                                 SimulationStateP& sharedState);
                              
 void Solve_Riemann_problem(
        int     qLoLimit,               
        int     qHiLimit,    
        double  origin,           
        double  delQ,
        double  time,
        double  gamma,
        double  p1,                     
        double  rho1,                   
        double  u1,                     
        double  a1,
        double  p4,                     
        double  rho4,                   
        double  u4,                     
        double  a4,
        double  *u_Rieman,              
        double  *a_Rieman,              
        double  *p_Rieman,              
        double  *rho_Rieman,            
        double  *T_Rieman,
        double  R);
                
  double p2_p1_ratio(
          double  gamma,
          double  p4_p1,
          double  p2_p1_guess,
          double  a4,
          double  a1,
          double  u4,
          double  u1  );

  void solveRiemannProblemInterface( 
          const double t_final,
          const double Length, int ncells,
          const double u4, const double p4, const double rho4,                                   
          const double u1, const double p1, const double rho1,
          const double diaphragm_locaton ,
          const int probeCell,
          double &press,    // at probe location
          double &Temp,
          double &rho,
          double &vel);
          
    void setNGCVelocity_BC(const Patch* patch,
                       const Patch::FaceType face,
                       CCVariable<Vector>& q_CC,
                       const string& var_desc,
                       const vector<IntVector> bound,
                       const string& bc_kind,
			  const int mat_id,
			  const int child,
                       SimulationStateP& sharedState);
                                     
//______________________________________________________________________
//   Function~  setNGC_Nozzle_BC
template<class T,class V >
void setNGC_Nozzle_BC(const Patch* patch,
                      const Patch::FaceType face,
                      T& q_CC,
                      const string& var_desc,       // variable description
                      const string& var_loc,        // variable location FC, CC
                      const vector<IntVector> bound,
                      const string& bc_kind,        //Dirichlet, Neumann, custom
			 const int mat_id,
			 const int child,
                      SimulationStateP& sharedState)
   {
  BCGeomBase* bc_geom_type = patch->getBCDataArray(face)->getChild(mat_id,child);
  cmp_type<CircleBCData> nozzle;
  
  // CC or FC variable
  IntVector offset(0,0,0);  //CC variable
  if(var_loc == "FC"){
    offset = IntVector(1,0,0); 
  }
  
  //__________________________________
  // if on the x- face and inside the nozzle
  if(bc_kind == "Custom" && 
     face == Patch::xminus && nozzle(bc_geom_type) ) {
     
    //__________________________________
    // problem specific hard coded variables
    // see fig 2 at top of NG_nozzleBCs.cc
    
    Vector dx_mg  = patch->dCell();               // dx of the main grid
    double Length = 2 * dx_mg.x();                // physical length of secondary grid
    int    ncells = 100;                          // # cells on secondary grid
    double dx_sg  = Length/(double)ncells;        // dx of the secondary grid             
    double diaphragm_location = 1.5 * dx_mg.x();  // relative to secondary grid
     
    //__________________________________
    // which cell is the probe cell
    int probeCell;
    if(var_loc == "CC"){   // ghostCell center location
      probeCell = (int)ceil(dx_mg.x()/dx_sg);
    }
    if(var_loc == "FC"){   // FC overlaps the diaphram location 
      probeCell = (int)ceil(diaphragm_location/dx_sg);
    }
    
    cout << " inside setNGC_NozzlePressure " << bc_kind 
         << " variable " << var_desc << " variable Loc:" << var_loc<<endl;
    cout << " probeCell " << probeCell<< endl;

    
    //__________________________________
    // compute the stagnation properties at location 4.
    double p4, rho4, u4, t_final, notUsed;
    computeStagnationProperties(p4,notUsed, rho4,t_final, sharedState);
    u4 = 0.0;
   
    vector<IntVector>::const_iterator iter;
    for (iter=bound.begin(); iter != bound.end(); iter++) {
      IntVector c = *iter + offset;
      IntVector adj = *iter + IntVector(1,0,0);
      
       // Properties at location 1 this is a complete hack right now
      double p1 = p4/10.0;
      double rho1 = rho4/10.0;
      double u1 = 0.0;
      
      
      double p, Temp, rho, vel;   // probed cell variables
      p    = getNan();
      Temp = getNan();
      rho  = getNan();
      vel  = getNan();
      solveRiemannProblemInterface( t_final,Length,ncells,
                              u4, p4, rho4,                                   
                              u1, p1, rho1,
                              diaphragm_location,
                              probeCell,
                              p, Temp, rho, vel);
      if(var_desc == "Pressure"){
        q_CC[c] = V(p);
      }
      if(var_desc == "Temperature"){
        q_CC[c] = V(Temp);
      }
      if(var_desc == "Density"){
        q_CC[c] = V(rho);
      }
      if(var_desc == "Velocity" || var_desc == "Vel_FC"){
        q_CC[c] = V(vel);
      }
      
      cout << c << " " << var_desc << " " << q_CC[c] << endl;
    }
  }
}
                                     
};
#endif
