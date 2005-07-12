#ifndef Packages_Uintah_CCA_Components_ICE_Rieman_h
#define Packages_Uintah_CCA_Components_ICE_Rieman_h

#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/CircleBCData.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCond.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Math/MiscMath.h>
#include <Packages/Uintah/CCA/Ports/Output.h>

namespace Uintah {

struct NG_BC_vars{
  CCVariable<double> press_CC;
  CCVariable<Vector> vel_CC;
  CCVariable<double> rho_CC;
  Output* dataArchiver;
  IntVector c;
  bool dumpNow;
};

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
          NG_BC_vars* ng,
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
                       SimulationStateP& sharedState,
                       NG_BC_vars* CCVars);

void addRequires_NGNozzle(Task* t, 
                          const string& where,
                          ICELabel* lb,
                          const MaterialSubset* ice_matls);
                               
void getVars_for_NGNozzle( DataWarehouse* old_dw,
                           DataWarehouse* new_dw,
                           ICELabel* lb,
                           const Patch* patch,
                           const string& where,
                           bool& setNGBcs,
                           NG_BC_vars* ng );
 
void BC_values_using_IsentropicRelations(const double stag_press,
                                         const double stag_rho,
                                         const double stag_temp,
                                         double &static_press,
                                         double &static_temp,
                                         double &static_rho,
                                         double &static_vel);
                               
bool using_NG_hack(const ProblemSpecP& prob_spec);
                                                                
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424 // Template parameter not used in declaring arguments.
#endif                // This turns off SGI compiler warning.
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
                      SimulationStateP& sharedState,
                      NG_BC_vars* ng)
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
  if(bc_kind == "Custom" && face == Patch::xminus && nozzle(bc_geom_type) ) {
 
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
    int probeCell = 0;
    if(var_loc == "CC"){   // ghostCell center location
      probeCell = (int)ceil(dx_mg.x()/dx_sg);
    }
    if(var_loc == "FC"){   // FC overlaps the diaphram location 
      probeCell = (int)ceil(diaphragm_location/dx_sg);
    }
#if 0   
    cout << " inside setNGC_NozzlePressure " << bc_kind 
         << " variable " << var_desc << " variable Loc:" 
         << var_loc
         <<"\t bound limits = "<< *bound.begin()<< " "<< *(bound.end()-1) << endl;
    cout << " probeCell " << probeCell<< endl;
#endif
    
    //__________________________________
    // compute the stagnation properties at location 4.
    double p4, rho4, u4, T4, t_final;
    computeStagnationProperties(p4,T4, rho4,t_final, sharedState);
    u4 = 0.0;
   
    vector<IntVector>::const_iterator iter;


    for (iter=bound.begin(); iter != bound.end(); iter++) {
      IntVector c = *iter + offset;

      // restrict where we compute the BCs to only xminus face
      // without the z ghostcells or yminus ghost cells.
      if( (c.y() != -1 && c.z() == 0) || var_loc == "FC") {
        IntVector adj = *iter + IntVector(1,0,0);

        // Properties at location 1 
        double p1   = ng->press_CC[adj];
        double rho1 = ng->rho_CC[adj];
        double u1   = ng->vel_CC[adj].x();
        ng->c = c;     // cell index    

        if (p1 <=0 || rho1 <= 0 || finite(p1) == 0 || finite(rho1) == 0){
          cout << " c " << c << " adj " << adj << " p1 " << p1 
               << " rho1 " << rho1 << endl;
          throw InternalError(" setNGC_Nozzle_BC: one of initial conditions (p1, rho1) = 0", __FILE__, __LINE__);
        }
        
        //cout << " c " << c << " adj " << adj << " p1 " << p1 << " rho1 " << rho1 << " u1 " << u1 << endl;
        
        double p, Temp, rho, vel;   // probed cell variables
  //      BC_values_using_IsentropicRelations(p4, rho4, T4, p,Temp, rho, vel);

        Uintah::solveRiemannProblemInterface( t_final,Length,ncells,
                                              u4, p4, rho4,                                   
                                              u1, p1, rho1,
                                              diaphragm_location,
                                              probeCell, ng,
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
#if 0
        if (c == IntVector(-1,0,0) || c == IntVector(0,0,0) || c == IntVector(-1,1,0) ){      
          cout << c << " adj " << adj << var_desc << " " << q_CC[c] << endl;
          cout << "\t  p1 " << p1 << "\t rho1 " << rho1 << "\t u1 " << u1 << endl;
          cout << "\t  p4 " << p4 << "\t rho4 " << rho4 << "\t u4 " << u4 << endl;
          cout << "\t Temp " << Temp<< "\t p " << p << " rho " << rho << " vel " << vel <<endl;
        }
#endif
      }   // only on c.y() == 0 && c.z() == 0
    } // bound iterator
  } // if on xminus face and inside nozzle
}
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif
                                     
} // end namespace Uintah

#endif
