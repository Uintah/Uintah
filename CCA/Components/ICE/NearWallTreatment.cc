#include <Packages/Uintah/CCA/Components/ICE/NearWallTreatment.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/BoundCond.h>

#define d_SMALL_NUM 1.0e-100
using namespace Uintah;

NearWallTreatment::NearWallTreatment()
{
}

NearWallTreatment::~NearWallTreatment()
{
}


/* ---------------------------------------------------------------------
  Function~ computeNearBoundaryWallValue 
  Purpose~ Calculate turbulent viscosity near boundary wall with damping function
   -----------------------------------------------------------------------  */
void NearWallTreatment::computeNearBoundaryWallValue(const Patch* patch,
                                                     const CCVariable<Vector>& vel_CC,
                                                     const CCVariable<double>& rho_CC,
                                                     const int mat_id,
                                                     const double viscosity,
                                                     CCVariable<double>& turb_viscosity)
{
  Vector dx = patch->dCell();
  double delX_2 = dx.x()/2;
  double delY_2 = dx.y()/2;
  double delZ_2 = dx.z()/2;   
  double tau_wall, u_tau, vel_tot, x_plus, y_plus, z_plus, damping_function;
  IntVector low,hi;
  low = vel_CC.getLowIndex() + IntVector(1,1,1);
  hi = vel_CC.getHighIndex() - IntVector(1,1,1);
    
  for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
    face=Patch::nextFace(face)){ 
    const BoundCondBase *bcs;
    const BoundCond<Vector>* new_bcs;
    string kind = "Velocity";
    if (patch->getBCType(face) == Patch::None) {
      bcs     = patch->getBCValues(mat_id,kind,face);
      new_bcs = dynamic_cast<const BoundCond<Vector> *>(bcs);
    } else
      continue; 
    if (new_bcs != 0) {
      if (new_bcs->getKind() == "Dirichlet"){ 
         if (new_bcs->getValue() == Vector(0.0,0.0,0.0)){

          switch (face) {
          case Patch::xplus:
           for (int j = low.y(); j<hi.y(); j++) {
            for (int k = low.z(); k<hi.z(); k++) {
                    IntVector c = IntVector(hi.x()-1,j,k);
                    vel_tot = sqrt(vel_CC[c].y()*vel_CC[c].y()+vel_CC[c].z()*vel_CC[c].z());
                    tau_wall = viscosity*vel_tot/(delX_2);
                    u_tau = sqrt(tau_wall/rho_CC[c]);
                    x_plus = rho_CC[c]*u_tau*(delX_2)/viscosity;
//                   damping_function = 1-exp(-(x_plus/25)*(x_plus/25));
                    damping_function = 1-exp(-x_plus/25);
                    turb_viscosity[c] = turb_viscosity[c]*damping_function*damping_function;
             }
            }
           break;
          case Patch::xminus:
           for (int j = low.y(); j<hi.y(); j++) {
            for (int k = low.z(); k<hi.z(); k++) {
                    IntVector c = IntVector(low.x(),j,k);
                    vel_tot = sqrt(vel_CC[c].y()*vel_CC[c].y()+vel_CC[c].z()*vel_CC[c].z());
                    tau_wall = viscosity*vel_tot/(delX_2);
                    u_tau = sqrt(tau_wall/rho_CC[c]);
                    x_plus = rho_CC[c]*u_tau*(delX_2)/viscosity;
//                   damping_function = 1-exp(-(x_plus/25)*(x_plus/25));
                    damping_function = 1-exp(-x_plus/25);
                    turb_viscosity[c] = turb_viscosity[c]*damping_function*damping_function;         
             }
            }
           break;
          case Patch::yplus:
           for (int i = low.x(); i<hi.x(); i++) {
            for (int k = low.z(); k<hi.z(); k++) {
                    IntVector c = IntVector(i,hi.y()-1,k);
                    vel_tot = sqrt(vel_CC[c].x()*vel_CC[c].x()+vel_CC[c].z()*vel_CC[c].z());
                    tau_wall = viscosity*vel_tot/(delY_2);
                    u_tau = sqrt(tau_wall/rho_CC[c]);
                    y_plus = rho_CC[c]*u_tau*(delY_2)/viscosity;
//                    damping_function = 1-exp(-(y_plus/25)*(y_plus/25));
                    damping_function = 1-exp(-y_plus/25);
                    turb_viscosity[c] = turb_viscosity[c]*damping_function*damping_function;         
             }
            }
           break;
          case Patch::yminus:
           for (int i = low.x(); i<hi.x(); i++) {
            for (int k = low.z(); k<hi.z(); k++) {
                    IntVector c = IntVector(i,low.y(),k);  
                    vel_tot = sqrt(vel_CC[c].x()*vel_CC[c].x()+vel_CC[c].z()*vel_CC[c].z());       
                    tau_wall = viscosity*vel_tot/(delX_2);
                    u_tau = sqrt(tau_wall/rho_CC[c]);
                    y_plus = rho_CC[c]*u_tau*(delY_2)/viscosity;
//                    damping_function = 1-exp(-(y_plus/25)*(y_plus/25));
                    damping_function = 1-exp(-y_plus/25);
                    turb_viscosity[c] = turb_viscosity[c]*damping_function*damping_function;                  
             }
            }
           break;
          case Patch::zplus:
           for (int i = low.x(); i<hi.x(); i++) {
            for (int j = low.y(); j<hi.y(); j++) {
                    IntVector c = IntVector(i,j,hi.z()-1);
                    vel_tot = sqrt(vel_CC[c].y()*vel_CC[c].y()+vel_CC[c].x()*vel_CC[c].x());
                    tau_wall = viscosity*vel_tot/(delZ_2);
                    u_tau = sqrt(tau_wall/rho_CC[c]);
                    z_plus = rho_CC[c]*u_tau*(delZ_2)/viscosity;
//                    damping_function = 1-exp(-(z_plus/25)*(z_plus/25));
                    damping_function = 1-exp(-z_plus/25);
                    turb_viscosity[c] = turb_viscosity[c]*damping_function*damping_function;         
             }
            }
           break;
          case Patch::zminus:
           for (int i = low.x(); i<hi.x(); i++) {
            for (int j = low.y(); j<hi.y(); j++) {
                    IntVector c = IntVector(i,j,low.z());
                    vel_tot = sqrt(vel_CC[c].y()*vel_CC[c].y()+vel_CC[c].x()*vel_CC[c].x());
                    tau_wall = viscosity*vel_tot/(delZ_2);
                    u_tau = sqrt(tau_wall/rho_CC[c]);
                    z_plus = rho_CC[c]*u_tau*(delZ_2)/viscosity;
//                   damping_function = 1-exp(-(z_plus/25)*(z_plus/25));
                    damping_function = 1-exp(-z_plus/25);
                    turb_viscosity[c] = turb_viscosity[c]*damping_function*damping_function;         
             }
            }
           break;
          case Patch::numFaces:
           break;
          case Patch::invalidFace:
           break;
          } //loop switch
 
         }// if new_bcs->getValue() == Vector(0.0,0.0,0.0)
      }// if new_bcs->getKind() == "Dirichlet"
    }// if new_bcs != 0
  }// end face loop
}

/* ---------------------------------------------------------------------
  Function~ computeNearSolidInterfaceValue 
  Purpose~ Calculate turbulent viscosity near moving structure with damping function
 -----------------------------------------------------------------------  */
void NearWallTreatment::computeNearSolidInterfaceValue(const Patch* patch,
                                                       const CCVariable<Vector>& vel_CC,
                                                       const CCVariable<double>& rho_CC,
                                                       const CCVariable<double>& vol_frac,
                                                       const NCVariable<double>& NC_CCweight,
                                                       const NCVariable<double>& NCsolidMass,
                                                       const NCVariable<Vector>& NCvelocity,
                                                       const double viscosity,
                                                       CCVariable<double>& turb_viscosity)
{

    Vector dx = patch->dCell();
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();
    double vol = delX * delY * delZ;
    double MaxNormal, aveNCvelX, aveNCvelY, aveNCvelZ; 
    double tau_wall, u_tau, vel_tot, distance_cc, distance_plus, damping_function;    

    //  Loop over cells
    //  find surface and surface normals
    //  compute wall function
    //  end loop over cells
        
    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      IntVector nodeIdx[8];
      patch->findNodesFromCell(*iter,nodeIdx);
      double MaxMass = d_SMALL_NUM;
      double MinMass = 1.0/d_SMALL_NUM;

      for (int nN=0; nN<8; nN++) {
            MaxMass = std::max(MaxMass,NC_CCweight[nodeIdx[nN]]*
                                       NCsolidMass[nodeIdx[nN]]);
            MinMass = std::min(MinMass,NC_CCweight[nodeIdx[nN]]*
                                       NCsolidMass[nodeIdx[nN]]);
      }
cout<<"ccccccccccc"<<c<<" "<<vol_frac[c]<<""<<vel_CC[c]<<endl;
      if ((MaxMass-MinMass)/MaxMass == 1.0 && (MaxMass > d_SMALL_NUM)){ 
      cout<<"aaaaaaaaaaaaa"<<c<<endl;           
           if ( vol_frac[c] > 0.001 && vol_frac[c] < (1-0.001)) {
cout<<"bbbbbbbbbbbbb"<<c<<" "<<vol_frac[c]<<endl;
            double gradRhoX = (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                     NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                     NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                     NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]])
                   -
                   ( NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                     NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
                     NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
                     NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]]) / dx.x();
            double gradRhoY = (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                     NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                     NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                     NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]])
                   - 
                   ( NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                     NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
                     NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
                     NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]]) / dx.y();
            double gradRhoZ = (NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                     NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
                     NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
                     NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                 -
                   ( NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                     NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                     NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                     NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]) / dx.z();

            double absGradRho = sqrt(gradRhoX*gradRhoX +
                                      gradRhoY*gradRhoY +
                                      gradRhoZ*gradRhoZ );

             double normalX = gradRhoX/absGradRho;
             double normalY = gradRhoY/absGradRho;
             double normalZ = gradRhoZ/absGradRho;

            MaxNormal = abs(normalX);
            MaxNormal = std::max(MaxNormal,abs(normalY));
            MaxNormal = std::max(MaxNormal,abs(normalZ));
            if (MaxNormal == abs(normalX)){
//              normalX = normalX/abs(normalX);
//          normalY = 0.0;
//          normalZ = 0.0;
    //calculate the distance from cell center to surface            
            
              distance_cc = abs(delX - vol*vol_frac[c]/(delY*delZ));

              if (normalX > 0){
            aveNCvelY = NCvelocity[nodeIdx[0]].y()*NC_CCweight[nodeIdx[0]]+
                     NCvelocity[nodeIdx[1]].y()*NC_CCweight[nodeIdx[1]]+
                     NCvelocity[nodeIdx[2]].y()*NC_CCweight[nodeIdx[2]]+
                     NCvelocity[nodeIdx[3]].y()*NC_CCweight[nodeIdx[3]];
            
                aveNCvelZ = NCvelocity[nodeIdx[0]].z()*NC_CCweight[nodeIdx[0]]+
                     NCvelocity[nodeIdx[1]].z()*NC_CCweight[nodeIdx[1]]+
                     NCvelocity[nodeIdx[2]].z()*NC_CCweight[nodeIdx[2]]+
                     NCvelocity[nodeIdx[3]].z()*NC_CCweight[nodeIdx[3]];
          }
          else {
            aveNCvelY = NCvelocity[nodeIdx[4]].y()*NC_CCweight[nodeIdx[4]]+
                     NCvelocity[nodeIdx[5]].y()*NC_CCweight[nodeIdx[5]]+
                     NCvelocity[nodeIdx[6]].y()*NC_CCweight[nodeIdx[6]]+
                     NCvelocity[nodeIdx[7]].y()*NC_CCweight[nodeIdx[7]];
            
                aveNCvelZ = NCvelocity[nodeIdx[4]].z()*NC_CCweight[nodeIdx[4]]+
                     NCvelocity[nodeIdx[5]].z()*NC_CCweight[nodeIdx[5]]+
                     NCvelocity[nodeIdx[6]].z()*NC_CCweight[nodeIdx[6]]+
                     NCvelocity[nodeIdx[7]].z()*NC_CCweight[nodeIdx[7]];
          }
          vel_tot = sqrt(pow((vel_CC[c].y()-aveNCvelY),2)
                           +pow((vel_CC[c].z()-aveNCvelZ),2)); 
          tau_wall = viscosity*vel_tot/distance_cc;
              u_tau = sqrt(tau_wall/rho_CC[c]);
              distance_plus = rho_CC[c]*u_tau*(distance_cc)/viscosity;
              damping_function = 1-exp(-distance_plus/25);
              turb_viscosity[c] = turb_viscosity[c]*damping_function*damping_function;
                                               
        }
            if (MaxNormal == abs(normalY)){
//              normalX = 0.0;
//          normalY = normalY/abs(normalY);
//          normalZ = 0.0;
    //calculate the distance from cell center to surface            
            
              distance_cc = abs(delY - vol*vol_frac[c]/(delX*delZ));

              if (normalY > 0){
            aveNCvelX = NCvelocity[nodeIdx[0]].x()*NC_CCweight[nodeIdx[0]]+
                     NCvelocity[nodeIdx[1]].x()*NC_CCweight[nodeIdx[1]]+
                     NCvelocity[nodeIdx[4]].x()*NC_CCweight[nodeIdx[4]]+
                     NCvelocity[nodeIdx[5]].x()*NC_CCweight[nodeIdx[5]];
            
                aveNCvelZ = NCvelocity[nodeIdx[2]].z()*NC_CCweight[nodeIdx[2]]+
                     NCvelocity[nodeIdx[3]].z()*NC_CCweight[nodeIdx[3]]+
                     NCvelocity[nodeIdx[6]].z()*NC_CCweight[nodeIdx[6]]+
                     NCvelocity[nodeIdx[7]].z()*NC_CCweight[nodeIdx[7]];
          }
          else {
            aveNCvelX = NCvelocity[nodeIdx[0]].x()*NC_CCweight[nodeIdx[0]]+
                     NCvelocity[nodeIdx[1]].x()*NC_CCweight[nodeIdx[1]]+
                     NCvelocity[nodeIdx[4]].x()*NC_CCweight[nodeIdx[4]]+
                     NCvelocity[nodeIdx[5]].x()*NC_CCweight[nodeIdx[5]];
            
                aveNCvelZ = NCvelocity[nodeIdx[2]].z()*NC_CCweight[nodeIdx[2]]+
                     NCvelocity[nodeIdx[3]].z()*NC_CCweight[nodeIdx[3]]+
                     NCvelocity[nodeIdx[6]].z()*NC_CCweight[nodeIdx[6]]+
                     NCvelocity[nodeIdx[7]].z()*NC_CCweight[nodeIdx[7]];
          }
          vel_tot = sqrt(pow((vel_CC[c].x()-aveNCvelX),2)
                           +pow((vel_CC[c].z()-aveNCvelZ),2)); 
          tau_wall = viscosity*vel_tot/distance_cc;
              u_tau = sqrt(tau_wall/rho_CC[c]);
              distance_plus = rho_CC[c]*u_tau*(distance_cc)/viscosity;
              damping_function = 1-exp(-distance_plus/25);
              turb_viscosity[c] = turb_viscosity[c]*damping_function*damping_function;            
            }
            if (MaxNormal == abs(normalZ)){
//              normalX = 0.0;
//          normalY = 0.0;
//          normalZ = normalZ/abs(normalZ);
     //calculate the distance from cell center to surface            
            
              distance_cc = abs(delZ - vol*vol_frac[c]/(delY*delX));

              if (normalZ > 0){
            aveNCvelY = NCvelocity[nodeIdx[1]].y()*NC_CCweight[nodeIdx[1]]+
                     NCvelocity[nodeIdx[3]].y()*NC_CCweight[nodeIdx[3]]+
                     NCvelocity[nodeIdx[5]].y()*NC_CCweight[nodeIdx[5]]+
                     NCvelocity[nodeIdx[7]].y()*NC_CCweight[nodeIdx[7]];
            
                aveNCvelX = NCvelocity[nodeIdx[0]].x()*NC_CCweight[nodeIdx[0]]+
                     NCvelocity[nodeIdx[2]].x()*NC_CCweight[nodeIdx[2]]+
                     NCvelocity[nodeIdx[4]].x()*NC_CCweight[nodeIdx[4]]+
                     NCvelocity[nodeIdx[6]].x()*NC_CCweight[nodeIdx[6]];
          }
          else {
            aveNCvelY = NCvelocity[nodeIdx[1]].y()*NC_CCweight[nodeIdx[1]]+
                     NCvelocity[nodeIdx[3]].y()*NC_CCweight[nodeIdx[3]]+
                     NCvelocity[nodeIdx[5]].y()*NC_CCweight[nodeIdx[5]]+
                     NCvelocity[nodeIdx[7]].y()*NC_CCweight[nodeIdx[7]];
            
                aveNCvelX = NCvelocity[nodeIdx[0]].x()*NC_CCweight[nodeIdx[0]]+
                     NCvelocity[nodeIdx[2]].x()*NC_CCweight[nodeIdx[2]]+
                     NCvelocity[nodeIdx[4]].x()*NC_CCweight[nodeIdx[4]]+
                     NCvelocity[nodeIdx[6]].x()*NC_CCweight[nodeIdx[6]];
          }
          vel_tot = sqrt(pow((vel_CC[c].y()-aveNCvelY),2)
                           +pow((vel_CC[c].x()-aveNCvelX),2)); 
          tau_wall = viscosity*vel_tot/distance_cc;
              u_tau = sqrt(tau_wall/rho_CC[c]);
              distance_plus = rho_CC[c]*u_tau*(distance_cc)/viscosity;
              damping_function = 1-exp(-distance_plus/25);
              turb_viscosity[c] = turb_viscosity[c]*damping_function*damping_function;            
            } //if MaxNormal
                
           } //if vol_frac
            
      }  // if a surface cell
     }    // cellIterator
 
}




