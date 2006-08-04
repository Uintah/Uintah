//  MPMMaterial.cc

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ParticleCreatorFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ParticleCreator.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryObject.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/GeometryPiece/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/NullGeometryPiece.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
//#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include   <iostream>
#include   <string>
#include   <list>
#include <sgi_stl_warnings_on.h>

#define d_TINY_RHO 1.0e-12 // also defined  ICE.cc and ICEMaterial.cc 

using namespace std;
using namespace Uintah;
using namespace SCIRun;

// Standard Constructor
MPMMaterial::MPMMaterial(ProblemSpecP& ps, SimulationStateP& ss)
  : Material(ps), d_cm(0),  d_particle_creator(0)
{
  d_lb = scinew MPMLabel();
  d_flag = scinew MPMFlags();
  d_flag->readMPMFlags(ps);
  // The standard set of initializations needed
  standardInitialization(ps);
  
  d_cm->setSharedState(ss.get_rep());
  // Check to see which ParticleCreator object we need

  d_particle_creator = ParticleCreatorFactory::create(ps,this,d_flag);

}

void
MPMMaterial::standardInitialization(ProblemSpecP& ps)

{
  // Follow the layout of the input file
  // Steps:
  // 1.  Determine the type of constitutive model and create it.
  // 2.  Get the general properties of the material such as
  //     density, thermal_conductivity, specific_heat.
  // 3.  Loop through all of the geometry pieces that make up a single
  //     geometry object.
  // 4.  Within the geometry object, assign the boundary conditions
  //     to the object.
  // 5.  Assign the velocity field.

  // Step 1 -- create the constitutive gmodel.
  d_cm = ConstitutiveModelFactory::create(ps,d_flag);
  if(!d_cm){
    ostringstream desc;
    desc << "An error occured in the ConstitutiveModelFactory that has \n" 
	 << " slipped through the existing bullet proofing. Please tell \n"
	 << " either Jim, John or Todd "<< endl; 
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }

  // Step 2 -- get the general material properties

  ps->require("density",d_density);
  ps->require("thermal_conductivity",d_thermalConductivity);
  ps->require("specific_heat",d_specificHeat);
  
  // Assume the the centered specific heat is C_v
  d_Cv = d_specificHeat;

  // Set C_p = C_v if not C_p data are entered
  d_Cp = d_Cv;
  ps->get("C_p",d_Cp);

  d_troom = 294.0; d_tmelt = 295.0;
  ps->get("room_temp", d_troom);
  ps->get("melt_temp", d_tmelt);

  // This is currently only used in the implicit code, but should
  // be put to use in the explicit code as well.
  d_is_rigid=false;
  ps->get("is_rigid", d_is_rigid);
   
  d_includeFlowWork = false;
  ps->get("includeFlowWork",d_includeFlowWork);

  // Step 3 -- Loop through all of the pieces in this geometry object
  //int piece_num = 0;
  list<string> geom_obj_data;
  geom_obj_data.push_back("temperature");

  for (ProblemSpecP geom_obj_ps = ps->findBlock("geom_object");
       geom_obj_ps != 0; 
       geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

    vector<GeometryPieceP> pieces;
    GeometryPieceFactory::create(geom_obj_ps, pieces);

    GeometryPieceP mainpiece;
    if(pieces.size() == 0){
      throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
    } else if(pieces.size() > 1){
      mainpiece = scinew UnionGeometryPiece(pieces);
    } else {
      mainpiece = pieces[0];
    }

    //    piece_num++;
    d_geom_objs.push_back(scinew GeometryObject(mainpiece, geom_obj_ps,
                                                geom_obj_data));
  }
}

// Default constructor
MPMMaterial::MPMMaterial() : d_cm(0), d_particle_creator(0)
{
  d_lb = scinew MPMLabel();
}

MPMMaterial::~MPMMaterial()
{
  delete d_lb;
  delete d_flag;
  delete d_cm;
  delete d_particle_creator;

  for (int i = 0; i<(int)d_geom_objs.size(); i++) {
    delete d_geom_objs[i];
  }
}

void MPMMaterial::registerParticleState(SimulationState* sharedState)
{
  sharedState->d_particleState.push_back(d_particle_creator->returnParticleState());
  sharedState->d_particleState_preReloc.push_back(d_particle_creator->returnParticleStatePreReloc());
}

ProblemSpecP MPMMaterial::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP mpm_ps = Material::outputProblemSpec(ps);
  mpm_ps->appendElement("density",d_density);
  mpm_ps->appendElement("thermal_conductivity",d_thermalConductivity);
  mpm_ps->appendElement("specific_heat",d_specificHeat);
  mpm_ps->appendElement("C_p",d_Cp);
  mpm_ps->appendElement("room_temp",d_troom);
  mpm_ps->appendElement("melt_temp",d_tmelt);
  mpm_ps->appendElement("is_rigid",d_is_rigid);
  mpm_ps->appendElement("includeFlowWork",d_includeFlowWork);
  d_cm->outputProblemSpec(mpm_ps);

  for (vector<GeometryObject*>::const_iterator it = d_geom_objs.begin();
       it != d_geom_objs.end(); it++) {
    (*it)->outputProblemSpec(mpm_ps);
  }

  return mpm_ps;
}

void
MPMMaterial::copyWithoutGeom(ProblemSpecP& ps,const MPMMaterial* mat, 
                             MPMFlags* flags)
{
  d_flag = flags;
  d_cm = mat->d_cm->clone();
  d_density = mat->d_density;
  d_thermalConductivity = mat->d_thermalConductivity;
  d_specificHeat = mat->d_specificHeat;
  d_Cv = mat->d_Cv;
  d_Cp = mat->d_Cp;
  d_troom = mat->d_troom;
  d_tmelt = mat->d_tmelt;
  d_is_rigid = mat->d_is_rigid;

  // Check to see which ParticleCreator object we need

  d_particle_creator = ParticleCreatorFactory::create(ps,this,flags);

}

ConstitutiveModel* MPMMaterial::getConstitutiveModel() const
{
  // Return the pointer to the constitutive model associated
  // with this material

  return d_cm;
}

particleIndex MPMMaterial::countParticles(const Patch* patch)
{
  return d_particle_creator->countParticles(patch,d_geom_objs);
}

void MPMMaterial::createParticles(particleIndex numParticles,
				  CCVariable<short int>& cellNAPID,
				  const Patch* patch,
				  DataWarehouse* new_dw)
{
  d_particle_creator->createParticles(this,numParticles,cellNAPID,
				      patch,new_dw,d_geom_objs);
}

ParticleCreator* MPMMaterial::getParticleCreator()
{
  return  d_particle_creator;
}

double MPMMaterial::getInitialDensity() const
{
  return d_density;
}

double 
MPMMaterial::getInitialCp() const
{
  return d_Cp;
}

double 
MPMMaterial::getInitialCv() const
{
  return d_Cv;
}

double MPMMaterial::getRoomTemperature() const
{
  return d_troom;
}

double MPMMaterial::getMeltTemperature() const
{
  return d_tmelt;
}

int MPMMaterial::nullGeomObject() const
{
  for (int obj = 0; obj <(int)d_geom_objs.size(); obj++) {
    GeometryPieceP piece = d_geom_objs[obj]->getPiece();
    NullGeometryPiece* null_piece = dynamic_cast<NullGeometryPiece*>(piece.get_rep());
    if (null_piece)
      return obj;
  }
  return -1;
}

bool MPMMaterial::getIsRigid() const
{
  return d_is_rigid;
}
/* --------------------------------------------------------------------- 
 Function~  MPMMaterial::initializeCells--
 Notes:  This function initializeCCVariables.  Reasonable values for 
 CC Variables need to be present in all the cells and evolve, even though
 there is no mass.  This is essentially the same routine that is in
 ICEMaterial.cc
_____________________________________________________________________*/
void MPMMaterial::initializeCCVariables(CCVariable<double>& rho_micro,
                                  CCVariable<double>& rho_CC,
                                  CCVariable<double>& temp,
                                  CCVariable<Vector>& vel_CC,
                                  int numMatls,
                                  const Patch* patch)
{ 
  // initialize to -9 so bullet proofing will catch it any cell that
  // isn't initialized
  vel_CC.initialize(Vector(0.,0.,0.));
  rho_micro.initialize(-9.0);
  rho_CC.initialize(-9.0);
  temp.initialize(-9.0);
  Vector dx = patch->dCell();
  
  for(int obj=0; obj<(int)d_geom_objs.size(); obj++){
   GeometryPieceP piece = d_geom_objs[obj]->getPiece();
   Box b1 = piece->getBoundingBox();
   //Box b2 = patch->getBox();
   //Box b = b1.intersect(b2);
   // Find the bounds of a region a little bigger than the piece's BBox.
   Point b1low(b1.lower().x()-3.*dx.x(),b1.lower().y()-3.*dx.y(),
                                        b1.lower().z()-3.*dx.z());
   Point b1up(b1.upper().x()+3.*dx.x(),b1.upper().y()+3.*dx.y(),
                                        b1.upper().z()+3.*dx.z());
   
   IntVector ppc = d_geom_objs[obj]->getNumParticlesPerCell();
   Vector dxpp    = patch->dCell()/ppc;
   Vector dcorner = dxpp*0.5;
   double totalppc = ppc.x()*ppc.y()*ppc.z();

  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
     Point lower = patch->nodePosition(*iter) + dcorner;
     int count = 0;
     for(int ix=0;ix < ppc.x(); ix++){
       for(int iy=0;iy < ppc.y(); iy++){
        for(int iz=0;iz < ppc.z(); iz++){
          IntVector idx(ix, iy, iz);
          Point p = lower + dxpp*idx;
          if(piece->inside(p))
            count++;
        }
       }
     }
  //__________________________________
  // For single materials with more than one object 
      if(numMatls == 1)  {
        if ( count > 0  && obj == 0) {
         // vol_frac_CC[*iter]= 1.0;
          vel_CC[*iter]     = d_geom_objs[obj]->getInitialVelocity();
          rho_micro[*iter]  = getInitialDensity();
          rho_CC[*iter]     = rho_micro[*iter] + d_TINY_RHO;
          temp[*iter]       = d_geom_objs[obj]->getInitialData("temperature");
        }

        if (count > 0 && obj > 0) {
         // vol_frac_CC[*iter]= 1.0;
          vel_CC[*iter]     = d_geom_objs[obj]->getInitialVelocity();
          rho_micro[*iter]  = getInitialDensity();
          rho_CC[*iter]     = rho_micro[*iter] + d_TINY_RHO;
          temp[*iter]       = d_geom_objs[obj]->getInitialData("temperature");
        } 
      }   
      if (numMatls > 1 ) {
        double vol_frac_CC= count/totalppc;       
        rho_micro[*iter]  = getInitialDensity();
        rho_CC[*iter]     = rho_micro[*iter] * vol_frac_CC + d_TINY_RHO;
        temp[*iter]       = 300.0;         
        Point pd = patch->cellPosition(*iter);
        if((pd.x() > b1low.x() && pd.y() > b1low.y() && pd.z() > b1low.z()) &&
           (pd.x() < b1up.x()  && pd.y() < b1up.y()  && pd.z() < b1up.z())){
            vel_CC[*iter]     = d_geom_objs[obj]->getInitialVelocity();
            temp[*iter]      = d_geom_objs[obj]->getInitialData("temperature");
        }    
      }    
    }  // Loop over domain
  }  // Loop over geom_objects
}

void 
MPMMaterial::initializeDummyCCVariables(CCVariable<double>& rho_micro,
                                        CCVariable<double>& rho_CC,
                                        CCVariable<double>& temp,
                                        CCVariable<Vector>& vel_CC,
                                        int ,
                                        const Patch* )
{ 
  vel_CC.initialize(Vector(0.,0.,0.));
  rho_micro.initialize(d_density);
  rho_CC.initialize(d_TINY_RHO);
  temp.initialize(d_troom);
}

