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

//  MPMMaterial.cc

#include <CCA/Components/MPM/Materials/MPMMaterial.h>

#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModelFactory.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/ErosionModel.h>
#include <CCA/Components/MPM/Materials/ParticleCreator/ParticleCreatorFactory.h>
#include <CCA/Components/MPM/Materials/ParticleCreator/ParticleCreator.h>
#include <CCA/Components/MPM/Materials/Diffusion/DiffusionModels/ScalarDiffusionModel.h>
#include <CCA/Components/MPM/Materials/Diffusion/ScalarDiffusionModelFactory.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/GeometryPiece/NullGeometryPiece.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <iostream>
#include <list>

#define d_TINY_RHO 1.0e-12 // also defined  ICE.cc and ICEMaterial.cc 

using namespace std;
using namespace Uintah;

// Standard Constructor
MPMMaterial::MPMMaterial(ProblemSpecP& ps, MaterialManagerP& ss,MPMFlags* flags,
                         const bool isRestart)
  : Material(ps), d_cm(0),  d_particle_creator(0)
{
  d_lb = scinew MPMLabel();
  // The standard set of initializations needed
  standardInitialization(ps,ss,flags,isRestart);
  
  d_cm->setMaterialManager(ss.get_rep());

  // Check to see which ParticleCreator object we need
  d_particle_creator = ParticleCreatorFactory::create(ps,this,flags);
}
//______________________________________________________________________
//
void
MPMMaterial::standardInitialization(ProblemSpecP& ps, 
                                    MaterialManagerP& ss,
                                    MPMFlags* flags, 
                                    const bool isRestart)

{
  // Follow the layout of the input file
  // Steps:
  // 1.  Determine the type of constitutive model and create it.
  // 2.  Create damage model.
  //
  // 3.  Determine if scalar diffusion is used and the type of
  //     scalar diffusion model and create it.
  //     Added for reactive flow component.
  // 4.  Get the general properties of the material such as
  //     density, thermal_conductivity, specific_heat.
  // 5.  Loop through all of the geometry pieces that make up a single
  //     geometry object.
  // 6.  Within the geometry object, assign the boundary conditions
  //     to the object.
  // 7.  Assign the velocity field.

  // Step 1 -- create the constitutive gmodel.
  bool ans;
  d_cm = ConstitutiveModelFactory::create(ps,flags, ans);
  set_pLocalizedComputed( ans );
  
  if(!d_cm){
    ostringstream desc;
    desc << "An error occured in the ConstitutiveModelFactory that has \n" 
         << " slipped through the existing bullet proofing. Please tell \n"
         << " either Jim, John or Todd "<< endl; 
    throw ParameterNotFound(desc.str(), __FILE__, __LINE__);
  }
  
  // Step 2 -- create the damage/erosion gmodel.
  d_damageModel = DamageModelFactory::create(ps,flags,ss.get_rep() );
  
  d_erosionModel = scinew ErosionModel( ps, flags, ss.get_rep() );

  // Step 3 -- check if scalar diffusion is used and
  // create the scalar diffusion model.
  if(flags->d_doScalarDiffusion){
    d_sdm = ScalarDiffusionModelFactory::create(ps,ss,flags);
  }else{
    d_sdm = nullptr;
  }
  
  

  // Step 4 -- get the general material properties

  ps->require("density",d_density);
  ps->require("thermal_conductivity",d_thermalConductivity);
  ps->require("specific_heat",d_specificHeat);
  
  // Assume the the centered specific heat is C_v
  d_Cv = d_specificHeat;

  // Set C_p = C_v if not C_p data are entered
  d_Cp = d_Cv;
  ps->get("C_p",d_Cp);

  d_troom = 294.0; d_tmelt = 295.0e10;
  ps->get("room_temp", d_troom);
  ps->get("melt_temp", d_tmelt);

  // This is currently only used in the implicit code, but should
  // be put to use in the explicit code as well.
  d_is_rigid=false;
  ps->get("is_rigid", d_is_rigid);

  // This is used for the autocycleflux boundary conditions
  d_do_conc_reduction = false;
  ps->get("do_conc_reduction", d_do_conc_reduction);

  // Step 5 -- Loop through all of the pieces in this geometry object
  //int piece_num = 0;
  list<GeometryObject::DataItem> geom_obj_data;
  geom_obj_data.push_back(GeometryObject::DataItem("res",                    GeometryObject::IntVector));
  geom_obj_data.push_back(GeometryObject::DataItem("temperature",            GeometryObject::Double));
  geom_obj_data.push_back(GeometryObject::DataItem("velocity",               GeometryObject::Vector));
  geom_obj_data.push_back(GeometryObject::DataItem("affineTransformation_A0",GeometryObject::Vector));
  geom_obj_data.push_back(GeometryObject::DataItem("affineTransformation_A1",GeometryObject::Vector));
  geom_obj_data.push_back(GeometryObject::DataItem("affineTransformation_A2",GeometryObject::Vector));
  geom_obj_data.push_back(GeometryObject::DataItem("affineTransformation_b", GeometryObject::Vector));
  geom_obj_data.push_back(GeometryObject::DataItem("volumeFraction",         GeometryObject::Double));
  if(flags->d_with_color){
    geom_obj_data.push_back(GeometryObject::DataItem("color",                GeometryObject::Double));
  } 

  // ReactiveFlow Diffusion Component
  if(flags->d_doScalarDiffusion){
    geom_obj_data.push_back(GeometryObject::DataItem("concentration", GeometryObject::Double));
  }

  if(flags->d_withGaussSolver){
    std::cout << "************With Gauss Solver***********" << std::endl;
    geom_obj_data.push_back(GeometryObject::DataItem("pos_charge_density", GeometryObject::Double));
    geom_obj_data.push_back(GeometryObject::DataItem("neg_charge_density", GeometryObject::Double));
    geom_obj_data.push_back(GeometryObject::DataItem("permittivity", GeometryObject::Double));
  }

  if(!isRestart){
    for (ProblemSpecP geom_obj_ps = ps->findBlock("geom_object");
         geom_obj_ps != nullptr; 
         geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

     vector<GeometryPieceP> pieces;
     GeometryPieceFactory::create(geom_obj_ps, pieces);

     GeometryPieceP mainpiece;
     if(pieces.size() == 0){
       throw ParameterNotFound("No piece specified in geom_object",
                                __FILE__, __LINE__);
     }
     else if(pieces.size() > 1){
       mainpiece = scinew UnionGeometryPiece(pieces);
     }
     else {
       mainpiece = pieces[0];
     }

     GeometryObject* obj = scinew GeometryObject(mainpiece, geom_obj_ps, geom_obj_data);

     d_geom_objs.push_back( obj );
    }
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
  delete d_cm;
  delete d_damageModel;
  delete d_erosionModel;
  delete d_particle_creator;

  for (int i = 0; i<(int)d_geom_objs.size(); i++) {
    delete d_geom_objs[i];
  }
  if(d_sdm){
    delete d_sdm;
  }
}

void MPMMaterial::registerParticleState( std::vector<std::vector<const VarLabel* > > &PState,
                                         std::vector<std::vector<const VarLabel* > > &PState_preReloc )
{
  PState.push_back         (d_particle_creator->returnParticleState());
  PState_preReloc.push_back(d_particle_creator->returnParticleStatePreReloc());
}

void MPMMaterial::deleteGeomObjects()
{
  for (int i = 0; i<(int)d_geom_objs.size(); i++) {
    delete d_geom_objs[i];
  }
  d_geom_objs.clear();
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

  d_cm->outputProblemSpec(mpm_ps);
  d_damageModel->outputProblemSpec(mpm_ps);
  d_erosionModel->outputProblemSpec(mpm_ps);
  
  if(getScalarDiffusionModel()){
    d_sdm->outputProblemSpec(mpm_ps);
  }

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

DamageModel* MPMMaterial::getDamageModel() const
{
  return d_damageModel;
}

ErosionModel* MPMMaterial::getErosionModel() const
{
  return d_erosionModel;
}

// is pLocalizedMPM computed upstream of damage models?
void
MPMMaterial::set_pLocalizedComputed( const bool ans ){
  d_pLocalizedComputed = ans;
}

bool
MPMMaterial::is_pLocalizedPreComputed() const
{
  return d_pLocalizedComputed;
}

ScalarDiffusionModel* MPMMaterial::getScalarDiffusionModel() const
{
  return d_sdm;
}

particleIndex MPMMaterial::createParticles(
                                  CCVariable<int>& cellNAPID,
                                  const Patch* patch,
                                  DataWarehouse* new_dw)
{
  return d_particle_creator->createParticles(this,cellNAPID,
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

double MPMMaterial::getSpecificHeat() const
{
  return d_specificHeat;
}

double MPMMaterial::getThermalConductivity() const
{
  return d_thermalConductivity;
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
                                  CCVariable<double>& vol_frac_CC,
                                  const Patch* patch)
{ 
  // initialize to -9 so bullet proofing will catch it any cell that
  // isn't initialized
  vel_CC.initialize(Vector(0.,0.,0.));
  rho_micro.initialize(-9.0);
  rho_CC.initialize(-9.0);
  temp.initialize(-9.0);
  vol_frac_CC.initialize(0.0);
  Vector dx = patch->dCell();
  
  for(int obj=0; obj<(int)d_geom_objs.size(); obj++){
    GeometryPieceP piece = d_geom_objs[obj]->getPiece();
    IntVector ppc = d_geom_objs[obj]->getInitialData_IntVector("res");
    Vector dxpp    = patch->dCell()/ppc;
    Vector dcorner = dxpp*0.5;
    double totalppc = ppc.x()*ppc.y()*ppc.z();
    
    // Find the bounds of a region a little bigger than the piece's BBox.
    Box bb = piece->getBoundingBox();
    
    Point bb_low( bb.lower().x() - 3.0*dx.x(),
                  bb.lower().y() - 3.0*dx.y(),
                  bb.lower().z() - 3.0*dx.z() );

    Point bb_up( bb.upper().x() + 3.0*dx.x(),
                 bb.upper().y() + 3.0*dx.y(),
                 bb.upper().z() + 3.0*dx.z() );
    

    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      
      Point lower = patch->nodePosition(*iter) + dcorner;
      int count = 0;
      
      for(int ix=0;ix < ppc.x(); ix++){
        for(int iy=0;iy < ppc.y(); iy++){
          for(int iz=0;iz < ppc.z(); iz++){
            IntVector idx(ix, iy, iz);
            Point p = lower + dxpp*idx;
            if(piece->inside(p, true)){
              count++;
            }
          }
        }
      }

      double ups_volFrac = d_geom_objs[obj]->getInitialData_double("volumeFraction");
      if( ups_volFrac == -1.0 ) {    
        vol_frac_CC[c] += count/totalppc;  // there can be contributions from multiple objects 
      } else {
        vol_frac_CC[c] = ups_volFrac * count/(totalppc);
      }
      
      rho_micro[c]  = getInitialDensity();
      rho_CC[c]     = rho_micro[c] * vol_frac_CC[c] + d_TINY_RHO;     
 
      // these values of temp_CC and vel_CC are only used away from the mpm objects
      // on the first timestep in interpolateNC_CC_0.  We just need reasonable values
      temp[c]       = 300.0;
      
      Point pd = patch->cellPosition(c);
      
      bool inside_bb_lo = (pd.x() > bb_low.x() && pd.y() > bb_low.y() && pd.z() > bb_low.z());
      bool inside_bb_up = (pd.x() < bb_up.x()  && pd.y() < bb_up.y()  && pd.z() < bb_up.z() );
      
      // warning:  If two objects share the same cell then the last object sets the values.
      //  This isn't a big deal since interpolateNC_CC_0 will only use these values on the first
      //  timmestep
      if( inside_bb_lo && inside_bb_up){
        vel_CC[c] = d_geom_objs[obj]->getInitialData_Vector("velocity");
        temp[c]   = d_geom_objs[obj]->getInitialData_double("temperature");
      }                
      
    }  // Loop over cells
  }  // Loop over geom_objects
}
//______________________________________________________________________
//
void 
MPMMaterial::initializeDummyCCVariables(CCVariable<double>& rho_micro,
                                        CCVariable<double>& rho_CC,
                                        CCVariable<double>& temp,
                                        CCVariable<Vector>& vel_CC,
                                        CCVariable<double>& vol_frac_CC,
                                        const Patch* )
{ 
  vel_CC.initialize(Vector(0.,0.,0.));
  rho_micro.initialize(d_density);
  rho_CC.initialize(d_TINY_RHO);
  temp.initialize(d_troom);
  vol_frac_CC.initialize(1.0);
}
