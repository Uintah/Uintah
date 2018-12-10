/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/ICE/EOS/EquationOfState.h>
#include <CCA/Components/ICE/EOS/EquationOfStateFactory.h>

#include <CCA/Components/ICE/Materials/ICEMaterial.h>
#include <CCA/Components/ICE/SpecificHeatModel/SpecificHeatFactory.h>

#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Geometry/IntVector.h>
#include <Core/GeometryPiece/FileGeometryPiece.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>

#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpec.h>

//#define d_TINY_RHO 1.0e-12 // also defined ICE.cc and MPMMaterial.cc 

using namespace std;
using namespace Uintah;

// Constructor
ICEMaterial::ICEMaterial( ProblemSpecP     & ps,
                          MaterialManagerP & materialManager,
                          const bool         isRestart ) :
  Material( ps )
{
  //__________________________________
  //  Create the different Models for this material
  d_eos = EquationOfStateFactory::create(ps);
  if( !d_eos ) {
    throw ParameterNotFound("ICE: No EOS specified", __FILE__, __LINE__);
  }

  ProblemSpecP cvModel_ps = ps->findBlock("SpecificHeatModel");
  d_cvModel = nullptr;
  if( cvModel_ps != nullptr ) {
    proc0cout << "Creating Specific heat model." << endl;
    d_cvModel = SpecificHeatFactory::create( ps );
  }
  
  //__________________________________
  // Thermodynamic Transport Properties
  ps->require("thermal_conductivity",d_thermalConductivity);
  ps->require("specific_heat",       d_specificHeat);
  ps->require("dynamic_viscosity",   d_viscosity);
  ps->require("gamma",               d_gamma);
  ps->getWithDefault("tiny_rho",     d_tiny_rho,1.e-12);

  //__________________________________
  //  Misc. Flags
  d_isSurroundingMatl = false;
  d_includeFlowWork   = true;  
  ps->get("isSurroundingMatl",  d_isSurroundingMatl);
  ps->get("includeFlowWork",    d_includeFlowWork);

  //__________________________________
  // Loop through all of the pieces in this geometry object
  int piece_num = 0;

  list<GeometryObject::DataItem> geom_obj_data;
  geom_obj_data.push_back(GeometryObject::DataItem("res",        GeometryObject::IntVector));
  geom_obj_data.push_back(GeometryObject::DataItem("temperature",GeometryObject::Double));
  geom_obj_data.push_back(GeometryObject::DataItem("pressure",   GeometryObject::Double));
  geom_obj_data.push_back(GeometryObject::DataItem("density",    GeometryObject::Double)); 
  try{
    geom_obj_data.push_back(GeometryObject::DataItem("volumeFraction", GeometryObject::Double));
   }
   catch(...)
   {}
  geom_obj_data.push_back(GeometryObject::DataItem("velocity",   GeometryObject::Vector));

  if(!isRestart){
    for (ProblemSpecP geom_obj_ps=ps->findBlock("geom_object"); geom_obj_ps != nullptr; geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

      vector<GeometryPieceP> pieces;
      GeometryPieceFactory::create(geom_obj_ps, pieces);

      GeometryPieceP mainpiece;
      if(pieces.size() == 0){
        throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
      }
      else if(pieces.size() > 1){
        mainpiece = scinew UnionGeometryPiece(pieces);
      }
      else {
        mainpiece = pieces[0];
      }

      piece_num++;
      d_geom_objs.push_back(scinew GeometryObject(mainpiece, geom_obj_ps, geom_obj_data));
    }
  }
}
//__________________________________
// Destructor
ICEMaterial::~ICEMaterial()
{
  delete d_eos;
  
  if( d_cvModel ){
    delete d_cvModel;
  }
  
  for (int i = 0; i< (int)d_geom_objs.size(); i++) {
    delete d_geom_objs[i];
  }
}
//______________________________________________________________________
//
ProblemSpecP ICEMaterial::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP ice_ps = Material::outputProblemSpec(ps);

  d_eos->outputProblemSpec(ice_ps);
  ice_ps->appendElement("thermal_conductivity",d_thermalConductivity);
  ice_ps->appendElement("specific_heat",       d_specificHeat);
  if( d_cvModel != nullptr ) {
    d_cvModel->outputProblemSpec( ice_ps );
  }
  ice_ps->appendElement("dynamic_viscosity",   d_viscosity);
  ice_ps->appendElement("gamma",               d_gamma);
  ice_ps->appendElement("isSurroundingMatl",   d_isSurroundingMatl);
  ice_ps->appendElement("includeFlowWork",     d_includeFlowWork);
  ice_ps->appendElement("tiny_rho",            d_tiny_rho);
    
  for (vector<GeometryObject*>::const_iterator it = d_geom_objs.begin();
       it != d_geom_objs.end(); it++) {
    (*it)->outputProblemSpec(ice_ps);
  }

  return ice_ps;
}
//__________________________________
//
EquationOfState * ICEMaterial::getEOS() const
{
  return d_eos;
}

SpecificHeat *ICEMaterial::getSpecificHeatModel() const
{
  return d_cvModel;
}

double ICEMaterial::getGamma() const
{
  return d_gamma;
}

double ICEMaterial::getTinyRho() const
{
  return d_tiny_rho;
}

double ICEMaterial::getViscosity() const
{
  return d_viscosity;
}

bool ICEMaterial::isSurroundingMatl() const
{
  return d_isSurroundingMatl;
}

bool ICEMaterial::getIncludeFlowWork() const
{
  return d_includeFlowWork;
}

double ICEMaterial::getSpecificHeat() const
{
  return d_specificHeat;
}

double ICEMaterial::getThermalConductivity() const
{
  return d_thermalConductivity;
}

double ICEMaterial::getInitialDensity() const
{
  return d_geom_objs[0]->getInitialData_double("density");
}

/* --------------------------------------------------------------------- 
 Function~  ICEMaterial::initializeCells--
 Purpose~ Initialize material dependent variables 
 Notes:  This is a tricky little routine.  ICE needs rho_micro, Temp_CC
 speedSound defined everywhere for all materials even if the mass is 0.0.
 
 We need to accomodate the following scenarios where the number designates
 a material and * represents a high temperature
 ____________________           ____________________
 | 1  | 1  | 1  | 1  |          | 1  | 1  | 1  | 1  |
 |____|____|____|____|          |____|____|____|____|
 | 1  | 1  | 1* | 1  |          | 1  | 1  | 2* | 1  |
 |____|____|____|____|          |____|____|____|____|
 | 1  | 1  | 1  | 1  |          | 1  | 1  | 1  | 1  |
 |____|____|____|____|          |____|____|____|____|=
_____________________________________________________________________*/
void ICEMaterial::initializeCells(CCVariable<double>& rho_micro,
                                  CCVariable<double>& rho_CC,
                                  CCVariable<double>& temp,
                                  CCVariable<double>& speedSound,
                                  CCVariable<double>& vol_frac_CC,
                                  CCVariable<Vector>& vel_CC,
                                  CCVariable<double>& press_CC,
                                  int numMatls,
                                  const Patch* patch,
                                  DataWarehouse* new_dw)
{
  CCVariable<int> IveBeenHere;
  new_dw->allocateTemporary(IveBeenHere, patch);
  
  // Zero the arrays so they don't get wacky values
  vel_CC.initialize(Vector(0.,0.,0.));
  rho_micro.initialize(0.);
  rho_CC.initialize(0.);
  temp.initialize(0.);
  vol_frac_CC.initialize(0.);
  speedSound.initialize(0.);
  IveBeenHere.initialize(-9);

  for(int obj=0; obj<(int)d_geom_objs.size(); obj++){
    GeometryPieceP piece = d_geom_objs[obj]->getPiece();
    // Box b1 = piece->getBoundingBox();
    // Box b2 = patch->getBox();

    FileGeometryPiece *fgp = dynamic_cast<FileGeometryPiece*>(piece.get_rep());

    if(fgp){
      // For some reason, if I call readPoints here, I get two copies of the
      // points, so evidently the fgp is being carried over from MPMMaterial
  //    fgp->readPoints(patch->getID());
      int numPts = fgp->returnPointCount();
      vector<Point>* points = fgp->getPoints();
      if(numMatls > 2)  {
        cerr << "ERROR!!!\n";
        cerr << "File Geometry Piece with ICE only supported for one ice matl.\n";
        exit(1);
      }

      // First initialize all variables everywhere.
      for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        vol_frac_CC[c]= 1.0;
        press_CC[c]   = d_geom_objs[obj]->getInitialData_double("pressure");
        vel_CC[c]     = d_geom_objs[obj]->getInitialData_Vector("velocity");
        rho_micro[c]  = d_geom_objs[obj]->getInitialData_double("density");
        rho_CC[c]     = rho_micro[*iter] + d_tiny_rho*rho_micro[*iter];
        temp[c]       = d_geom_objs[obj]->getInitialData_double("temperature");
        IveBeenHere[c]= 1;
      }

      IntVector ppc = d_geom_objs[obj]->getInitialData_IntVector("res");
      double ppc_tot = ppc.x()*ppc.y()*ppc.z();
      cout << "ppc_tot = " << ppc_tot << endl;
      cout << "numPts = " << numPts << endl;
      IntVector cell_idx;
      
      for (int ii = 0; ii < numPts; ++ii) {
        Point p = points->at(ii);
        patch->findCell(p,cell_idx);
        vol_frac_CC[cell_idx] -= 1./ppc_tot;
        IveBeenHere[cell_idx]= obj;
      }

      for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        rho_CC[c] = rho_micro[c] * vol_frac_CC[c] + d_tiny_rho*rho_micro[c];
      }
    } else {

      IntVector ppc = d_geom_objs[obj]->getInitialData_IntVector("res");
      Vector dxpp     = patch->dCell()/ppc;
      Vector dcorner  = dxpp*0.5;
      double totalppc = ppc.x()*ppc.y()*ppc.z();

      for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        Point lower = patch->nodePosition(c) + dcorner;
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
          if ( count > 0 ) {
            vol_frac_CC[c]= 1.0;
            press_CC[c]   = d_geom_objs[obj]->getInitialData_double("pressure");
            vel_CC[c]     = d_geom_objs[obj]->getInitialData_Vector("velocity");
            rho_micro[c]  = d_geom_objs[obj]->getInitialData_double("density");
            rho_CC[c]     = rho_micro[c] + d_tiny_rho*rho_micro[c];
            temp[c]       = d_geom_objs[obj]->getInitialData_double("temperature");
            IveBeenHere[c]= 1;
          }
        }   

        //__________________________________
        //  Multiple matls
        if (numMatls > 1 ) {

          double ups_volFrac = d_geom_objs[obj]->getInitialData_double("volumeFraction");
          if( ups_volFrac == -1.0 ) {    
            vol_frac_CC[c] += count/totalppc;  // there can be contributions from multiple objects 
          } else {
            vol_frac_CC[c] = ups_volFrac * count/(totalppc);
          }

          if(IveBeenHere[c] == -9){
            // This cell hasn't been hit for this matl yet so set values
            // to ensure that everything is set to something everywhere
            vel_CC[c]     = d_geom_objs[obj]->getInitialData_Vector("velocity");
            rho_micro[c]  = d_geom_objs[obj]->getInitialData_double("density");
            rho_CC[c]     = rho_micro[c] * vol_frac_CC[c] + d_tiny_rho*rho_micro[c];
            temp[c]       = d_geom_objs[obj]->getInitialData_double("temperature");
            IveBeenHere[c]= obj; 
          }
          if(IveBeenHere[c] != -9 && count > 0){
            // This cell HAS been hit but another object has values to
            // override it, possibly in a cell that was just set by default
            // in the above section.
            press_CC[c]   = d_geom_objs[obj]->getInitialData_double("pressure");
            vel_CC[c]     = d_geom_objs[obj]->getInitialData_Vector("velocity");
            rho_micro[c]  = d_geom_objs[obj]->getInitialData_double("density");
            rho_CC[c]     = rho_micro[c] * vol_frac_CC[c] + d_tiny_rho*rho_micro[c];
            temp[c]       = d_geom_objs[obj]->getInitialData_double("temperature");
            IveBeenHere[c]= obj; 
          }
          if(IveBeenHere[c] != -9 && count == 0){
            // This cell has been initialized, the current object doesn't
            // occupy this cell, so don't screw with it.  All bases are
            // covered.
          }
        }    
      }  // Loop over domain
    }
  }  // Loop over geom_objects
}

