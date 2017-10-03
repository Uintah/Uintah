/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/ICE/customInitialize.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MersenneTwister.h>

using namespace std;

using namespace Uintah;
namespace Uintah {
/*_____________________________________________________________________ 
 Function~  customInitialization_problemSetup--
 Purpose~   This function reads in the input parameters for each of the
            customized initialization.  All of the inputs are stuffed into
            the customInitialize_basket.
_____________________________________________________________________*/
void customInitialization_problemSetup( const ProblemSpecP& cfd_ice_ps,
                                        customInitialize_basket* cib,
                                        GridP& grid)
{
  //__________________________________
  //  search the ICE problem spec for 
  // custom initialization inputs
  ProblemSpecP c_init_ps= cfd_ice_ps->findBlock("customInitialization");
  // defaults
  cib->which = "none"; 
  cib->doesComputePressure = false;
  
  if(c_init_ps){

    //_______________________________________________
    // multiple vortices section
    ProblemSpecP vortices_ps= c_init_ps->findBlock("vortices");    
    if( vortices_ps ) {
      cib->vortex_inputs = scinew vortices();
      cib->which = "vortices";
      cib->doesComputePressure = true;
      
      for( ProblemSpecP vortex_ps = vortices_ps->findBlock( "vortex" ); vortex_ps != nullptr; vortex_ps = vortex_ps->findNextBlock( "vortex" ) ) {
        Point origin;
        double strength;
        double radius;
        if(vortex_ps){
          vortex_ps->require("origin",   origin);
          vortex_ps->require("strength", strength);
          vortex_ps->require("radius",   radius);
          cib->vortex_inputs->origin.push_back(origin);
          cib->vortex_inputs->strength.push_back(strength);
          cib->vortex_inputs->radius.push_back(radius);
        }
      }
    }  // multiple vortices
    
    
    ProblemSpecP gaussTemp_ps= c_init_ps->findBlock("gaussianTemperature");
    if(gaussTemp_ps){
      cib->which = "gaussianTemp";
      cib->gaussTemp_inputs = scinew gaussTemp();
      double spread_x;
      double spread_y;
      Point  origin;
      double amp;
      gaussTemp_ps->require("amplitude", amp); 
      gaussTemp_ps->require("origin",    origin);      
      gaussTemp_ps->require("spread_x",  spread_x);  
      gaussTemp_ps->require("spread_y",  spread_y);  
      cib->gaussTemp_inputs->amplitude = amp;
      cib->gaussTemp_inputs->origin    = origin;
      cib->gaussTemp_inputs->spread_x  = spread_x;
      cib->gaussTemp_inputs->spread_y  = spread_y;
    }
    
    //_______________________________________________
    //  method of manufactured solutions
    ProblemSpecP mms_ps= c_init_ps->findBlock("manufacturedSolution");
    if(mms_ps) {
      
      map<string,string> whichmms;
      mms_ps->getAttributes(whichmms);
      
      cib->which = whichmms["type"];
      
      if(cib->which == "mms_1") {
        cib->doesComputePressure = true;
        cib->mms_inputs = scinew mms();
        mms_ps->require("A", cib->mms_inputs->A);
      }
      if(cib->which == "mms_3") {
        cib->doesComputePressure = false;
        cib->mms_inputs = scinew mms();
        mms_ps->require("angle", cib->mms_inputs->angle);
      }
    } 
    
    //_______________________________________________
    //  2D counterflow in the x & y plane
    ProblemSpecP cf_ps= c_init_ps->findBlock("counterflow");
    if(cf_ps) {
      cib->which = "counterflow";
      cib->doesComputePressure = true;
      cib->counterflow_inputs = scinew counterflow();
      cf_ps->require("strainRate",   cib->counterflow_inputs->strainRate);
      cf_ps->require("referenceCell", cib->counterflow_inputs->refCell);
      
      grid->getLength(cib->counterflow_inputs->domainLength, "minusExtraCells");
    }

    //_______________________________________________
    //  Channel Flow initialized with powerlaw velocity profile
    // and variance. the x & y plane
    ProblemSpecP pl_ps= c_init_ps->findBlock("powerLawProfile");
    if(pl_ps) {
      cib->which = "powerLaw";
      cib->doesComputePressure = true;

      cib->powerLaw_inputs = scinew powerLaw();
      powerLaw* inputs = cib->powerLaw_inputs;   // for code readability
            
      // geometry: computational domain
      BBox b;
      grid->getInteriorSpatialRange(b);
      cib->powerLaw_inputs->gridMin = b.min();
      cib->powerLaw_inputs->gridMax = b.max();
      
      pl_ps -> require( "U_infinity",        inputs->U_infinity   );    
      pl_ps -> require( "exponent",          inputs->exponent    );     
      pl_ps -> require( "verticalDirection", inputs->verticalDir );

      Vector tmp = b.max() - b.min();
      double maxHeight = tmp[ inputs->verticalDir ];   // default value

      pl_ps -> get( "maxHeight",             maxHeight   );
      Vector lo = b.min().asVector();
      inputs->maxHeight = maxHeight - lo[ inputs->verticalDir ];

      //__________________________________
      //  Add variance to the velocity profile
      inputs->addVariance = false;
      ProblemSpecP var_ps = pl_ps->findBlock("variance");
      if (var_ps) {
        inputs->addVariance = true;
        var_ps -> get( "C_mu",        inputs->C_mu );
        var_ps -> get( "frictionVel", inputs->u_star );
      }
    }  // powerLaw inputs    
  }
}
/*_____________________________________________________________________ 
 Function~  customInitialization()--
 Purpose~  overwrite the initialization from the ups file.
_____________________________________________________________________*/ 
void customInitialization(const Patch* patch,
                          CCVariable<double>& rho_CC,
                          CCVariable<double>& temp_CC,
                          CCVariable<Vector>& vel_CC,
                          CCVariable<double>& press_CC,
                          ICEMaterial* ice_matl,
                          const customInitialize_basket* cib)
{
  //_______________________________________________
  //  multiple vortices
  //See "Boundary Conditions for Direct Simulations of Compressible Viscous
  //     Flows" by Poinsot & LeLe pg 121
  if (cib->which == "vortices"){
    for (int i = 0; i<(int) cib->vortex_inputs->origin.size(); i++) {
      
      Point origin = cib->vortex_inputs->origin[i]; // vortex origin
      double C1 = cib->vortex_inputs->strength[i];  // vortex strength
      double R = cib->vortex_inputs->radius[i];     // vortex radius
      double R_sqr = R * R;
      double p_ref  = 101325;        // assumed reference pressure
     
      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        Point pt = patch->cellPosition(c);

        double x = pt.x() - origin.x();
        double y = pt.y() - origin.y();
        double r = sqrt( x*x + y*y);

        if(r <= R ) {
          double A = exp( -((x*x) + (y*y) )/(2 * R_sqr)); 

          press_CC[c] = p_ref + rho_CC[c] * ((C1*C1)/R_sqr) * A; 

          double uvel = vel_CC[c].x();
          double vvel = vel_CC[c].y();

          vel_CC[c].x(uvel - ((C1 * y)/(rho_CC[c] * R_sqr) ) * A);
          vel_CC[c].y(vvel + ((C1 * x)/(rho_CC[c] * R_sqr) ) * A);
        }
      } 
    }  // loop
  } // vortices
  //_______________________________________________
  // gaussian Temperature
  if(cib->which == "gaussianTemp"){
  
    double amp = cib->gaussTemp_inputs->amplitude;
    Point origin     = cib->gaussTemp_inputs->origin;
    double spread_x  = cib->gaussTemp_inputs->spread_x;
    double spread_y  = cib->gaussTemp_inputs->spread_y;
    
    double x0 = origin.x();
    double y0 = origin.y();
    
    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;
      Point pt = patch->cellPosition(c);
      double x = pt.x();
      double y = pt.y();

      double a = ( (x-x0) * (x-x0) )/ (2*spread_x*spread_x + 1e-100);
      double b = ( (y-y0) * (y-y0) )/ (2*spread_y*spread_y + 1e-100);
      
      double Z = amp*exp(-(a + b));
      temp_CC[c] = 300 + Z;
    }
  } 
  
  //_______________________________________________
  // 2D counterflow flowfield
  // See:  "Characteristic Boundary conditions for direct simulations
  //        of turbulent counterflow flames" by Yoo, Wang Trouve and IM
  //        Combustion Theory and Modelling, Vol 9. No 4., Nov. 2005, 617-646
  if(cib->which == "counterflow"){
  
    double strainRate   = cib->counterflow_inputs->strainRate;
    Vector domainLength = cib->counterflow_inputs->domainLength;
    IntVector refCell   = cib->counterflow_inputs->refCell;
    
    double u_ref   = vel_CC[refCell].x();
    double v_ref   = vel_CC[refCell].y();
    double p_ref   = 101325;
    double rho_ref = rho_CC[refCell];
    
    for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;
      Point pt = patch->cellPosition(c);
      double x = pt.x();
      double y = pt.y();

      double u = -strainRate * (x - domainLength.x()/2.0);
      double v =  strainRate * (y - domainLength.y()/2.0);
      // for a purely converging flow 
      //v = -strainRate * (y - domainLength.y()/2.0);
      
      vel_CC[c].x( u );
      vel_CC[c].y( v );
 
      press_CC[c] = p_ref
                  + 0.5 * rho_ref * (u_ref * u_ref + v_ref * v_ref)
                  - 0.5 * rho_ref * (u * u + v * v);
    }
  }
  
  
  //_______________________________________________
  // method of manufactured solution 1
  // See:  "A non-trival analytical solution to the 2d incompressible
  //        Navier-Stokes equations" by Randy McDermott
  if(cib->which == "mms_1"){
    double t = 0.0; 
    double A = cib->mms_inputs->A;
    double nu = ice_matl->getViscosity();
    double cv = ice_matl->getSpecificHeat();
    double gamma = ice_matl->getGamma();
    double p_ref = 101325;
    
    for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;
      Point pt = patch->cellPosition(c);
      double x = pt.x(); 
      double y = pt.y();
    
      press_CC[c] = p_ref - A*A/4.0 * exp(-4.0*nu*t)
                   *( cos(2.0*(x-t)) + cos(2.0*(y-t)) );

      vel_CC[c].x( 1.0 - A * cos(x-t) * sin(y -t) * exp(-2.0*nu*t));
      vel_CC[c].y( 1.0 + A * sin(x-t) * cos(y -t) * exp(-2.0*nu*t));
      
      // back out temperature from the perfect gas law
      temp_CC[c]= press_CC[c]/ ( (gamma - 1.0) * cv * rho_CC[c] );
    }
  } // mms_1
  
  //_______________________________________________
  // method of manufactured solution 2
  // See:  "Code Verification by the MMS SAND2000-1444
  // This is a steady state solution
  // This has not been verifed.
  
  if(cib->which == "mms_2"){
    double cv = ice_matl->getSpecificHeat();
    double gamma = ice_matl->getGamma();
    
    for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;
      Point pt = patch->cellPosition(c);
      double x = pt.x(); 
      double y = pt.y();
    
      press_CC[c] = rho_CC[c] * ( -0.5 * (exp(2*x) +  exp(2*y) )
                                  + exp(x+y) * cos(x+y));

      vel_CC[c].x( exp(x) * sin(y) + exp(y) * sin(x));
      vel_CC[c].y( exp(x) * cos(y) - exp(y) * cos(x));
      
      // back out temperature from the perfect gas law
      temp_CC[c]= press_CC[c]/ ( (gamma - 1.0) * cv * rho_CC[c] );
    }
  } // mms_2
  
  //_______________________________________________
  // method of manufactured solution 3
  // See:  "Small-scale structure of the Taylor-Green vortex", M. Brachet et al.
  //       J. Fluid Mech, vol. 130, pp. 411-452, 1983.
  //   These equations are slightly different than eq. 1.1 in reference and have
  //   been provided by James Sutherland
  
  if(cib->which == "mms_3"){
    double angle = cib->mms_inputs->angle;
    double A = ( 2.0/sqrt(3) ) ; 
    double B = (2.0 * M_PI/3.0);
    
    for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;
      Point pt = patch->cellPosition(c);
      double x = pt.x(); 
      double y = pt.y();
      double z = pt.z();

      vel_CC[c].x( A * sin(angle + B) * cos(y) * cos(z) * sin(x));
      vel_CC[c].y( A * sin(angle - B) * cos(x) * cos(z) * sin(y));
      vel_CC[c].z( A * sin(angle)     * cos(x) * cos(y) * sin(z));
    }
  } // mms_3

  //_______________________________________________
  //  power law velocity profile + variance 
  // u = U_infinity * pow( h/height )^n
  if(cib->which == "powerLaw"){
    int vDir          =  cib->powerLaw_inputs->verticalDir;
    double d          =  cib->powerLaw_inputs->gridMin(vDir);
    double gridHeight =  cib->powerLaw_inputs->gridMax(vDir);
    double height     =  cib->powerLaw_inputs->maxHeight;
    Vector U_infinity =  cib->powerLaw_inputs->U_infinity;
    double n          =  cib->powerLaw_inputs->exponent;
    const Level* level = patch->getLevel();
    
    //std::cout << "     height: " << height << " exponent: " << n << " U_infinity: " << U_infinity 
    //     << " nDir: " << nDir << " vDir: " << vDir << endl;

    
    for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++) {
      IntVector c = *iter; 

      Point here   = level->getCellPosition(c);
      double h     = here.asVector()[vDir] ;

      vel_CC[c]    = U_infinity;             // set the components that are not normal to the face           
      double ratio = (h - d)/height;
      ratio = Clamp(ratio,0.0,1.0);

      if( h > d && h < height){
        vel_CC[c] = U_infinity * pow(ratio, n);
      }else{                                // if height < h < gridHeight
        vel_CC[c] = U_infinity;
      }

      // Clamp edge/corner values 
      if( h < d || h > gridHeight ){
        vel_CC[c] = Vector(0,0,0);
      }
    } 

    //__________________________________
    //  Addition of a 'kick' or variance to the mean velocity profile
    //  This matches the Turbulent Kinetic Energy profile of 1/sqrt(C_u) * u_star^2 ( 1- Z/height)^2
    //   where:
    //          C_mu:     empirical constant
    //          u_star:  frictionVelocity
    //          Z:       height above the ground
    //          height:  Boundar layer height, assumed to be the domain height
    //
    //   TKE = 1/2 * (sigma.x^2 + sigma.y^2 + sigma.z^2)
    //    where sigma.x^2 = (1/N-1) * sum( u_mean - u)^2
    //
    //%  Reference: Castro, I, Apsley, D. "Flow and dispersion over topography;
    //             A comparison between numerical and Laboratory Data for 
    //             two-dimensional flows", Atmospheric Environment Vol. 31, No. 6
    //             pp 839-850, 1997.
    
    if (cib->powerLaw_inputs->addVariance ){ 
      MTRand mTwister;

      double gridHeight =  cib->powerLaw_inputs->gridMax(vDir); 
      double d          =  cib->powerLaw_inputs->gridMin(vDir);
      double inv_Cmu    = 1.0/cib->powerLaw_inputs->C_mu;
      double u_star2    = cib->powerLaw_inputs->u_star * cib->powerLaw_inputs->u_star;
      
      for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        
        Point here = level->getCellPosition(c);
        double z   = here.asVector()[vDir] ;
        
        double ratio = (z - d)/gridHeight;
        
        double TKE = inv_Cmu * u_star2 * pow( (1 - ratio),2 );
        
        // Assume that the TKE is evenly distrubuted between all three components of velocity
        // 1/2 * (sigma.x^2 + sigma.y^2 + sigma.z^2) = 3/2 * sigma^2
        
        const double variance = sqrt(0.66666 * TKE);
        
        //__________________________________
        // from the random number compute the new velocity knowing the mean velcity and variance
        vel_CC[c].x( mTwister.randNorm( vel_CC[c].x(), variance ) );
        vel_CC[c].y( mTwister.randNorm( vel_CC[c].y(), variance ) );
        vel_CC[c].z( mTwister.randNorm( vel_CC[c].z(), variance ) );
                
        // Clamp edge/c orner values 
        if(z < d || z > gridHeight ){
          vel_CC[c] = Vector(0,0,0);
        }
      }
    }  // add variance
  }
    
}
} // end uintah namespace
