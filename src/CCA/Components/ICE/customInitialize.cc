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

#include <CCA/Components/ICE/customInitialize.h>
#include <CCA/Components/ICE/Materials/ICEMaterial.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MersenneTwister.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/StringUtil.h>

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
  cib->doesComputePressure = false;

  if(c_init_ps){

    //_______________________________________________
    // multiple vortices section
    ProblemSpecP vortices_ps= c_init_ps->findBlock("vortices");
    if( vortices_ps ) {
      cib->vortex_vars = vortices();
      cib->whichMethod.push_back( "vortices" ) ;
      cib->doesComputePressure = true;

      for( ProblemSpecP vortex_ps = vortices_ps->findBlock( "vortex" ); vortex_ps != nullptr; vortex_ps = vortex_ps->findNextBlock( "vortex" ) ) {
        Point origin;
        double strength;
        double radius;
        string axis;

        if(vortex_ps){
          vortex_ps->require("origin",   origin);
          vortex_ps->require("strength", strength);
          vortex_ps->require("radius",   radius);
          vortex_ps->require( "axis",    axis);

          cib->vortex_vars.origin.push_back(   origin   );
          cib->vortex_vars.strength.push_back( strength );
          cib->vortex_vars.radius.push_back(   radius   );
          cib->vortex_vars.axis.push_back( string_toupper(axis) );  // upper case

        }
      }
    }  // multiple vortices

    //_______________________________________________
    // vortex pairs section
    ProblemSpecP vortexPairs_ps= c_init_ps->findBlock("vortexPairs");
    if( vortexPairs_ps ) {
      cib->vortexPairs_vars = vortexPairs();
      cib->whichMethod.push_back( "vortexPairs" ) ;

      double strength;
      double nPairs;
      string axis;

      vortexPairs_ps->require("strength",       strength);
      vortexPairs_ps->require("numVortexPairs", nPairs);
      vortexPairs_ps->require( "axis",          axis);

      cib->vortexPairs_vars.strength = strength;
      cib->vortexPairs_vars.nPairs   = nPairs;
      cib->vortexPairs_vars.axis     = string_toupper(axis);  // upper case

    }  // multiple vortices

    //__________________________________
    //
    ProblemSpecP gaussTemp_ps= c_init_ps->findBlock("gaussianTemperature");
    if(gaussTemp_ps){
      cib->whichMethod.push_back( "gaussianTemp" );
      cib->gaussTemp_vars = gaussTemp();

      gaussTemp_ps->require("amplitude", cib->gaussTemp_vars.amplitude);
      gaussTemp_ps->require("origin",    cib->gaussTemp_vars.origin);
      gaussTemp_ps->require("spread_x",  cib->gaussTemp_vars.spread_x);
      gaussTemp_ps->require("spread_y",  cib->gaussTemp_vars.spread_y);

    }

    //_______________________________________________
    //  method of manufactured solutions
    ProblemSpecP mms_ps= c_init_ps->findBlock("manufacturedSolution");

    if(mms_ps) {
      map<string,string> whichmms;
      mms_ps->getAttributes(whichmms);

      std::string whichMethod = whichmms["type"];

      cib->whichMethod.push_back( whichMethod );

      if( whichMethod == "mms_1" ) {
        cib->doesComputePressure = true;
        cib->mms_vars = mms();
        mms_ps->require("A", cib->mms_vars.A);
      }
      if( whichMethod == "mms_3" ) {
        cib->doesComputePressure = false;
        cib->mms_vars = mms();
        mms_ps->require("angle", cib->mms_vars.angle);
      }
    }

    //_______________________________________________
    //  2D counterflow in the x & y plane
    ProblemSpecP cf_ps= c_init_ps->findBlock("counterflow");

    if( cf_ps ) {
      cib->whichMethod.push_back( "counterflow" );
      cib->doesComputePressure = true;
      cib->counterflow_vars = counterflow();

      cf_ps->require("strainRate",    cib->counterflow_vars.strainRate);
      cf_ps->require("referenceCell", cib->counterflow_vars.refCell);

      grid->getLength(cib->counterflow_vars.domainLength, "minusExtraCells");
    }

    //_______________________________________________
    //  Channel Flow initialized with powerlaw velocity profile
    // and variance. the x & y plane
    ProblemSpecP pL_ps= c_init_ps->findBlock("powerLawProfile");
    if(pL_ps) {
      cib->whichMethod.push_back( "powerLaw" );
      cib->powerLaw_vars = powerLaw();

      // geometry: computational domain
      BBox b;
      grid->getInteriorSpatialRange(b);
      cib->powerLaw_vars.gridMin = b.min();
      cib->powerLaw_vars.gridMax = b.max();

      int vDir = -9;
      pL_ps->require( "U_infinity",        cib->powerLaw_vars.U_infinity );
      pL_ps->require( "exponent",          cib->powerLaw_vars.exponent );
      pL_ps->require( "verticalDirection", vDir );
      cib->powerLaw_vars.verticalDir = vDir;

      Vector tmp = b.max() - b.min();
      double maxHeight = tmp[ vDir ];   // default value

      pL_ps->get( "maxHeight",  maxHeight   );
      Vector lo = b.min().asVector();
      cib->powerLaw_vars.maxHeight = maxHeight - lo[ vDir ];

      //__________________________________
      //  Add variance to the velocity profile
      cib->powerLaw_vars.addVariance = false;
      ProblemSpecP var_ps = pL_ps->findBlock("variance");
      if (var_ps) {
        cib->powerLaw_vars.addVariance = true;
        var_ps->get( "C_mu",        cib->powerLaw_vars.C_mu );
        var_ps->get( "frictionVel", cib->powerLaw_vars.u_star );
      }
    }  // powerLaw inputs

    //_______________________________________________
    //  Channel Flow initialized with powerlaw velocity profile
    // and variance. the x & y plane
    ProblemSpecP pL2_ps= c_init_ps->findBlock("powerLawProfile2");
    if(pL2_ps) {
      cib->whichMethod.push_back( "powerLaw2" );
      cib->powerLaw2_vars = powerLaw2();

      int vDir = -9;
      int pDir = -9;
      double halfChannelHeight = -9;
      pL2_ps->require( "Re_tau",       cib->powerLaw2_vars.Re_tau );
      pL2_ps->require( "verticalDir",  vDir );
      pL2_ps->require( "principalDir", pDir );
      pL2_ps->require( "halfChannelHeight", halfChannelHeight );

      cib->powerLaw2_vars.verticalDir  = vDir;
      cib->powerLaw2_vars.principalDir = pDir;
      cib->powerLaw2_vars.halfChanHeight = halfChannelHeight;
      
      // determine the floor and ceiling of the channel
      double floor   = DBL_MAX;
      double ceiling = DBL_MAX;
      pL2_ps->get( "channelFloor",   floor );
      pL2_ps->get( "channelCeiling", ceiling );      
   
      // use computational grid if floor & ceiling haven't been specified
      if( floor == DBL_MAX || ceiling == DBL_MAX) {
        BBox bb;
        grid->getInteriorSpatialRange(bb);
        floor   = bb.min()(vDir);
        ceiling = bb.max()(vDir);
      }
      cib->powerLaw2_vars.floor   = floor;
      cib->powerLaw2_vars.ceiling = ceiling;
    }  // powerLaw2 inputs

    //_______________________________________________
    //  Channel Flow initialized according to Moser's profile
    //  Flow is assumed to be from x- to x+
    ProblemSpecP dns_ps= c_init_ps->findBlock("DNS_Moser");
    if(dns_ps) {
      cib->whichMethod.push_back( "DNS_Moser" );

      cib->DNS_Moser_vars = DNS_Moser();

      // geometry: computational domain floor and ceiling
      BBox b;
      grid->getInteriorSpatialRange(b);
      cib->DNS_Moser_vars.gridFloor = b.min();
      cib->DNS_Moser_vars.gridCeil  = b.max();

      dns_ps -> require( "verticalDirection", cib->DNS_Moser_vars.verticalDir );
      dns_ps -> require( "dpdx",              cib->DNS_Moser_vars.dpdx );
    }
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
  // reverse iterator
  for(auto rit = cib->whichMethod.rbegin(); rit != cib->whichMethod.rend(); ++rit) {

    std::string whichMethod = *rit;
    //_______________________________________________
    //  multiple vortices
    // See "Boundary Conditions for Direct Simulations of Compressible Viscous
    //     Flows" by Poinsot & LeLe pg 121

    if ( whichMethod == "vortices" ){
      for (int v = 0; v<(int) cib->vortex_vars.origin.size(); v++) {

        Point origin = cib->vortex_vars.origin[v];    // vortex origin
        double C1    = cib->vortex_vars.strength[v];  // vortex strength
        double R     = cib->vortex_vars.radius[v];    // vortex radius
        double R_sqr = R * R;
        double p_ref  = 101325;                       // assumed reference pressure

        // axis of vortex
        string axis = cib->vortex_vars.axis[v];
        int j = -1;
        int k = -1;

        if(axis == "X"){
          j = 1;
          k = 2;
        }
        else if( axis == "Y" ){
          j = 0;
          k = 2;
        }
        else if( axis == "Z" ){
          j = 0;
          k = 1;
        }

        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
          IntVector c = *iter;
          Point pt = patch->cellPosition(c);

          double y = pt(j) - origin(j);
          double z = pt(k) - origin(k);
          double r = sqrt( y*y + z*z);

          if(r <= R ) {
            double A = exp( -((y*y) + (z*z) )/(2 * R_sqr));

            press_CC[c] = p_ref + rho_CC[c] * ((C1*C1)/R_sqr) * A;

            double uvel = vel_CC[c][j];
            double vvel = vel_CC[c][k];

            vel_CC[c][j] = ( uvel - ((C1 * z)/( rho_CC[c] * R_sqr ) ) * A );
            vel_CC[c][k] = ( vvel + ((C1 * y)/( rho_CC[c] * R_sqr ) ) * A );
          }
        }
      }  // loop
    } // vortices

    //__________________________________
    //  Vortex Pairs
    if ( whichMethod == "vortexPairs" ){
      double vortexAmp = cib->vortexPairs_vars.strength;
      double nPairs    = cib->vortexPairs_vars.nPairs; 

      // axis of vortex
      string axis = cib->vortexPairs_vars.axis;
      int j = -1;
      int k = -1;

      if(axis == "X"){
        j = 1;
        k = 2;
      }
      else if( axis == "Y" ){
        j = 0;
        k = 2;
      }
      else if( axis == "Z" ){
        j = 0;
        k = 1;
      }

      // geometry: computational domain
      GridP grid = patch->getLevel()->getGrid();
      BBox b;
      grid->getInteriorSpatialRange(b);

      Vector gridLength = ( b.max() - b.min() );
      Vector lo   = b.min().asVector();
      Vector dx_2 = patch->dCell()/Vector(2,2,2);

      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        Vector here     = patch->cellPosition(c).asVector();
        Vector ratio    = ( here - lo )/gridLength;
        Vector ratio_FC = ( here + dx_2 - lo )/gridLength;

        vel_CC[c][j] += -vortexAmp * std::cos(nPairs * 2. * M_PI * ratio[k] )    * std::sin( M_PI * ratio_FC[j] );
        vel_CC[c][k] +=  vortexAmp * std::sin(nPairs * 2. * M_PI * ratio_FC[k] ) * std::cos( M_PI * ratio[j] );

      }
    } // vortices
    //_______________________________________________
    // gaussian Temperature
    if( whichMethod == "gaussianTemp" ){

      double amp = cib->gaussTemp_vars.amplitude;
      Point origin     = cib->gaussTemp_vars.origin;
      double spread_x  = cib->gaussTemp_vars.spread_x;
      double spread_y  = cib->gaussTemp_vars.spread_y;

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
    if( whichMethod == "counterflow" ){

      double strainRate   = cib->counterflow_vars.strainRate;
      Vector domainLength = cib->counterflow_vars.domainLength;
      IntVector refCell   = cib->counterflow_vars.refCell;

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
    if( whichMethod == "mms_1" ){
      double t = 0.0;
      double A = cib->mms_vars.A;
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

// This code is incomplete!!!

    if( whichMethod == "mms_2" ){
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

    if( whichMethod == "mms_3" ){
      double angle = cib->mms_vars.angle;
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
    if( whichMethod == "powerLaw" ){
      int vDir          =  cib->powerLaw_vars.verticalDir;
      double d          =  cib->powerLaw_vars.gridMin(vDir);
      double gridHeight =  cib->powerLaw_vars.gridMax(vDir);
      double height     =  cib->powerLaw_vars.maxHeight;
      Vector U_infinity =  cib->powerLaw_vars.U_infinity;
      double n          =  cib->powerLaw_vars.exponent;
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

      if (cib->powerLaw_vars.addVariance ){
        MTRand mTwister;

        double gridHeight =  cib->powerLaw_vars.gridMax(vDir);
        double d          =  cib->powerLaw_vars.gridMin(vDir);
        double inv_Cmu    = 1.0/cib->powerLaw_vars.C_mu;
        double u_star2    = cib->powerLaw_vars.u_star * cib->powerLaw_vars.u_star;

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

    //_______________________________________________
    //  power law velocity profile
    //  Ref:  Jeremy Gibbs
    if( whichMethod == "powerLaw2" ){

      int vDir          =  cib->powerLaw2_vars.verticalDir;
      int pDir          =  cib->powerLaw2_vars.principalDir;
      double floor      =  cib->powerLaw2_vars.floor;
      double ceiling    =  cib->powerLaw2_vars.ceiling;
      double halfChanHeight  =  cib->powerLaw2_vars.halfChanHeight;
      const Level* level = patch->getLevel();

      double visc      = ice_matl->getViscosity();
      double rho       = ice_matl->getInitialDensity();
      double Re_tau    = cib->powerLaw2_vars.Re_tau;
      double u_tau     = ( visc * Re_tau/(rho * halfChanHeight) ) ;
      double vonKarman = 0.4;
      
      //std::cout << "     halfChannelHeight: " << halfChanHeight << " vDir: " << vDir << " pDir: " << pDir 
      //          << " visc: " << visc << " Re_tau: " << Re_tau << " u_tau: " << u_tau << endl;

      for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;

        Point here   = level->getCellPosition(c);
        double h     = here.asVector()[vDir] ;
        
         // Clamp cells outide of channel
        if( h < floor || h > ceiling ){
          vel_CC[c] = Vector(0,0,0);
          continue;
        }
        
        double ratio = -9;
        if ( h < halfChanHeight) {
          ratio = (h - floor)/halfChanHeight;
        } 
        else {
          ratio = 1 - (h - halfChanHeight)/halfChanHeight;
        }
        
        ratio = Clamp(ratio,0.0,1.0);
        
        vel_CC[c][pDir] = (u_tau/vonKarman) * std::log( ratio * Re_tau );
      }
    }
    //_______________________________________________
    //   DNS velocity profile
    //   Reference:
    //    "R. D. Moser, J. Kim, N. N. Mansour, " Direct numberical simulation of turbulent
    //     channel flow up to Re_tau=590", Physics of Fluids, Vol 11, Number 4, 1999, 943-945
    //     This assumes that the flow is from x- to x+

    if ( whichMethod == "DNS_Moser" ){

      double visc       =  ice_matl->getViscosity();
      double rho        =  ice_matl->getInitialDensity();
      double nu         =  visc / rho;
      int vDir          =  cib->DNS_Moser_vars.verticalDir;
      double dpdx       =  cib->DNS_Moser_vars.dpdx;
      double gridCeil   =  cib->DNS_Moser_vars.gridCeil(vDir);
      double gridFloor  =  cib->DNS_Moser_vars.gridFloor(vDir);
      const Level* level = patch->getLevel();

      if ( visc  == 0 || dpdx == 0 ){
        ostringstream warn;
        warn << "  ERROR: ICE: CustomInitialization DNS_moser \n"
             << "  Either the material viscosity or dpdx equals 0.0.  They must be non-zero.\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }

      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;

        Point here = level->getCellPosition(c);
        double y   = here.asVector()[vDir];

        vel_CC[c] = Vector(0,0,0);
        double u = 1./(2. * nu) * dpdx * ( y*y - gridCeil*y );
        vel_CC[c].x( u );

        // Clamp for edge/corner cells
        if( y < gridFloor || y > gridCeil ){
          vel_CC[c] = Vector(0,0,0);
        }
      }
    }
  }  // loop over whichMethod
}

} // end uintah namespace
