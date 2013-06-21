/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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

//--- Local (Wasatch) includes ---//
#include "Properties.h"
#include "GraphHelperTools.h"
#include "ParseTools.h"
#include "FieldAdaptor.h"
#include "Expressions/TabPropsEvaluator.h"
#include "Expressions/TabPropsHeatLossEvaluator.h"
#include "Expressions/DensityCalculator.h"
#include "Expressions/RadPropsEvaluator.h"
#include "Expressions/SolnVarEst.h"
#include "TagNames.h"

//--- ExprLib includes ---//
#include <expression/ExpressionFactory.h>

//--- TabProps includes ---//
#include <tabprops/StateTable.h>

//--- RadProps includes ---//
#include <radprops/RadiativeSpecies.h>

//--- Uintah includes ---//
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <fstream>
#include <iterator>

using std::endl;
using std::flush;

#ifndef TabProps_BSPLINE
// jcs for some reason the serialization doesn't work without this:
Interp1D i1d;
Interp2D i2d;
Interp3D i3d;
#endif

namespace Wasatch{

  //====================================================================

  void parse_radprops( Uintah::ProblemSpecP& params,
                       GraphHelper& gh )
  {
    //_________________________________________________
    // Particle radiative properties
    // jcs do we need multiple evaluators?  If so, we need a loop here...
    Uintah::ProblemSpecP pParams = params->findBlock("Particles");
    if( pParams ){

      const Uintah::ProblemSpecP refIxParams = pParams->findBlock("RefractiveIndex");
      std::complex<double> refIx;
      refIxParams->getAttribute( "real", refIx.real() );
      refIxParams->getAttribute( "imag", refIx.imag() );

      ParticleRadProp propSelection;
      const std::string prop = pParams->findBlock("RadCoefType")->getNodeValue();
      if     ( prop == "PLANCK_ABS"    ) propSelection = PLANCK_ABSORPTION_COEFF;
      else if( prop == "PLANCK_SCA"    ) propSelection = PLANCK_SCATTERING_COEFF;
      else if( prop == "ROSSELAND_ABS" ) propSelection = ROSSELAND_ABSORPTION_COEFF;
      else if( prop == "ROSSELAND_SCA" ) propSelection = ROSSELAND_SCATTERING_COEFF;
      else{
        std::ostringstream msg;
        msg << std::endl << "Unsupported particle radiative property selection found: " << prop << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      proc0cout << "Particle properties using refractive index: " << refIx << std::endl;

      typedef ParticleRadProps<SpatialOps::structured::SVolField>::Builder ParticleProps;
      gh.exprFactory->register_expression(
          new ParticleProps( propSelection,
                             parse_nametag( pParams->findBlock("NameTag") ),
                             parse_nametag( pParams->findBlock("Temperature"   )->findBlock("NameTag")),
                             parse_nametag( pParams->findBlock("ParticleRadius")->findBlock("NameTag")),
                             refIx ) );
    }

    //___________________________________________________________
    // for now, we only support grey gas properties.
    Uintah::ProblemSpecP ggParams = params->findBlock("GreyGasAbsCoef");

    /* Procedure:
     *
     *  1. Parse the file name
     *  2. Determine the independent variables that are required
     *  3. Register the expression to evaluate the radiative property
     */
    std::string fileName;
    ggParams->get("FileName",fileName);

    proc0cout << "Loading RadProps file: " << fileName << std::endl;

    //___________________________________________________________
    // get information for the independent variables in the table

    RadSpecMap spMap;

    for( Uintah::ProblemSpecP spParams = ggParams->findBlock("SpeciesMoleFraction");
         spParams != 0;
         spParams = spParams->findNextBlock("SpeciesMoleFraction") ){
      std::string spnam;   spParams->getAttribute("name",spnam);
      spMap[ species_enum( spnam ) ] = parse_nametag( spParams->findBlock("NameTag") );
    }
    typedef RadPropsEvaluator<SpatialOps::structured::SVolField>::Builder RadPropsExpr;
    gh.exprFactory->register_expression( new RadPropsExpr( parse_nametag(ggParams->findBlock("NameTag")),
                                                           parse_nametag(ggParams->findBlock("Temperature")->findBlock("NameTag")),
                                                           spMap,fileName) );
  }

  //====================================================================

  /**
   *  \ingroup WasatchParser
   *  \brief set up TabProps for use on the given GraphHelper
   *  \param params - the parser parameters for the TabProps specification.
   *  \param gh - the GraphHelper associated with this instance of TabProps.
   *  \param cat - the Category specifying the task associated with this 
   *         instance of TabProps.
   *  \param doDenstPlus - the boolean showing whether we have a variable
   *         density case and we want to do pressure projection or not
   */
  void parse_tabprops( Uintah::ProblemSpecP& params,
                       GraphHelper& gh,
                       const Category cat,
                       const bool doDenstPlus ) 
  {
    std::string fileName;
    params->get("FileNamePrefix",fileName);

    proc0cout << "Loading TabProps file '" << fileName << "' ... " << std::flush;

    StateTable table;
    try{
      table.read_table( fileName+".tbl" );
    }
    catch( std::exception& e ){
      std::ostringstream msg;
      msg << e.what() << std::endl << std::endl
          << "Error reading TabProps file '" << fileName << ".tbl'" << std::endl
          << "Check to ensure that the file exists." << std::endl
          << "It is also possible that there was an error loading the file.  This could be caused" << std::endl
          << "by a file in a format incompatible with the version of TabProps linked in here." << std::endl
          << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    proc0cout << "done" << std::endl;

    //___________________________________________________________
    // get information for the independent variables in the table
    typedef std::map<std::string,Expr::Tag> VarNameMap;
    VarNameMap ivarMap;

    for( Uintah::ProblemSpecP ivarParams = params->findBlock("IndependentVariable");
         ivarParams != 0;
         ivarParams = ivarParams->findNextBlock("IndependentVariable") ){
      std::string ivarTableName;
      const Expr::Tag ivarTag = parse_nametag( ivarParams->findBlock("NameTag") );
      ivarParams->get( "NameInTable", ivarTableName );
      ivarMap[ivarTableName] = ivarTag;
    }

    //______________________________________________________________
    // NOTE: the independent variable names must be specified in the
    // exact order dictated by the table.  This order will determine
    // the ordering for the arguments to the evaluator later on.
    typedef std::vector<std::string> Names;
    Expr::TagList ivarNames;
    const Names& ivars = table.get_indepvar_names();
    for( Names::const_iterator inm=ivars.begin(); inm!=ivars.end(); ++inm ){
      ivarNames.push_back( ivarMap[*inm] );
    }

    //________________________________________________________________
    // create an expression for each property.  Alternatively, we
    // could create an expression that evaluated all required
    // properties at once, since the expression has that capability...
    for( Uintah::ProblemSpecP dvarParams = params->findBlock("ExtractVariable");
         dvarParams != 0;
         dvarParams = dvarParams->findNextBlock("ExtractVariable") ){

      //_______________________________________
      // extract dependent variable information
      const Expr::Tag dvarTag = parse_nametag( dvarParams->findBlock("NameTag") );

      std::string dvarTableName;
      dvarParams->get( "NameInTable", dvarTableName );
      if( !table.has_depvar(dvarTableName) ){
        std::ostringstream msg;
        msg << "Table '" << fileName
            << "' has no dependent variable named '" << dvarTableName << "'"
            << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      proc0cout << "Constructing property evaluator for '" << dvarTag
                << "' from file '" << fileName << "'." << std::endl;
      const InterpT* const interp = table.find_entry( dvarTableName );
      //____________________________________________
      // get the type of field that we will evaluate
      std::string fieldType;
      dvarParams->getWithDefault( "type", fieldType, "SVOL" );

      switch( get_field_type(fieldType) ){
      case SVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SVolField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, interp->clone(), ivarNames ) );
        break;
      }
      case XVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfXField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, interp->clone(), ivarNames ) );
        break;
      }
      case YVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfYField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, interp->clone(), ivarNames ) );
        break;
      }
      case ZVOL: {
        typedef TabPropsEvaluator<SpatialOps::structured::SSurfZField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, interp->clone(), ivarNames ) );
        break;
      }
      default:
        std::ostringstream msg;
        msg << "ERROR: unsupported field type named '" << fieldType << "'" << endl
            << __FILE__ << " : " << __LINE__ << endl;
        throw std::runtime_error( msg.str() );
      }
    }

    //____________________________________________________________
    // create an expression to compute the heat loss, if requested
    // jcs we could automatically create some of these based on the
    //     names in the table, since we have these names hard-coded
    //     for heat loss cases
    if( params->findBlock("HeatLoss") ){
      Uintah::ProblemSpecP hlParams = params->findBlock("HeatLoss");
      const std::string hlName="HeatLoss";
      const Names::const_iterator ivarIter = std::find(ivars.begin(),ivars.end(),hlName);
      const size_t hlIx = std::distance( ivars.begin(), ivarIter );
      Expr::TagList hlIvars = ivarNames;
      hlIvars.erase( hlIvars.begin() + hlIx );
      if( hlIx >= ivarNames.size() ){
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << endl
            << "ERROR: heat loss specified (" << hlName << ") was not found in the table" << endl;
        throw std::runtime_error( msg.str() );
      }

      const InterpT* const adEnthInterp   = table.find_entry( "AdiabaticEnthalpy" );
      const InterpT* const sensEnthInterp = table.find_entry( "SensibleEnthalpy"  );
      const InterpT* const enthInterp     = table.find_entry( "Enthalpy"          );
      typedef TabPropsHeatLossEvaluator<SpatialOps::structured::SVolField>::Builder HLEval;
      gh.exprFactory->register_expression( scinew HLEval( parse_nametag( hlParams->findBlock("NameTag") ),
                                                          adEnthInterp  ->clone(),
                                                          sensEnthInterp->clone(),
                                                          enthInterp    ->clone(),
                                                          hlIx,
                                                          hlIvars ) );
    }

    //________________________________________________________________
    // create an expression specifically for density.
    const Uintah::ProblemSpecP densityParams = params->findBlock("ExtractDensity");
    if( densityParams ){
      Expr::TagList rhoEtaTags, etaTags;
      for( Uintah::ProblemSpecP rhoEtaParams = densityParams->findBlock("DensityWeightedIVar");
          rhoEtaParams != 0;
          rhoEtaParams = rhoEtaParams->findNextBlock("DensityWeightedIVar") ){
        const Expr::Tag rhoEtaTag = parse_nametag( rhoEtaParams->findBlock("NameTag") );
        rhoEtaTags.push_back( rhoEtaTag );
        Uintah::ProblemSpecP etaParams = rhoEtaParams->findBlock("RelatedIVar");
        const Expr::Tag etaTag = parse_nametag( etaParams->findBlock("NameTag") );
        etaTags.push_back( etaTag );
      }


      //_______________________________________
      // extract density variable information
      const std::string dvarTableName = "Density";
      if( !table.has_depvar(dvarTableName) ){
        std::ostringstream msg;
        msg << "Table '" << fileName
            << "' has no density entry in it, but density was requested through your input file!"
            << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      const InterpT* const interp = table.find_entry( dvarTableName );

      //_____________________________________
      // register the expression for density
      const Expr::Tag densityTag = parse_nametag( densityParams->findBlock("NameTag") );
      typedef DensityCalculator<SpatialOps::structured::SVolField>::Builder DensCalc;
      gh.exprFactory->register_expression( scinew DensCalc( densityTag, interp->clone(), rhoEtaTags,
                                                            etaTags, ivarNames ) );

      //_________________________________________________________________________________________
      // preparing the input arguments of the expression for density at the next RK time stages if they are needed to be estimated
      Expr::Tag densityStarTag = Expr::Tag();
      Expr::TagList rhoEtaStarTags, ivarStarTags, etaStarTags;

      Expr::Tag density2StarTag = Expr::Tag();
      Expr::TagList rhoEta2StarTags, ivar2StarTags, eta2StarTags;

      if( doDenstPlus && cat==INITIALIZATION ){
        const TagNames& tagNames = TagNames::self();
        densityStarTag  = Expr::Tag(densityTag.name() + tagNames.star,       Expr::STATE_NONE);
        density2StarTag = Expr::Tag(densityTag.name() + tagNames.doubleStar, Expr::STATE_NONE);
        typedef DensityCalculator<SpatialOps::structured::SVolField>::Builder DensCalc;
        const Expr::ExpressionID densStarID  = gh.exprFactory->register_expression( scinew DensCalc( densityStarTag,  interp->clone(), rhoEtaTags, etaTags, ivarNames ) );
        const Expr::ExpressionID dens2StarID = gh.exprFactory->register_expression( scinew DensCalc( density2StarTag, interp->clone(), rhoEtaTags, etaTags, ivarNames ) );
        gh.rootIDs.insert(densStarID );
        gh.rootIDs.insert(dens2StarID);
      }
      //============================================================
      // Note that it is currently assumed that we solve transport 
      // equation for non density weighted independent variables as
      // well, so, we have them in SolnVarTags/SolnVarStarTags and we 
      // should be able to update them for the next RK time stage
      //============================================================
      
      // This calculations should only take place for variable density cases when we have both momentum and scalar transport equations
      //___________________________________
      // estimate the density field at RK stage "*" and "**"
      if( doDenstPlus && cat==ADVANCE_SOLUTION ){
        // Here the soln variables in density weighted form will be separated to generate rhoEta at the "*" stage
        const TagNames& tagNames = TagNames::self();
        BOOST_FOREACH( const Expr::Tag& tag, rhoEtaTags ){
          rhoEtaStarTags .push_back( Expr::Tag( tag.name() + tagNames.star,       Expr::STATE_NONE ) );
          rhoEta2StarTags.push_back( Expr::Tag( tag.name() + tagNames.doubleStar, Expr::STATE_NONE ) );
        }
        BOOST_FOREACH( const Expr::Tag& tag, ivarNames ){
          ivarStarTags. push_back( Expr::Tag( tag.name() + tagNames.star,       Expr::STATE_NONE ) );
          ivar2StarTags.push_back( Expr::Tag( tag.name() + tagNames.doubleStar, Expr::STATE_NONE ) );
        }
        BOOST_FOREACH( const Expr::Tag& tag, etaTags ){
          etaStarTags .push_back( Expr::Tag( tag.name() + tagNames.star,       Expr::STATE_NONE ) );
          eta2StarTags.push_back( Expr::Tag( tag.name() + tagNames.doubleStar, Expr::STATE_NONE ) );
        }
        
        // register the expression for density at RK time stage
        densityStarTag = Expr::Tag(densityTag.name() + tagNames.star, Expr::CARRY_FORWARD);
        const Expr::ExpressionID densStar = gh.exprFactory->register_expression( scinew DensCalc( densityStarTag, interp->clone(), rhoEtaStarTags, etaStarTags, ivarStarTags ) );
        gh.exprFactory->cleave_from_children ( densStar );

        // register the expression for density at RK time stage
        density2StarTag = Expr::Tag(densityTag.name() + tagNames.doubleStar, Expr::CARRY_FORWARD);
        const Expr::ExpressionID dens2Star = gh.exprFactory->register_expression( scinew DensCalc( density2StarTag, interp->clone(), rhoEta2StarTags, eta2StarTags, ivar2StarTags ) );
        gh.exprFactory->cleave_from_children ( dens2Star );
        
      } // density predictor

    } // density

  }
  //====================================================================

  void
  setup_property_evaluation( Uintah::ProblemSpecP& params,
                             GraphCategories& gc )
  {
    //__________________________________________________________________________
    // extract the density tag in the cases that it is needed

    Uintah::ProblemSpecP densityParams  = params->findBlock("Density");
    Uintah::ProblemSpecP tabPropsParams = params->findBlock("TabProps");
    Uintah::ProblemSpecP radPropsParams = params->findBlock("RadProps");

    if( radPropsParams ){
      parse_radprops( radPropsParams, *gc[ADVANCE_SOLUTION] );
    }

    if (tabPropsParams) {
      if (tabPropsParams->findBlock("ExtractDensity") && !densityParams) {
        std::ostringstream msg;
        msg << "ERROR: You need a tag for density when you want to extract it using TabProps." << endl
            << "       Please include the \"Density\" block in wasatch in your input file." << endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }

    for( Uintah::ProblemSpecP tabPropsParams = params->findBlock("TabProps");
         tabPropsParams != 0;
         tabPropsParams = tabPropsParams->findNextBlock("TabProps") ){

      // determine which task list this goes on
      Category cat = parse_tasklist( tabPropsParams,false);

      /* Check to see if we have scalar transport equation already set up and the
       * problem is variable density, so we can obtain solution variables in order
       * to estimate their values at "*" RK stage to be able to estimate the value
       * of density at this RK stage
       */
      Uintah::ProblemSpecP transEqnParams  = params->findBlock("TransportEquation");      
      Uintah::ProblemSpecP momEqnParams  = params->findBlock("MomentumEquations");      
      bool isConstDensity;
      densityParams->get("IsConstant",isConstDensity);

      Expr::TagList solnVarStarTags=Expr::TagList();
      Expr::TagList solnVar2StarTags=Expr::TagList();
      const bool doDenstPlus = momEqnParams && transEqnParams && !isConstDensity;
      if (doDenstPlus && cat==ADVANCE_SOLUTION) {
        std::string solnVarName;
        
        Expr::Tag solnVarTag, solnVarRHSTag, solnVarStarTag;
        Expr::Tag solnVarRHSStarTag, solnVar2StarTag;
        const TagNames& tagNames = TagNames::self();
        
        for( ; transEqnParams != 0; transEqnParams=transEqnParams->findNextBlock("TransportEquation") ) {
          transEqnParams->get( "SolutionVariable", solnVarName );
          
          // Here we get the variables needed for calculations at the stage "*" 
          solnVarTag = Expr::Tag( solnVarName, Expr::STATE_N );
          solnVarRHSTag = Expr::Tag( solnVarName+"_rhs", Expr::STATE_NONE);
          solnVarStarTag =  Expr::Tag( solnVarName+tagNames.star, Expr::STATE_NONE) ;

          if( !gc[ADVANCE_SOLUTION]->exprFactory->have_entry( solnVarStarTag ) ){
            gc[ADVANCE_SOLUTION]->exprFactory->register_expression( scinew SolnVarEst<SVolField>::Builder( solnVarStarTag, solnVarTag, solnVarRHSTag, tagNames.timestep ));
          }
          solnVarStarTags.push_back( solnVarStarTag );          
          
          // Here we get the variables needed for calculations at the stage "**" 
          solnVarRHSStarTag = Expr::Tag( solnVarName+"_rhs"+tagNames.star, Expr::STATE_NONE);
          solnVar2StarTag =  Expr::Tag( solnVarName+tagNames.doubleStar, Expr::STATE_NONE) ;
          
          if( !gc[ADVANCE_SOLUTION]->exprFactory->have_entry( solnVar2StarTag ) ){
            gc[ADVANCE_SOLUTION]->exprFactory->register_expression( scinew SolnVarEst<SVolField>::Builder( solnVar2StarTag, solnVarStarTag, solnVarRHSStarTag, tagNames.timestep ));
          }
          solnVar2StarTags.push_back( solnVar2StarTag );          
          
        }
      }
      parse_tabprops( tabPropsParams, *gc[cat], cat, doDenstPlus );
    }
  }

  //====================================================================

} // namespace Wasatch
