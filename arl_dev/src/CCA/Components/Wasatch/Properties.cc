/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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

// jcs for some reason the serialization doesn't work without this:
Interp1D i1d;
Interp2D i2d;
Interp3D i3d;


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
      double preal=0, pimag=0;
      refIxParams->getAttribute( "real", preal );
      refIxParams->getAttribute( "imag", pimag );
      const std::complex<double> refIx( preal, pimag );

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

      typedef ParticleRadProps<SpatialOps::SVolField>::Builder ParticleProps;
      gh.exprFactory->register_expression(
          scinew ParticleProps( propSelection,
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
    typedef RadPropsEvaluator<SpatialOps::SVolField>::Builder RadPropsExpr;
    gh.exprFactory->register_expression( scinew RadPropsExpr( parse_nametag(ggParams->findBlock("NameTag")),
                                                              parse_nametag(ggParams->findBlock("Temperature")->findBlock("NameTag")),
                                                              spMap,fileName) );
  }

  //====================================================================

  enum DensityEvaluationLevel
  {
    NORMAL,
    STAR,
    STARSTAR
  };

  Expr::ExpressionID
  parse_density_solver( const Uintah::ProblemSpecP& params,
                        const StateTable& table,
                        Expr::Tag densityTag,
                        const DensityEvaluationLevel densLevel,
                        GraphHelper& gh,
                        std::set<std::string>& lockedFields )
  {
    Expr::ExpressionID densCalcID;  // BE SURE TO POPULATE THIS BELOW!

    // Lock the density because on initialization this may be an intermediate
    // quantity, but is always needed as a guess for the solver here.
    lockedFields.insert( densityTag.name() );

    const std::string densTableName = "Density";
    if( !table.has_depvar(densTableName) ){
      throw Uintah::ProblemSetupException( "Table has no density entry in it, but density was requested through your input file!", __FILE__, __LINE__ );
    }
    const InterpT* const densInterp = table.find_entry( densTableName );

    Expr::ExpressionFactory& factory = *gh.exprFactory;

    std::string tagNameAppend, scalarTagNameAppend;
    switch (densLevel){
      case NORMAL    : tagNameAppend=scalarTagNameAppend="";                          break;
      case STAR      : tagNameAppend=TagNames::self().star; scalarTagNameAppend = ""; break;
      case STARSTAR  : tagNameAppend=scalarTagNameAppend=TagNames::self().doubleStar; break;
    }

    densityTag.name() += tagNameAppend;

    if( params->findBlock("ModelBasedOnMixtureFraction") ){

      const Uintah::ProblemSpecP modelParams = params->findBlock("ModelBasedOnMixtureFraction");
      Expr::Tag rhofTag = parse_nametag( modelParams->findBlock("DensityWeightedMixtureFraction")->findBlock("NameTag") );
      if( densLevel != NORMAL   ) rhofTag.context() = Expr::STATE_NONE;
      rhofTag.name() += scalarTagNameAppend;

      typedef DensFromMixfrac<SVolField>::Builder DensCalc;
      densCalcID = factory.register_expression( scinew DensCalc( *densInterp, densityTag, rhofTag ) );

    }
    else if( params->findBlock("ModelBasedOnMixtureFractionAndHeatLoss") ){

      if( !table.has_depvar("Enthalpy") ){
        throw Uintah::ProblemSetupException( "Table has no enthalpy entry in it, but enthalpy is required for the heat loss class of models!", __FILE__, __LINE__ );
      }
      const InterpT* const enthInterp = table.find_entry("Enthalpy");

      const Uintah::ProblemSpecP modelParams = params->findBlock("ModelBasedOnMixtureFractionAndHeatLoss");
      Expr::Tag rhofTag    = parse_nametag( modelParams->findBlock("DensityWeightedMixtureFraction")->findBlock("NameTag") );
      Expr::Tag rhohTag    = parse_nametag( modelParams->findBlock("DensityWeightedEnthalpy")->findBlock("NameTag") );
      Expr::Tag heatLossTag= parse_nametag( modelParams->findBlock("HeatLoss")->findBlock("NameTag") );

      lockedFields.insert( heatLossTag.name() ); // ensure that Uintah knows about this field

      // modify name & context when we are calculating density at newer time
      // levels since this will be using STATE_NONE information as opposed to
      // potentially STATE_N information.
      if( densLevel != NORMAL ){
        rhofTag.context()     = Expr::STATE_NONE;
        rhohTag.context()     = Expr::STATE_NONE;
        rhofTag.name()        += scalarTagNameAppend;
        rhohTag.name()        += scalarTagNameAppend;
        heatLossTag.name()    += scalarTagNameAppend;
      }

      typedef DensHeatLossMixfrac<SVolField>::Builder DensCalc;
      densCalcID = factory.register_expression( scinew DensCalc( densityTag, heatLossTag, rhofTag, rhohTag, *densInterp, *enthInterp ) );

    }
    return densCalcID;
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
   *  \param [inout] lockedFields the set of fields that should be controlled by
   *         Uintah and not allowed to be scratch/temporary fields.
   */
  void parse_tabprops( Uintah::ProblemSpecP& params,
                       GraphHelper& gh,
                       const Category cat,
                       const bool doDenstPlus,
                       std::set<std::string>& lockedFields )
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
      if( ivarMap.find(*inm) == ivarMap.end() ){
        std::ostringstream msg;
        msg << "ERROR: table variable '" << *inm << "' was not provided\n";
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
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
      assert( interp != NULL );

      //____________________________________________
      // get the type of field that we will evaluate
      std::string fieldType;
      dvarParams->getWithDefault( "type", fieldType, "SVOL" );

      switch( get_field_type(fieldType) ){
      case SVOL: {
        typedef TabPropsEvaluator<SpatialOps::SVolField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, *interp, ivarNames ) );
        if( doDenstPlus && dvarTableName=="Density" ){
          const Expr::Tag densStarTag ( dvarTag.name()+TagNames::self().star,       dvarTag.context() );
          const Expr::Tag densStar2Tag( dvarTag.name()+TagNames::self().doubleStar, dvarTag.context() );
          gh.rootIDs.insert( gh.exprFactory->register_expression( scinew PropEvaluator( densStarTag,  *interp, ivarNames ) ) );
          gh.rootIDs.insert( gh.exprFactory->register_expression( scinew PropEvaluator( densStar2Tag, *interp, ivarNames ) ) );
        }
        break;
      }
      case XVOL: {
        typedef TabPropsEvaluator<SpatialOps::SSurfXField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, *interp, ivarNames ) );
        break;
      }
      case YVOL: {
        typedef TabPropsEvaluator<SpatialOps::SSurfYField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, *interp, ivarNames ) );
        break;
      }
      case ZVOL: {
        typedef TabPropsEvaluator<SpatialOps::SSurfZField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, *interp, ivarNames ) );
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
      typedef TabPropsHeatLossEvaluator<SpatialOps::SVolField>::Builder HLEval;
      gh.exprFactory->register_expression( scinew HLEval( parse_nametag( hlParams->findBlock("NameTag") ),
                                                          *adEnthInterp  ,
                                                          *sensEnthInterp,
                                                          *enthInterp    ,
                                                          hlIx,
                                                          hlIvars ) );
    }

    //________________________________________________________________
    // create an expression specifically for density.
    const Uintah::ProblemSpecP densityParams = params->findBlock("ExtractDensity");
    if( densityParams ){
      const Expr::Tag densityTag = parse_nametag( densityParams->findBlock("NameTag") );
      parse_density_solver( densityParams, table, densityTag, NORMAL, gh, lockedFields );
      if( doDenstPlus ){
        const Expr::ExpressionID id1 = parse_density_solver( densityParams, table, densityTag, STAR,     gh, lockedFields );
        const Expr::ExpressionID id2 = parse_density_solver( densityParams, table, densityTag, STARSTAR, gh, lockedFields );
        gh.exprFactory->cleave_from_children( id1 );
        gh.exprFactory->cleave_from_children( id2 );
      }
    }

  }
  //====================================================================

  void
  parse_twostream_mixing( Uintah::ProblemSpecP params,
                          const bool doDenstPlus,
                          GraphCategories& gc )
  {
    const Expr::Tag fTag    = parse_nametag( params->findBlock("MixtureFraction")->findBlock("NameTag") );
    const Expr::Tag rhofTag = parse_nametag( params->findBlock("DensityWeightedMixtureFraction")->findBlock("NameTag") );
    const Expr::Tag rhoTag  = parse_nametag( params->findBlock("Density")->findBlock("NameTag") );
    double rho0, rho1;
    params->getAttribute("rho0",rho0);
    params->getAttribute("rho1",rho1);

    // initial conditions for density
    {
      GraphHelper& gh = *gc[INITIALIZATION];
      typedef TwoStreamDensFromMixfr<SVolField>::Builder ICDensExpr;
      const Expr::Tag icRhoTag( rhoTag.name(), Expr::STATE_NONE );
      gh.rootIDs.insert( gh.exprFactory->register_expression( scinew ICDensExpr(icRhoTag,fTag,rho0,rho1) ) );

      if( doDenstPlus ){
        const Expr::Tag icRhoStarTag ( rhoTag.name()+TagNames::self().star,       Expr::STATE_NONE );
        const Expr::Tag icRhoStar2Tag( rhoTag.name()+TagNames::self().doubleStar, Expr::STATE_NONE );
        gh.rootIDs.insert( gh.exprFactory->register_expression( scinew ICDensExpr(icRhoStarTag, fTag,rho0,rho1) ) );
        gh.rootIDs.insert( gh.exprFactory->register_expression( scinew ICDensExpr(icRhoStar2Tag,fTag,rho0,rho1) ) );
      }
    }

    typedef TwoStreamMixingDensity<SVolField>::Builder DensExpr;
    gc[ADVANCE_SOLUTION]->exprFactory->register_expression( scinew DensExpr(rhoTag,rhofTag,rho0,rho1) );

    if( doDenstPlus ){
      const TagNames& names = TagNames::self();

      Expr::Tag rhoStar ( rhoTag .name()+names.star, rhoTag.context() );
      Expr::Tag rhofStar( rhofTag.name(), Expr::STATE_NONE );
      const Expr::ExpressionID id1 = gc[ADVANCE_SOLUTION]->exprFactory->register_expression( scinew DensExpr(rhoStar,rhofStar,rho0,rho1) );

      rhoStar .name() = rhoTag.name()  + names.doubleStar;
      rhofStar.name() = rhofTag.name() + names.doubleStar;
      const Expr::ExpressionID id2 = gc[ADVANCE_SOLUTION]->exprFactory->register_expression( scinew DensExpr(rhoStar,rhofStar,rho0,rho1) );
      gc[ADVANCE_SOLUTION]->exprFactory->cleave_from_children(id1);
      gc[ADVANCE_SOLUTION]->exprFactory->cleave_from_children(id2);
    }
  }

  //====================================================================

  void setup_scalar_predictors( Uintah::ProblemSpecP params,
                                GraphHelper& solnGraphHelper )
  {
    /* Check to see if we have scalar transport equation already set up and the
     * problem is variable density, so we can obtain solution variables in order
     * to estimate their values at "*" RK stage to be able to estimate the value
     * of density at this RK stage
     */
    const TagNames& tagNames = TagNames::self();

    for( Uintah::ProblemSpecP transEqnParams= params->findBlock("TransportEquation");
        transEqnParams != 0;
        transEqnParams=transEqnParams->findNextBlock("TransportEquation") )
    {
      std::string solnVarName;
      transEqnParams->get( "SolutionVariable", solnVarName );

      // Here we get the variables needed for calculations at the stage "*"
      const Expr::Tag solnVarTagNp1  ( solnVarName,                 Expr::STATE_NONE ); // tag for rhof_{n+1}

      // Here we get the variables needed for calculations at the stage "**"
      const Expr::Tag solnVarRHSStarTag = tagNames.make_star_rhs(solnVarName);;
      const Expr::Tag solnVar2StarTag   = tagNames.make_double_star(solnVarName);

      if( !solnGraphHelper.exprFactory->have_entry( solnVar2StarTag ) ){
        solnGraphHelper.exprFactory->register_expression( scinew SolnVarEst<SVolField>::Builder( solnVar2StarTag, solnVarTagNp1, solnVarRHSStarTag, tagNames.dt ));
      }
    }
  }

  //====================================================================

  void
  setup_property_evaluation( Uintah::ProblemSpecP& params,
                             GraphCategories& gc,
                             std::set<std::string>& lockedFields )
  {
    //__________________________________________________________________________
    // extract the density tag in the cases that it is needed

    Uintah::ProblemSpecP densityParams  = params->findBlock("Density");
    
    if (!densityParams) {
      std::ostringstream msg;
      msg << std::endl << "Error: You must specify a <Density> block. Please revise your input file." << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    
    Uintah::ProblemSpecP tabPropsParams = params->findBlock("TabProps");
    Uintah::ProblemSpecP radPropsParams = params->findBlock("RadProps");
    Uintah::ProblemSpecP twoStreamParams= params->findBlock("TwoStreamMixing");

    if( radPropsParams ){
      parse_radprops( radPropsParams, *gc[ADVANCE_SOLUTION] );
    }

    const bool isConstDensity = densityParams->findBlock("Constant");
    const bool doDenstPlus = !isConstDensity
                           && params->findBlock("MomentumEquations")
                           && params->findBlock("TransportEquation");

    if( twoStreamParams ){
      parse_twostream_mixing( twoStreamParams, doDenstPlus, gc );
    }

    // TabProps
    for( Uintah::ProblemSpecP tabPropsParams = params->findBlock("TabProps");
         tabPropsParams != 0;
         tabPropsParams = tabPropsParams->findNextBlock("TabProps") )
    {
      const Category cat = parse_tasklist( tabPropsParams,false);
      parse_tabprops( tabPropsParams, *gc[cat], cat, doDenstPlus, lockedFields );
    }

    if( doDenstPlus ) setup_scalar_predictors( params, *gc[ADVANCE_SOLUTION] );
  }

  //====================================================================

} // namespace Wasatch
