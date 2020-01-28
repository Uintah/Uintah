/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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
#include <CCA/Components/Wasatch/Properties.h>
#include <CCA/Components/Wasatch/GraphHelperTools.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/Expressions/TabPropsEvaluator.h>
#include <CCA/Components/Wasatch/Expressions/TabPropsHeatLossEvaluator.h>
#include <CCA/Components/Wasatch/Expressions/RadPropsEvaluator.h>
#include <CCA/Components/Wasatch/Expressions/SolnVarEst.h>
#include <CCA/Components/Wasatch/Expressions/SpeciesDiffusivityFromLewisNumber.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/DensityFromMixFrac.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/DensityFromMixFracAndHeatLoss.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/TwoStreamMixingDensity.h>

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


namespace WasatchCore{

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
         spParams != nullptr;
         spParams = spParams->findNextBlock("SpeciesMoleFraction") ){
      std::string spnam;   spParams->getAttribute("name",spnam);
      spMap[ RadProps::species_enum( spnam ) ] = parse_nametag( spParams->findBlock("NameTag") );
    }
    typedef RadPropsEvaluator<SpatialOps::SVolField>::Builder RadPropsExpr;
    gh.exprFactory->register_expression( scinew RadPropsExpr( parse_nametag(ggParams->findBlock("NameTag")),
                                                              parse_nametag(ggParams->findBlock("Temperature")->findBlock("NameTag")),
                                                              spMap,fileName) );
  }

  //====================================================================

  Expr::ExpressionID
  parse_density_solver( const Uintah::ProblemSpecP& params,
                        const StateTable& table,
                        Expr::Tag densityTag,
                        GraphHelper& gh,
                        const Category& cat,
                        std::set<std::string>& persistentFields,
                        const bool weakForm)
  {
    if (cat == INITIALIZATION) {
      throw Uintah::ProblemSetupException( "You cannot currently use a density calculator for Initialization of the density. Please use ExtractVariable rather than ExtractDensity in your initial condition for TabProps.", __FILE__, __LINE__ );
    }
    Expr::ExpressionID densCalcID;  // BE SURE TO POPULATE THIS BELOW!

    const TagNames& tagNames = TagNames::self();

    // Lock the density because on initialization this may be an intermediate
    // quantity, but is always needed as a guess for the solver here.
    persistentFields.insert( densityTag.name() );

    const std::string densTableName = "Density";
    if( !table.has_depvar(densTableName) ){
      throw Uintah::ProblemSetupException( "Table has no density entry in it, but density was requested through your input file!", __FILE__, __LINE__ );
    }
    const InterpT* const densInterp = table.find_entry( densTableName );

    Expr::ExpressionFactory& factory = *gh.exprFactory;

    std::string tagNameAppend, scalarTagNameAppend;

    double rtol = 1e-6;
    int maxIter = 5;
    

    if (params->findAttribute("tolerance")) params->getAttribute("tolerance",rtol);
    if (params->findAttribute("maxiter")) params->getAttribute("maxiter",maxIter);
    if( params->findBlock("ModelBasedOnMixtureFraction") ){

      const Uintah::ProblemSpecP modelParams = params->findBlock("ModelBasedOnMixtureFraction");
      Expr::Tag rhofTag = parse_nametag( modelParams->findBlock("DensityWeightedMixtureFraction")->findBlock("NameTag") );
      Expr::Tag fTag    = parse_nametag( modelParams->findBlock("MixtureFraction"               )->findBlock("NameTag") );
      persistentFields.insert( fTag.name() ); // ensure that Uintah knows about this field

      rhofTag.reset_context( Expr::STATE_NP1 );
      if (weakForm) fTag.reset_context( Expr::STATE_NP1 );
      
      const Expr::Tag dRhodFTag = tagNames.derivative_tag(densityTag, fTag);
      
      // register placeholder for the old density
      const Expr::Tag rhoOldTag( densityTag.name(), Expr::STATE_N );

      typedef Expr::PlaceHolder<SVolField>  PlcHolder;
      factory.register_expression( new PlcHolder::Builder(rhoOldTag), true );

      // if weak-form mixture fraction is being transported, density can be evaluated from a lookup table directly
      if(weakForm)
      {
        typedef TabPropsEvaluator<SVolField>::Builder TPEval;

        // density
        factory.register_expression( new TPEval( densityTag, *densInterp, tag_list(fTag) ) );

        // \f[  \frac{d \rho}{df} \f].
        factory.register_expression( new TPEval( dRhodFTag, *densInterp, tag_list(fTag), fTag  ) );
      }
      else // strong-form transport 
      {
        typedef DensityFromMixFrac<SVolField>::Builder DensCalculator;
        const Expr::Tag fOldTag(fTag.name(), Expr::STATE_N);
          
        densCalcID =                                        
        factory.register_expression( scinew DensCalculator( densityTag, 
                                                            dRhodFTag, 
                                                            *densInterp, 
                                                            rhoOldTag, 
                                                            rhofTag, 
                                                            fOldTag, 
                                                            rtol, 
                                                            (unsigned)maxIter ));
      }

    }
    else if( params->findBlock("ModelBasedOnMixtureFractionAndHeatLoss") ){

      if( !table.has_depvar("Enthalpy") ){
        throw Uintah::ProblemSetupException( "Table has no enthalpy entry in it, but enthalpy is required for the heat loss class of models!", __FILE__, __LINE__ );
      }
      const InterpT* const enthInterp = table.find_entry("Enthalpy");

      const Uintah::ProblemSpecP modelParams = params->findBlock("ModelBasedOnMixtureFractionAndHeatLoss");
      Expr::Tag rhofTag    = parse_nametag( modelParams->findBlock("DensityWeightedMixtureFraction")->findBlock("NameTag") );
      Expr::Tag fTag       = parse_nametag( modelParams->findBlock("MixtureFraction"               )->findBlock("NameTag") );
      Expr::Tag hTag       = parse_nametag( modelParams->findBlock("Enthalpy"                      )->findBlock("NameTag") );
      Expr::Tag rhohTag    = parse_nametag( modelParams->findBlock("DensityWeightedEnthalpy"       )->findBlock("NameTag") );
      Expr::Tag heatLossTag= parse_nametag( modelParams->findBlock("HeatLoss"                      )->findBlock("NameTag") );

      typedef Expr::PlaceHolder<SVolField>  PlcHolder; 

      // If specified, set mixture fraction diffusivity Lewis number
      const Uintah::ProblemSpecP lewisNoParams = params->findBlock("LewisNumber");
      if(lewisNoParams){
        double lewisNo = 1;

        assert(lewisNoParams->findAttribute("value"));
        lewisNoParams->getAttribute("value",lewisNo);

        Expr::Tag diffusivityTag = parse_nametag( lewisNoParams->findBlock("DiffusionCoefficient")->findBlock("NameTag") );
        Expr::Tag thermCondTag   = parse_nametag( lewisNoParams->findBlock("ThermalConductivity" )->findBlock("NameTag") );
        Expr::Tag cpTag          = parse_nametag( lewisNoParams->findBlock("HeatCapacity"        )->findBlock("NameTag") );

        diffusivityTag.reset_context( Expr::STATE_NP1 );
        thermCondTag  .reset_context( Expr::STATE_NP1 );
        cpTag         .reset_context( Expr::STATE_NP1 );

        // todo: more might need to be done for cases with a turbulence model..
        factory.register_expression( scinew typename SpeciesDiffusivityFromLewisNumber<SVolField>::
                                     Builder( diffusivityTag,
                                              densityTag,
                                              thermCondTag,
                                              cpTag,
                                              lewisNo ));

        const Expr::Tag oldDiffusivityTag = Expr::Tag(diffusivityTag.name(), Expr::STATE_N);
        factory.register_expression( scinew PlcHolder::
                                     Builder(oldDiffusivityTag) );
 
        persistentFields.insert( diffusivityTag.name() );
      }

      persistentFields.insert( heatLossTag.name() ); // ensure that Uintah knows about this field
      persistentFields.insert( hTag       .name() );

      // modify name & context when we are calculating density at newer time
      // levels since this will be using STATE_NP1 information as opposed to
      // potentially STATE_N information.
      rhofTag.reset_context( Expr::STATE_NP1 );
      rhohTag.reset_context( Expr::STATE_NP1 );
      heatLossTag.reset_context( Expr::STATE_NP1 );

      const Expr::Tag rhoOldTag     ( densityTag .name(), Expr::STATE_N );
      const Expr::Tag fOldTag       ( fTag       .name(), Expr::STATE_N );
      const Expr::Tag hOldTag       ( hTag       .name(), Expr::STATE_N );
      const Expr::Tag heatLossOldTag( heatLossTag.name(), Expr::STATE_N );
      factory.register_expression( new PlcHolder::Builder(rhoOldTag     ), true );
      factory.register_expression( new PlcHolder::Builder(heatLossOldTag), true );


      const Expr::Tag dRhodHTag = tagNames.derivative_tag(densityTag,hTag);
      const Expr::Tag dRhodFTag = tagNames.derivative_tag(densityTag,fTag);
      persistentFields.insert( dRhodHTag.name() );
      persistentFields.insert( dRhodFTag.name() );
      
      factory.cleave_from_parents(factory.get_id(heatLossOldTag));
    
      typedef DensityFromMixFracAndHeatLoss<SVolField>::Builder DensCalculator;

      densCalcID = 
      factory.register_expression( scinew DensCalculator( densityTag,
                                                          heatLossTag,
                                                          dRhodFTag,
                                                          dRhodHTag,
                                                          *densInterp, 
                                                          *enthInterp,
                                                          rhoOldTag,
                                                          rhofTag,
                                                          rhohTag,
                                                          fOldTag,
                                                          hOldTag,
                                                          heatLossOldTag,
                                                          rtol,
                                                          (unsigned)maxIter ));
    gh.rootIDs.insert(densCalcID);
    }
    
    // factory.cleave_from_children(densCalcID);
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
   *  \param [inout] persistentFields the set of fields that should be controlled by
   *         Uintah and not allowed to be scratch/temporary fields.
   */
  void parse_tabprops( Uintah::ProblemSpecP& params,
                       GraphHelper& gh,
                       const Category cat,
                       const bool doDenstPlus,
                       std::set<std::string>& persistentFields,
                       const bool weakForm )
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
         ivarParams != nullptr;
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
         dvarParams != nullptr;
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
      assert( interp != nullptr );

      //____________________________________________
      // get the type of field that we will evaluate
      std::string fieldType;
      dvarParams->getWithDefault( "Type", fieldType, "SVOL" );

      switch( get_field_type(fieldType) ){
      case SVOL: {
        typedef TabPropsEvaluator<SpatialOps::SVolField>::Builder PropEvaluator;
        gh.exprFactory->register_expression( scinew PropEvaluator( dvarTag, *interp, ivarNames ) );
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
      std::string densityName;
      densityParams->findBlock("NameTag")->getAttribute( "name", densityName );
      const Expr::Tag densityTag   = Expr::Tag(densityName, Expr::STATE_NP1  );
      parse_density_solver( densityParams, table, densityTag, gh, cat, persistentFields, weakForm );
    }

  }
  //====================================================================

  void
  parse_twostream_mixing( Uintah::ProblemSpecP params,
                          const bool doDenstPlus,
                          GraphCategories& gc,
                          std::set<std::string>& persistentFields,
                          const bool weakForm)
  {
    const TagNames& tagNames = TagNames::self();
    
    Expr::Tag fTag    = parse_nametag( params->findBlock("MixtureFraction")->findBlock("NameTag") );
    const Expr::Tag rhofTag = parse_nametag( params->findBlock("DensityWeightedMixtureFraction")->findBlock("NameTag") );

    std::string densityName;
    params->findBlock("Density")->findBlock("NameTag")->getAttribute( "name", densityName );
    const Expr::Tag rhoTag  = Expr::Tag( densityName, Expr::STATE_N  );

    // Lock the density because on initialization this may be an intermediate
    // quantity, but is always needed as a guess for the solver here.
    persistentFields.insert( rhoTag.name() );

    double rho0, rho1;
    params->getAttribute("rho0",rho0);
    params->getAttribute("rho1",rho1);

    const Expr::Tag dRhoDfTag = tagNames.derivative_tag(rhoTag,fTag);
    // initial conditions for density

    {
      GraphHelper& gh = *gc[INITIALIZATION];
      typedef TwoStreamDensFromMixfr<SVolField>::Builder ICDensExpr;
      
      const Expr::Tag icRhoTag( rhoTag.name(), Expr::STATE_NONE );
      const Expr::TagList theTagList( tag_list( icRhoTag, dRhoDfTag ) );
      
      gh.rootIDs.insert( gh.exprFactory->register_expression( scinew ICDensExpr(theTagList,fTag,rho0,rho1) ) );
    }

    typedef TwoStreamMixingDensity<SVolField>::Builder DensExpr;
    typedef TwoStreamDensFromMixfr<SVolField>::Builder DensFromFExpr;
    
    Expr::ExpressionFactory& factory = *gc[ADVANCE_SOLUTION]->exprFactory;

    if (weakForm) {
      fTag.reset_context( Expr::STATE_N );
      factory.register_expression( new typename Expr::PlaceHolder<SVolField>::Builder(rhoTag) );
    } else {
      factory.register_expression( new Expr::PlaceHolder<SVolField>::Builder(rhoTag) );
    }
    

    if( doDenstPlus ){
      Expr::Tag rhoNP1Tag ( rhoTag .name(), Expr::STATE_NP1 );
      Expr::Tag fNP1Tag   ( fTag   .name(), Expr::STATE_NP1 );
      Expr::Tag rhofNP1Tag( rhofTag.name(), Expr::STATE_NP1 );

      const Expr::TagList theTagList( tag_list( rhoNP1Tag, dRhoDfTag ));
      Expr::ExpressionID id1;
      
      if (weakForm) {
        id1 = factory.register_expression( scinew DensFromFExpr(theTagList,fTag,rho0,rho1) );
      } else {
        id1 = factory.register_expression( scinew DensExpr(theTagList,rhofNP1Tag,rho0,rho1) );
      }
//      gc[ADVANCE_SOLUTION]->exprFactory->cleave_from_children(id1);
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
    for( Uintah::ProblemSpecP transEqnParams= params->findBlock("TransportEquation");
        transEqnParams != nullptr;
        transEqnParams=transEqnParams->findNextBlock("TransportEquation") )
    {
      std::string solnVarName;
      transEqnParams->get( "SolutionVariable", solnVarName );
      const Expr::Tag solnVarTagNp1( solnVarName, Expr::STATE_NONE ); // tag for rhof_{n+1}
    }
  }

  //====================================================================

  void
  setup_property_evaluation( Uintah::ProblemSpecP& wasatchSpec,
                             GraphCategories& gc,
                             std::set<std::string>& persistentFields )
  {
    //__________________________________________________________________________
    // extract the density tag in the cases that it is needed

    Uintah::ProblemSpecP densityParams  = wasatchSpec->findBlock("Density");
    
    if (!densityParams) {
      std::ostringstream msg;
      msg << std::endl << "Error: You must specify a <Density> block. Please revise your input file." << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    
    Uintah::ProblemSpecP tabPropsParams = wasatchSpec->findBlock("TabProps");
    Uintah::ProblemSpecP radPropsParams = wasatchSpec->findBlock("RadProps");
    Uintah::ProblemSpecP twoStreamParams= wasatchSpec->findBlock("TwoStreamMixing");

    if( radPropsParams ){
      parse_radprops( radPropsParams, *gc[ADVANCE_SOLUTION] );
    }

    const bool isConstDensity = densityParams->findBlock("Constant");
    const bool doDenstPlus = !isConstDensity
                           && wasatchSpec->findBlock("MomentumEquations")
                           && wasatchSpec->findBlock("TransportEquation");

    // find out if we are using a weak formulation for scalar transport
    bool weakForm = false;
    if (wasatchSpec->findBlock("TransportEquation")) {
      std::string eqnLabel;
      wasatchSpec->findBlock("TransportEquation")->getAttribute( "equation", eqnLabel );
      if (eqnLabel=="enthalpy" || eqnLabel=="mixturefraction") {
        std::string form;
        wasatchSpec->findBlock("TransportEquation")->getAttribute( "form", form );
        if (form == "weak") {
          weakForm = true;
        }
      }
    }
    
    if( twoStreamParams ){
      parse_twostream_mixing( twoStreamParams, doDenstPlus, gc, persistentFields, weakForm );
    }

    // TabProps
    for( Uintah::ProblemSpecP tabPropsParams = wasatchSpec->findBlock("TabProps");
         tabPropsParams != nullptr;
         tabPropsParams = tabPropsParams->findNextBlock("TabProps") )
    {
      const Category cat = parse_tasklist( tabPropsParams,false);
      parse_tabprops( tabPropsParams, *gc[cat], cat, doDenstPlus, persistentFields, weakForm );
    }
  }

  //====================================================================

} // namespace WasatchCore
