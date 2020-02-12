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

/*
 *  puda.cc: Print out a uintah data archive
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 2000
 *
 */

#include <StandAlone/tools/puda/puda.h>

#include <Core/DataArchive/DataArchive.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Parallel/Parallel.h>

#include <StandAlone/tools/puda/AA_MMS.h>
#include <StandAlone/tools/puda/asci.h>
#include <StandAlone/tools/puda/ER_MMS.h>
#include <StandAlone/tools/puda/gridStats.h>
#include <StandAlone/tools/puda/GV_MMS.h>
#include <StandAlone/tools/puda/ICE_momentum.h>
#include <StandAlone/tools/puda/jacquie.h>
#include <StandAlone/tools/puda/jim1.h>
#include <StandAlone/tools/puda/jim2.h>
#include <StandAlone/tools/puda/DOP.h>
#include <StandAlone/tools/puda/PIC.h>
#include <StandAlone/tools/puda/POL.h>
#include <StandAlone/tools/puda/pressure.h>
#include <StandAlone/tools/puda/printCellStresses.h>
#include <StandAlone/tools/puda/printParticleVar.h>
#include <StandAlone/tools/puda/pStressHistogram.h>
#include <StandAlone/tools/puda/todd1.h>
#include <StandAlone/tools/puda/util.h>
#include <StandAlone/tools/puda/varsummary.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
using namespace Uintah;


/////////////////////////////////////////////////////////////////

void
usage( const std::string& badarg, const std::string& progname )
{
  if(badarg != "") {
    cerr << "Error parsing argument: " << badarg << "\n";
  }
  cerr << "Usage: " << progname << " [options] <archive file>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -h[elp]\n";
  cerr << "  -timesteps\n";
  cerr << "  -gridstats\n";
  cerr << "  -listvariables\n";
  cerr << "  -varsummary          (output min/max of all variables for each patch) \n";
  cerr << "  -brief               (Makes varsummary print out a subset of information.)\n";
  cerr << "  -jim1\n";
  cerr << "  -jim2\n";
  cerr << "  -DOP\n";
  cerr << "  -todd1               ( 1st Law of thermo. control volume analysis) \n";
  cerr << "  -ICE_momentum        ( momentum control volume analysis) \n";
  cerr << "  -jacquie             (finds burn rate vs pressure)\n";
  cerr << "  -pressure            (finds  pressure)\n";
  cerr << "  -AA_MMS_1            (1D periodic bar MMS)\n";
  cerr << "  -AA_MMS_2            (3D Axis aligned MMS)\n";
  cerr << "  -GV_MMS              (GeneralizedVortex MMS)\n"; //MMS
  cerr << "  -ER_MMS              (Expanding Ring MMS)\n"; 
  cerr << "  -partvar <variable name>\n";
  cerr << "  -asci\n";
  cerr << "  -no_extra_cells      (Excludes extra cells when iterating over cells.\n";
  cerr << "                        Default is to include extra cells.)\n";
  cerr << "  -cell_stresses       (output nodal stresses )\n";
  cerr << "  -verbose             (prints status of output)\n";
  cerr << "  -timesteplow  <int>  (only outputs timestep from int)\n";
  cerr << "  -timestephigh <int>  (only outputs timesteps upto int)\n";
  cerr << "  -matl         <int>  (only outputs data for matl)\n";
  cerr << "  -pic                 (prints particle ids of all particles  in cell\n";
  cerr << "                        <i> <j> <k> [ints] on the specified timesteps)\n";
  cerr << "  -pol                 (prints out average of all particles in a cell over an\n";
  cerr << "                       entire line on a line of cells and is called with:\n";
  cerr << "                       <axis: [x,y,z]> <ortho1> <ortho2> <average; default=true>\n";
  cerr << "                       <stressSplitting; default=false>\n";
  cerr << "                       'ortho1' and 'ortho2' inidicate the coordinates in the plane\n";
  cerr << "                       orthogonal to 'axis'.  'average' tells whether to average\n";
  cerr << "                       over all particles in the cell, or just to use the first\n";
  cerr << "                       particle encountered.  'stressSplitting' only takes affect\n";
  cerr << "                       if the particle variable is p.stress, and splits the stress\n";
  cerr << "                       into hydrostatic and deviatoric parts.)\n";
    
  cerr << "USAGE IS NOT FINISHED\n\n";
  exit( 1 );
}
//______________________________________________________________________
//______________________________________________________________________
//
int
main( int argc, char *argv[] )
{
  Uintah::Parallel::initializeManager(argc, argv);

  if (argc <= 1) {
    // Print out the usage and die
    usage("", argv[0]);
  }

  CommandLineFlags clf;

  // defaults
  int material_of_interest = -1; //not part of clf
  int cellx   = -1; 
  int celly   = -1; 
  int cellz   = -1;
  char axis   = 'n';
  int ortho1  = -1;
  int ortho2  = -1;
  int dir = 1;
  bool doPOLAverage     = true;
  bool doPOLStressSplit = false;

  // set defaults for cout.
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(8);

  //______________________________________________________________________
  //   Parse arguments

  for(int i=1;i<argc;i++){
    string s = argv[i];
    
    if(s == "-timesteps"){
      clf.do_timesteps=true;
    } 
    else if( s == "-no_extra_cells" ||
             s == "-no_extracells" ||
             s == "-no_ExtraCells" ) {
      clf.use_extra_cells = false;
    } 
    else if(s == "-gridstats" ||
              s == "-gridStats" ||
              s == "-grid_stats"){
      clf.do_gridstats=true;
    } 
    else if(s == "-listvariables" || 
              s == "-listVariables" || 
              s == "-list_variables"){
      clf.do_listvars=true;
    } 
    else if(s == "-varsummary" ||
            s == "-varSummary" ||
            s == "-var_summary"){
      clf.do_varsummary = true;
    } 
    else if(s == "-brief" ) {
      clf.be_brief = true;
    } 
    else if(s == "-jacquie"){
      clf.do_jacquie = true;
    } 
    else if(s == "-pressure"){
      clf.do_pressure = true;
    } 
    else if(s == "-jim1"){
      clf.do_jim1 = true;
    } 
    else if(s == "-jim2"){
      clf.do_jim2 = true;
    } 
    else if(s == "-DOP"){
      clf.do_DOP = true;
    } 
    else if(s == "-todd1"){
      clf.do_todd1 = true;
    } 
    else if(s == "-ICE_momentum"){
      clf.do_ice_momentum = true;
    }
    else if(s == "-pStressHistogram"){
      clf.do_pStressHstgrm = true;
    }
    else if(s == "-pic"){
      clf.do_PIC = true;

      if(i+3 >= argc){
        usage("-pic", argv[0]);
      } 

      cellx = strtoul(argv[++i],(char**)nullptr,10);
      celly = strtoul(argv[++i],(char**)nullptr,10);
      cellz = strtoul(argv[++i],(char**)nullptr,10);
    } 
    else if(s == "-pol") {
      if(i+3 >= argc){
        usage("-pol", argv[0]);
      } 

      axis = *argv[++i];
      ortho1 = strtoul(argv[++i],(char**)nullptr,10);
      ortho2 = strtoul(argv[++i],(char**)nullptr,10);

      clf.do_POL = true;

      // check if optional arguments were found
      if(i+1 < argc){
        if(string(argv[i+1]) == "true"){
          doPOLAverage = true;
          i++;
        } 
        else if(string(argv[i+1]) == "false"){
          doPOLAverage = false;
          i++;
        }
      }

      if(i+1 < argc){
        if(string(argv[i+1]) == "true"){
          doPOLStressSplit = true;
          i++;
        } 
        else if(string(argv[i+1]) == "false"){
          doPOLStressSplit = false;
          i++;
        }
      }
    }
    else if(s == "-AA_MMS_1"){
      clf.do_AA_MMS_1 = true;
    } 
    else if(s == "-AA_MMS_2"){
      clf.do_AA_MMS_2 = true;
    } 
    else if(s == "-GV_MMS"){
      clf.do_GV_MMS = true;
    } 
    else if(s == "-ER_MMS"){
      clf.do_ER_MMS = true;
    } 
    else if(s == "-partvar"){
      clf.do_partvar = true;
      if(i+1 >= argc){
        usage("-partvar",argv[0]);
      }
      clf.particleVariable = argv[++i]; 
      if (clf.particleVariable[0] == '-') {
        usage("-partvar <particle variable name>", argv[0]);
      }
    } 
    else if(s == "-asci"){
      clf.do_asci=true;
    } 
    else if(s == "-dir"){
      clf.dir = atoi(argv[++i]); 
    } 
    else if(s == "-cell_stresses"){
      clf.do_cell_stresses=true;
    } 
    else if (s == "-material" ||
             s == "-matl"     || 
             s == "-mat") {
      if(i+1 >= argc){
        usage("-mat", argv[0]);
      }
      clf.matl = strtoul(argv[++i],(char**)nullptr,10);
      material_of_interest = clf.matl;

    } 
    else if (s == "-verbose") {
      clf.do_verbose = true;
    } 
    else if (s == "-timesteplow" ||
             s == "-timeStepLow" ||
             s == "-timestep_low") {
      if(i+1 >= argc){
        usage("-timesteplow", argv[0]);
      }
      clf.time_step_lower = strtoul(argv[++i],(char**)nullptr,10);
      clf.tslow_set = true;
    } 
    else if (s == "-timestephigh" ||
             s == "-timeStepHigh" ||
             s == "-timestep_high") {
      if(i+1 >= argc){
        usage("-timestephigh", argv[0]);
      }
      clf.time_step_upper = strtoul(argv[++i],(char**)nullptr,10);
      clf.tsup_set = true;
    } else if (s == "-timestepinc" ||
               s == "-timestepInc" ||
               s == "-timestep_inc") {
      if(i+1 >= argc) {
        usage("-timestepinc", argv[0]);
      }
      clf.time_step_inc = strtoul(argv[++i],(char**)nullptr,10);
    } else if( s == "-help" || 
               s == "-h" ) {
      usage( "", argv[0] );
    } else if( clf.filebase == "") {

      if( argv[i][0] == '-' ) { // File name can't start with a dash.
        usage( s, argv[0]);
      }
      clf.filebase = argv[i];
    } 
    else {
      usage( s, argv[0]);
    }
  }

  if( clf.filebase == "" ) {
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }
  
  //______________________________________________________________________
  //
  try {
    DataArchive* da = scinew DataArchive( clf.filebase );
    
    //__________________________________
    //  LIST TIMESTEPS
    if(clf.do_timesteps){
      vector<int> index;
      vector<double> times;
      da->queryTimesteps( index, times );
      ASSERTEQ( index.size(), times.size() );
      cout << "There are " << index.size() << " timesteps:\n";
      
      // Please don't change this.  We need 16
      // significant digits for detailed comparative studies. -Todd
      cout.setf(ios::scientific,ios::floatfield);
      cout.precision(16);
      
      for(int i=0;i<(int)index.size();i++) {
        cout << index[i] << ": " << times[i] << "\n";
      }
    }
    
    //__________________________________
    //  LIST VARIABLES
    if( clf.do_listvars ){
      vector<string> vars;
      vector<int> num_matls;
      vector<const Uintah::TypeDescription*> types;
      da->queryVariables( vars, num_matls, types );
      
      cout << "There are " << vars.size() << " variables:\n";
      for( int i = 0; i < (int)vars.size(); i++ ){
        cout << vars[i] << ": " << types[i]->getName() << "\n";
      }
    }

    //__________________________________
    // 
    if( clf.do_gridstats ){
      gridstats( da, clf );
    }
    
    if ( clf.do_partvar && !clf.do_POL ) {
      printParticleVariable( da, clf, material_of_interest );
    }

    if( clf.do_varsummary ){
      varsummary( da, clf, material_of_interest );
    }

    if( clf.do_pressure ){
      pressure( da, clf );
    }

    if( clf.do_jim1 ){
      jim1( da, clf );
    }

    if( clf.do_jacquie ){
      jacquie( da, clf );
    }

    if( clf.do_jim2 ){
      jim2( da, clf );
    }

    if( clf.do_DOP ){
      DOP( da, clf );
    }

    if( clf.do_todd1 ){
      todd1( da, clf );
    }
    
    if( clf.do_ice_momentum ){
      ICE_momentum( da, clf );
    }

    if( clf.do_PIC ){
      PIC( da, clf, cellx, celly, cellz );
    }

    if( clf.do_POL ){
      POL( da, clf, axis, ortho1, ortho2, doPOLAverage, doPOLStressSplit );
    }
    
    if( clf.do_pStressHstgrm ){
      pStressHistogram( da, clf );
    }
    
    if( clf.do_AA_MMS_1 || clf.do_AA_MMS_2 ){
      AA_MMS( da, clf );
    }

    if( clf.do_GV_MMS ){
      GV_MMS( da, clf );
    }
    
    if( clf.do_ER_MMS ){
      ER_MMS( da, clf );
    }

    if ( clf.do_asci ){
      asci( da, clf.tslow_set, clf.tsup_set, clf.time_step_lower, clf.time_step_upper );
    }

    if ( clf.do_cell_stresses ){
      printCellStresses( da, clf, material_of_interest );
    } 

  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << "\n";
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }

  return 0;

} // end main()
