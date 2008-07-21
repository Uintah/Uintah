#ifndef PUDA_H
#define PUDA_H

#include <string>

namespace Uintah {

  struct CommandLineFlags 
  {
    bool do_timesteps;
    bool do_gridstats;
    bool do_listvars;
    bool do_varsummary;
    bool do_jim1;
    bool do_jim2;
    bool do_jim3;
    bool do_partvar;
    bool do_asci;
    bool do_cell_stresses;
    bool do_rtdata;
    bool do_NCvar_double;
    bool do_NCvar_float;
    bool do_NCvar_point;
    bool do_NCvar_vector;
    bool do_NCvar_matrix3;
    bool do_CCvar_double;
    bool do_CCvar_float;
    bool do_CCvar_point;
    bool do_CCvar_vector;
    bool do_CCvar_matrix3;
    bool do_PTvar;
    bool do_PTvar_all;
    bool do_patch;
    bool do_material;
    bool do_verbose;
    bool do_tecplot;
    bool do_all_ccvars;

    bool use_extra_cells;

    unsigned long time_step_lower;
    unsigned long time_step_upper;
    unsigned long time_step_inc;
    bool tslow_set;
    bool tsup_set;
    int tskip;
    int matl_jim;
    std::string i_xd;
    std::string filebase;
    std::string raydatadir;
    std::string particleVariable;
    std::string ccVarInput;

    CommandLineFlags() {
      do_timesteps = false;
      do_gridstats = false;
      do_listvars = false;
      do_varsummary = false;
      do_jim1 = false;
      do_jim2 = false;
      do_jim3 = false;
      do_partvar = false;
      do_asci = false;
      do_cell_stresses = false;
      do_rtdata  =  false;
      do_NCvar_double  =  false;
      do_NCvar_float  =  false;
      do_NCvar_point  =  false;
      do_NCvar_vector  =  false;
      do_NCvar_matrix3  =  false;
      do_CCvar_double  =  false;
      do_CCvar_float  =  false;
      do_CCvar_point  =  false;
      do_CCvar_vector  =  false;
      do_CCvar_matrix3  =  false;
      do_PTvar  =  false;
      do_PTvar_all  =  true;
      do_patch  =  false;
      do_material  =  false;
      do_verbose = false;
      do_tecplot = false;
      do_all_ccvars = false;

      use_extra_cells = true;

      time_step_lower = 0;
      time_step_upper = 1;
      time_step_inc = 1;
      tslow_set = false;
      tsup_set = false;
      tskip = 1;
      matl_jim  = 0;
    }
  };

} // end namespace Uintah

#endif
