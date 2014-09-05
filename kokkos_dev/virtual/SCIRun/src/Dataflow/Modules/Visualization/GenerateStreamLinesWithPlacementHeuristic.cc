/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  GenerateStreamLinesWithPlacementHeuristic.h:  Create optimal stream lines
 *
 *  Written by:
 *   Frank B. Sachse
 *   CVRTI
 *   University of Utah
 *   February 2004, JULY 2004
 */

#include <Dataflow/Modules/Visualization/GenerateStreamLinesWithPlacementHeuristic.h>

namespace SCIRun {

  DECLARE_MAKER(GenerateStreamLinesWithPlacementHeuristic)

  CompileInfoHandle
  GenerateStreamLinesWithPlacementHeuristicAlgo::get_compile_info(const TypeDescription *fsrc,
                                               const TypeDescription *fvec,
                                               const TypeDescription *fsp
                                               )
  {
    //cerr << "GenerateStreamLinesWithPlacementHeuristicAlgo::get_compile_info started\n";
 
    // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
    static const string include_path(TypeDescription::cc_to_h(__FILE__));
    static const string template_class_name("GenerateStreamLinesWithPlacementHeuristicAlgoT");
    static const string base_class_name("GenerateStreamLinesWithPlacementHeuristicAlgo");
    
    CompileInfo *rval = 
      scinew CompileInfo(template_class_name + "." +
                         fsrc->get_filename() + "." +
			 fvec->get_filename() + "." +
                         fsp->get_filename() + ".",
                         base_class_name, 
                         template_class_name, 
                         fsrc->get_name() + ", " +
                         fvec->get_name() + ", " +
                         fsp->get_name() 
                          );
    
    // Add in the include path to compile this obj
    rval->add_basis_include("../src/Core/Basis/CrvLinearLgn.h");
    rval->add_mesh_include("../src/Core/Datatypes/CurveMesh.h");
    fsrc->fill_compile_info(rval);
    fvec->fill_compile_info(rval);
    fsp->fill_compile_info(rval);
    rval->add_mesh_include(include_path);
  
    //cerr << "GenerateStreamLinesWithPlacementHeuristicAlgo::get_compile_info finished\n";

    return rval;
  }
  
} // End namespace SCIRun
