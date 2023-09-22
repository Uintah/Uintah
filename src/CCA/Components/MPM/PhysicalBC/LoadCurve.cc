/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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
 
 
#include <CCA/Components/MPM/PhysicalBC/LoadCurve.h>

using namespace Uintah;


//______________________________________________________________________
//
// Construct a load curve from the problem spec
template<class T>
LoadCurve<T>::LoadCurve(ProblemSpecP& ps) 
{
  ProblemSpecP loadCurve = ps->findBlock("load_curve");
  if (!loadCurve){ 
     throw ProblemSetupException("**ERROR** No load curve specified.", 
                                  __FILE__, __LINE__);
  }

  loadCurve->require("id", d_id);
  loadCurve->getWithDefault("material", d_matl, -99);

  for( ProblemSpecP timeLoad = loadCurve->findBlock("time_point");
       timeLoad != nullptr;
       timeLoad = timeLoad->findNextBlock("time_point") ) {

    double time = 0.0;
    T load;

    timeLoad->require("time", time);
    timeLoad->require("load", load);
    d_time.push_back(time);
    d_load.push_back(load);
  }

  //__________________________________
  // bulletproofing
  for(int i = 1; i<(int)d_time.size(); i++){
  
    std::cout << " time: " << d_time[i] << std::endl;
    std::ostringstream msg;
    msg << "**ERROR** LoadCurve: id("<<d_id<<")";
    
    if ( d_time[i]==d_time[i-1] ){
      throw ProblemSetupException( msg.str() + " Identical <time> entries.",
                                   __FILE__, __LINE__);
    }
    if ( 0 > d_time[i] ){
      throw ProblemSetupException(msg.str() + " negative <time> entry detected.",
                                   __FILE__, __LINE__);
    }

    if ( d_time[i]<=d_time[i-1] ){
      throw ProblemSetupException( msg.str() + " <time> must increase between points.",
                                   __FILE__, __LINE__);
    }
  }
}
//______________________________________________________________________
//
template<class T>
void LoadCurve<T>::outputProblemSpec(ProblemSpecP& ps) 
{
  ProblemSpecP lc_ps = ps->appendChild("load_curve");
  lc_ps->appendElement("id",d_id);
  lc_ps->appendElement("material",d_matl);

  for (int i = 0; i<(int)d_time.size();i++) {
    ProblemSpecP time_ps = lc_ps->appendChild("time_point");
    time_ps->appendElement("time",d_time[i]);
    time_ps->appendElement("load",d_load[i]);
  }
}


  
//______________________________________________________________________
// Instantiate the explicit template instantiations.
//
template LoadCurve<double>::LoadCurve( ProblemSpecP& ps );
template LoadCurve<Vector>::LoadCurve( ProblemSpecP& ps );

template void LoadCurve<double>::outputProblemSpec( ProblemSpecP& ps );
template void LoadCurve<Vector>::outputProblemSpec( ProblemSpecP& ps );
