/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#ifndef Wasatch_CoordinateHelper_h
#define Wasatch_CoordinateHelper_h

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>

//-- Uintah Framework Includes --//
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SchedulerP.h>

//-- ExprLib includes --//
#include <expression/Tag.h>

#include <CCA/Components/Wasatch/GraphHelperTools.h>


namespace Expr{ class ExpressionFactory; }

namespace WasatchCore{
  
  class CoordinateNames
  {
  public:
    
    typedef std::map<Expr::Tag, std::string> CoordMap;

    /**
     *  Access the VolumeFraction names and tags.
     */
    static const std::map<Expr::Tag, std::string>& coordinate_map();
    
  private:
    static const CoordinateNames& self();
    CoordMap coordMap_;
    CoordinateNames();
  };

  void register_coordinate_expressions( GraphCategories& gc,
                                        const bool isPeriodic );

} // namespace WasatchCore

#endif // Wasatch_CoordHelper_h
