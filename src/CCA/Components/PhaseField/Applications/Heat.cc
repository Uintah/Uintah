/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

/**
 * @file CCA/Components/PhaseField/Applications/Heat.cc
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 *
 * In this the names used by ApplicationFactory for the differnt implementations
 * of the Heat application are defined as well as their explicit instantiation.
 */

#include <CCA/Components/PhaseField/Applications/Heat.h>

namespace Uintah {
namespace PhaseField {

#ifndef _DOXY_IGNORE_
template<> const std::string HeatProblem<CC, P5>::Name = "HeatProblem";
template<> const std::string HeatProblem<NC, P5>::Name = "HeatProblem";
template<> const std::string HeatProblem<CC, P7>::Name = "HeatProblem";
template<> const std::string HeatProblem<NC, P7>::Name = "HeatProblem";

template<> const std::string Heat<CC,D2,P5>::Name = "heat|cc|d2|p5";
template<> const std::string Heat<NC,D2,P5>::Name = "heat|nc|d2|p5";
template<> const std::string Heat<CC,D3,P7>::Name = "heat|cc|d3|p7";
template<> const std::string Heat<NC,D3,P7>::Name = "heat|nc|d3|p7";

template<> const std::string Heat<CC,D2,P5,AMR>::Name = "amr|heat|cc|d2|p5";
template<> const std::string Heat<NC,D2,P5,AMR>::Name = "amr|heat|nc|d2|p5";
template<> const std::string Heat<CC,D3,P7,AMR>::Name = "amr|heat|cc|d3|p7";
template<> const std::string Heat<NC,D3,P7,AMR>::Name = "amr|heat|nc|d3|p7";

template class Heat<CC,D2,P5>;
template class Heat<NC,D2,P5>;
template class Heat<CC,D3,P7>;
template class Heat<NC,D3,P7>;

template class Heat<CC,D2,P5,AMR>;
template class Heat<NC,D2,P5,AMR>;
template class Heat<CC,D3,P7,AMR>;
template class Heat<NC,D3,P7,AMR>;
#endif

} // namespace Uintah
} // namespace PhaseField
