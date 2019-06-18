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
 * @file CCA/Components/PhaseField/Applications/Benchmark.cc
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 *
 * In this the names used by ApplicationFactory for the differnt implementations
 * of the Benchmark applications are defined as well as their explicit
 * instantiation.
 */

#include <CCA/Components/PhaseField/Applications/Benchmark01.h>
#include <CCA/Components/PhaseField/Applications/Benchmark02.h>
#include <CCA/Components/PhaseField/Applications/Benchmark03.h>
#include <CCA/Components/PhaseField/Applications/Benchmark04.h>

namespace Uintah {
namespace PhaseField {

#ifndef _DOXY_IGNORE_
template<> const std::string Benchmark01<CC,P5>::Name = "benchmark01|cc|d2|p5";
template<> const std::string Benchmark01<NC,P5>::Name = "benchmark01|nc|d2|p5";
template<> const std::string Benchmark02<CC,P5>::Name = "benchmark02|cc|d2|p5";
template<> const std::string Benchmark02<NC,P5>::Name = "benchmark02|nc|d2|p5";
template<> const std::string Benchmark03<CC,P3>::Name = "benchmark03|cc|d1|p3";
template<> const std::string Benchmark03<NC,P3>::Name = "benchmark03|nc|d1|p3";
template<> const std::string Benchmark04<CC,P5>::Name = "benchmark04|cc|d2|p5";
template<> const std::string Benchmark04<NC,P5>::Name = "benchmark04|nc|d2|p5";

template class Benchmark01<CC,P5>;
template class Benchmark01<NC,P5>;
template class Benchmark02<CC,P5>;
template class Benchmark02<NC,P5>;
template class Benchmark03<CC,P3>;
template class Benchmark03<NC,P3>;
template class Benchmark04<CC,P5>;
template class Benchmark04<NC,P5>;
#endif

} // namespace Uintah
} // namespace PhaseField
