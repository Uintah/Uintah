#ifndef CORE_GRID_VARIABLES_STATICINSTANTIATE
#define CORE_GRID_VARIABLES_STATICINSTANTIATE

/*
 * The MIT License
 *
 * Copyright (c) 2018 The University of Utah
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


namespace Uintah {

  // This function is (indirectly) called by support programs (such as
  // puda, compare_uda, etc) when built in a static build.
  // (Specifically, called when the 1st DataArchive is created.)  This
  // function will cause all (most - some more types may need to be
  // listed if a problem is found) of our Uintah variable types to be
  // instantiated (and thus registered with the Uintah type system).
  // This normally happens (I think) for shared lib builds when the
  // CCVariable (etc) class is loaded (and static constructors fire).
  // However, for some (unknown) reason (perhaps the optimizer has
  // removed them?), this does not happen with static builds, and thus
  // when a tool (such as puda) goes to load a variable, the Uintah
  // type system throws an error because it doesn't know what type of
  // variables (eg: CCVariable) exist.

  void instantiateVariableTypes();
  
}

#endif
