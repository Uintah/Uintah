/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifdef USE_VAMPIR

#include "Vampir.h"

void VTsetup()
{
  VT_symdef(VT_SEND_PARTICLES, "SendParticles", "Send");
  VT_symdef(VT_RECV_PARTICLES, "RecvParticles", "Recv");
  VT_symdef(VT_SEND_INITDATA, "SendInitData", "Send");
  VT_symdef(VT_RECV_INITDATA, "RecvInitData", "Recv");
  VT_symdef(VT_CHECKSUM, "CheckSum", "AllReduce");
  VT_symdef(VT_SEND_COMPUTES, "SendComputes", "Send");
  VT_symdef(VT_RECV_DEPENDENCIES, "RecvDependencies", "Recv");
  VT_symdef(VT_PERFORM_TASK, "PerformTask", "Calculation");
  VT_symdef(VT_EXECUTE, "Execute", "Calculation");
  VT_traceon();
}

#endif







