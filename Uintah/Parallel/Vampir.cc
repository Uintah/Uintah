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







