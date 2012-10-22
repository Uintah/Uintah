/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

//-------------------------- MPI_Abort ------------------------
#define NAME     Abort
#define TEXTNAME "MPI_Abort"
#define CALLSIG  MPI_Comm arg0, int arg1
#define VARS     MPI_Comm arg0; int arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Accumulate ------------------------
#define NAME     Accumulate
#define TEXTNAME "MPI_Accumulate"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, MPI_Aint arg4, int arg5, MPI_Datatype arg6, MPI_Op arg7, MPI_Win arg8
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; MPI_Aint arg4; int arg5; MPI_Datatype arg6; MPI_Op arg7; MPI_Win arg8; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), arg7(arg7), arg8(arg8), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Address ------------------------
#define NAME     Address
#define TEXTNAME "MPI_Address"
#define CALLSIG  void* arg0, MPI_Aint* arg1
#define VARS     void* arg0; MPI_Aint* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Allgather ------------------------
#define NAME     Allgather
#define TEXTNAME "MPI_Allgather"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, void* arg3, int arg4, MPI_Datatype arg5, MPI_Comm arg6
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; void* arg3; int arg4; MPI_Datatype arg5; MPI_Comm arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Allgatherv ------------------------
#define NAME     Allgatherv
#define TEXTNAME "MPI_Allgatherv"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, void* arg3, int* arg4, int* arg5, MPI_Datatype arg6, MPI_Comm arg7
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; void* arg3; int* arg4; int* arg5; MPI_Datatype arg6; MPI_Comm arg7; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), arg7(arg7), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Alloc_mem ------------------------
#define NAME     Alloc_mem
#define TEXTNAME "MPI_Alloc_mem"
#define CALLSIG  MPI_Aint arg0, MPI_Info arg1, void* arg2
#define VARS     MPI_Aint arg0; MPI_Info arg1; void* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Allreduce ------------------------
#define NAME     Allreduce
#define TEXTNAME "MPI_Allreduce"
#define CALLSIG  void* arg0, void* arg1, int arg2, MPI_Datatype arg3, MPI_Op arg4, MPI_Comm arg5
#define VARS     void* arg0; void* arg1; int arg2; MPI_Datatype arg3; MPI_Op arg4; MPI_Comm arg5; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Alltoall ------------------------
#define NAME     Alltoall
#define TEXTNAME "MPI_Alltoall"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, void* arg3, int arg4, MPI_Datatype arg5, MPI_Comm arg6
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; void* arg3; int arg4; MPI_Datatype arg5; MPI_Comm arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Alltoallv ------------------------
#define NAME     Alltoallv
#define TEXTNAME "MPI_Alltoallv"
#define CALLSIG  void* arg0, int* arg1, int* arg2, MPI_Datatype arg3, void* arg4, int* arg5, int* arg6, MPI_Datatype arg7, MPI_Comm arg8
#define VARS     void* arg0; int* arg1; int* arg2; MPI_Datatype arg3; void* arg4; int* arg5; int* arg6; MPI_Datatype arg7; MPI_Comm arg8; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), arg7(arg7), arg8(arg8), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Attr_delete ------------------------
#define NAME     Attr_delete
#define TEXTNAME "MPI_Attr_delete"
#define CALLSIG  MPI_Comm arg0, int arg1
#define VARS     MPI_Comm arg0; int arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Attr_get ------------------------
#define NAME     Attr_get
#define TEXTNAME "MPI_Attr_get"
#define CALLSIG  MPI_Comm arg0, int arg1, void* arg2, int* arg3
#define VARS     MPI_Comm arg0; int arg1; void* arg2; int* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Attr_put ------------------------
#define NAME     Attr_put
#define TEXTNAME "MPI_Attr_put"
#define CALLSIG  MPI_Comm arg0, int arg1, void* arg2
#define VARS     MPI_Comm arg0; int arg1; void* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Barrier ------------------------
#define NAME     Barrier
#define TEXTNAME "MPI_Barrier"
#define CALLSIG  MPI_Comm arg0
#define VARS     MPI_Comm arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Bcast ------------------------
#define NAME     Bcast
#define TEXTNAME "MPI_Bcast"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, MPI_Comm arg4
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; MPI_Comm arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Bsend ------------------------
#define NAME     Bsend
#define TEXTNAME "MPI_Bsend"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Bsend_init ------------------------
#define NAME     Bsend_init
#define TEXTNAME "MPI_Bsend_init"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5, MPI_Request* arg6
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; MPI_Request* arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Buffer_attach ------------------------
#define NAME     Buffer_attach
#define TEXTNAME "MPI_Buffer_attach"
#define CALLSIG  void* arg0, int arg1
#define VARS     void* arg0; int arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Buffer_detach ------------------------
#define NAME     Buffer_detach
#define TEXTNAME "MPI_Buffer_detach"
#define CALLSIG  void* arg0, int* arg1
#define VARS     void* arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Cancel ------------------------
#define NAME     Cancel
#define TEXTNAME "MPI_Cancel"
#define CALLSIG  MPI_Request* arg0
#define VARS     MPI_Request* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Cart_coords ------------------------
#define NAME     Cart_coords
#define TEXTNAME "MPI_Cart_coords"
#define CALLSIG  MPI_Comm arg0, int arg1, int arg2, int* arg3
#define VARS     MPI_Comm arg0; int arg1; int arg2; int* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Cart_create ------------------------
#define NAME     Cart_create
#define TEXTNAME "MPI_Cart_create"
#define CALLSIG  MPI_Comm arg0, int arg1, int* arg2, int* arg3, int arg4, MPI_Comm* arg5
#define VARS     MPI_Comm arg0; int arg1; int* arg2; int* arg3; int arg4; MPI_Comm* arg5; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Cartdim_get ------------------------
#define NAME     Cartdim_get
#define TEXTNAME "MPI_Cartdim_get"
#define CALLSIG  MPI_Comm arg0, int* arg1
#define VARS     MPI_Comm arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Cart_get ------------------------
#define NAME     Cart_get
#define TEXTNAME "MPI_Cart_get"
#define CALLSIG  MPI_Comm arg0, int arg1, int* arg2, int* arg3, int* arg4
#define VARS     MPI_Comm arg0; int arg1; int* arg2; int* arg3; int* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Cart_map ------------------------
#define NAME     Cart_map
#define TEXTNAME "MPI_Cart_map"
#define CALLSIG  MPI_Comm arg0, int arg1, int* arg2, int* arg3, int* arg4
#define VARS     MPI_Comm arg0; int arg1; int* arg2; int* arg3; int* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Cart_rank ------------------------
#define NAME     Cart_rank
#define TEXTNAME "MPI_Cart_rank"
#define CALLSIG  MPI_Comm arg0, int* arg1, int* arg2
#define VARS     MPI_Comm arg0; int* arg1; int* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Cart_shift ------------------------
#define NAME     Cart_shift
#define TEXTNAME "MPI_Cart_shift"
#define CALLSIG  MPI_Comm arg0, int arg1, int arg2, int* arg3, int* arg4
#define VARS     MPI_Comm arg0; int arg1; int arg2; int* arg3; int* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Cart_sub ------------------------
#define NAME     Cart_sub
#define TEXTNAME "MPI_Cart_sub"
#define CALLSIG  MPI_Comm arg0, int* arg1, MPI_Comm* arg2
#define VARS     MPI_Comm arg0; int* arg1; MPI_Comm* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_compare ------------------------
#define NAME     Comm_compare
#define TEXTNAME "MPI_Comm_compare"
#define CALLSIG  MPI_Comm arg0, MPI_Comm arg1, int* arg2
#define VARS     MPI_Comm arg0; MPI_Comm arg1; int* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_create ------------------------
#define NAME     Comm_create
#define TEXTNAME "MPI_Comm_create"
#define CALLSIG  MPI_Comm arg0, MPI_Group arg1, MPI_Comm* arg2
#define VARS     MPI_Comm arg0; MPI_Group arg1; MPI_Comm* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_dup ------------------------
#define NAME     Comm_dup
#define TEXTNAME "MPI_Comm_dup"
#define CALLSIG  MPI_Comm arg0, MPI_Comm* arg1
#define VARS     MPI_Comm arg0; MPI_Comm* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_free ------------------------
#define NAME     Comm_free
#define TEXTNAME "MPI_Comm_free"
#define CALLSIG  MPI_Comm* arg0
#define VARS     MPI_Comm* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_get_attr ------------------------
#define NAME     Comm_get_attr
#define TEXTNAME "MPI_Comm_get_attr"
#define CALLSIG  MPI_Comm arg0, int arg1, void* arg2, int* arg3
#define VARS     MPI_Comm arg0; int arg1; void* arg2; int* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_get_name ------------------------
#define NAME     Comm_get_name
#define TEXTNAME "MPI_Comm_get_name"
#define CALLSIG  MPI_Comm arg0, char* arg1, int* arg2
#define VARS     MPI_Comm arg0; char* arg1; int* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_group ------------------------
#define NAME     Comm_group
#define TEXTNAME "MPI_Comm_group"
#define CALLSIG  MPI_Comm arg0, MPI_Group* arg1
#define VARS     MPI_Comm arg0; MPI_Group* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_rank ------------------------
#define NAME     Comm_rank
#define TEXTNAME "MPI_Comm_rank"
#define CALLSIG  MPI_Comm arg0, int* arg1
#define VARS     MPI_Comm arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_remote_group ------------------------
#define NAME     Comm_remote_group
#define TEXTNAME "MPI_Comm_remote_group"
#define CALLSIG  MPI_Comm arg0, MPI_Group* arg1
#define VARS     MPI_Comm arg0; MPI_Group* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_remote_size ------------------------
#define NAME     Comm_remote_size
#define TEXTNAME "MPI_Comm_remote_size"
#define CALLSIG  MPI_Comm arg0, int* arg1
#define VARS     MPI_Comm arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_set_name ------------------------
#define NAME     Comm_set_name
#define TEXTNAME "MPI_Comm_set_name"
#define CALLSIG  MPI_Comm arg0, char* arg1
#define VARS     MPI_Comm arg0; char* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_size ------------------------
#define NAME     Comm_size
#define TEXTNAME "MPI_Comm_size"
#define CALLSIG  MPI_Comm arg0, int* arg1
#define VARS     MPI_Comm arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_split ------------------------
#define NAME     Comm_split
#define TEXTNAME "MPI_Comm_split"
#define CALLSIG  MPI_Comm arg0, int arg1, int arg2, MPI_Comm* arg3
#define VARS     MPI_Comm arg0; int arg1; int arg2; MPI_Comm* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_test_inter ------------------------
#define NAME     Comm_test_inter
#define TEXTNAME "MPI_Comm_test_inter"
#define CALLSIG  MPI_Comm arg0, int* arg1
#define VARS     MPI_Comm arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Dims_create ------------------------
#define NAME     Dims_create
#define TEXTNAME "MPI_Dims_create"
#define CALLSIG  int arg0, int arg1, int* arg2
#define VARS     int arg0; int arg1; int* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Errhandler_create ------------------------
#define NAME     Errhandler_create
#define TEXTNAME "MPI_Errhandler_create"
#define CALLSIG  MPI_Handler_function* arg0, MPI_Errhandler* arg1
#define VARS     MPI_Handler_function* arg0; MPI_Errhandler* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Errhandler_free ------------------------
#define NAME     Errhandler_free
#define TEXTNAME "MPI_Errhandler_free"
#define CALLSIG  MPI_Errhandler* arg0
#define VARS     MPI_Errhandler* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Errhandler_get ------------------------
#define NAME     Errhandler_get
#define TEXTNAME "MPI_Errhandler_get"
#define CALLSIG  MPI_Comm arg0, MPI_Errhandler* arg1
#define VARS     MPI_Comm arg0; MPI_Errhandler* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Errhandler_set ------------------------
#define NAME     Errhandler_set
#define TEXTNAME "MPI_Errhandler_set"
#define CALLSIG  MPI_Comm arg0, MPI_Errhandler arg1
#define VARS     MPI_Comm arg0; MPI_Errhandler arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Error_class ------------------------
#define NAME     Error_class
#define TEXTNAME "MPI_Error_class"
#define CALLSIG  int arg0, int* arg1
#define VARS     int arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Error_string ------------------------
#define NAME     Error_string
#define TEXTNAME "MPI_Error_string"
#define CALLSIG  int arg0, char* arg1, int* arg2
#define VARS     int arg0; char* arg1; int* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Finalized ------------------------
#define NAME     Finalized
#define TEXTNAME "MPI_Finalized"
#define CALLSIG  int* arg0
#define VARS     int* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Free_mem ------------------------
#define NAME     Free_mem
#define TEXTNAME "MPI_Free_mem"
#define CALLSIG  void* arg0
#define VARS     void* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Gather ------------------------
#define NAME     Gather
#define TEXTNAME "MPI_Gather"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, void* arg3, int arg4, MPI_Datatype arg5, int arg6, MPI_Comm arg7
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; void* arg3; int arg4; MPI_Datatype arg5; int arg6; MPI_Comm arg7; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), arg7(arg7), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Gatherv ------------------------
#define NAME     Gatherv
#define TEXTNAME "MPI_Gatherv"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, void* arg3, int* arg4, int* arg5, MPI_Datatype arg6, int arg7, MPI_Comm arg8
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; void* arg3; int* arg4; int* arg5; MPI_Datatype arg6; int arg7; MPI_Comm arg8; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), arg7(arg7), arg8(arg8), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Get_address ------------------------
#define NAME     Get_address
#define TEXTNAME "MPI_Get_address"
#define CALLSIG  void* arg0, MPI_Aint* arg1
#define VARS     void* arg0; MPI_Aint* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Get ------------------------
#define NAME     Get
#define TEXTNAME "MPI_Get"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, MPI_Aint arg4, int arg5, MPI_Datatype arg6, MPI_Win arg7
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; MPI_Aint arg4; int arg5; MPI_Datatype arg6; MPI_Win arg7; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), arg7(arg7), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Get_count ------------------------
#define NAME     Get_count
#define TEXTNAME "MPI_Get_count"
#define CALLSIG  MPI_Status* arg0, MPI_Datatype arg1, int* arg2
#define VARS     MPI_Status* arg0; MPI_Datatype arg1; int* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Get_elements ------------------------
#define NAME     Get_elements
#define TEXTNAME "MPI_Get_elements"
#define CALLSIG  MPI_Status* arg0, MPI_Datatype arg1, int* arg2
#define VARS     MPI_Status* arg0; MPI_Datatype arg1; int* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Get_processor_name ------------------------
#define NAME     Get_processor_name
#define TEXTNAME "MPI_Get_processor_name"
#define CALLSIG  char* arg0, int* arg1
#define VARS     char* arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Get_version ------------------------
#define NAME     Get_version
#define TEXTNAME "MPI_Get_version"
#define CALLSIG  int* arg0, int* arg1
#define VARS     int* arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Graph_create ------------------------
#define NAME     Graph_create
#define TEXTNAME "MPI_Graph_create"
#define CALLSIG  MPI_Comm arg0, int arg1, int* arg2, int* arg3, int arg4, MPI_Comm* arg5
#define VARS     MPI_Comm arg0; int arg1; int* arg2; int* arg3; int arg4; MPI_Comm* arg5; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Graphdims_get ------------------------
#define NAME     Graphdims_get
#define TEXTNAME "MPI_Graphdims_get"
#define CALLSIG  MPI_Comm arg0, int* arg1, int* arg2
#define VARS     MPI_Comm arg0; int* arg1; int* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Graph_get ------------------------
#define NAME     Graph_get
#define TEXTNAME "MPI_Graph_get"
#define CALLSIG  MPI_Comm arg0, int arg1, int arg2, int* arg3, int* arg4
#define VARS     MPI_Comm arg0; int arg1; int arg2; int* arg3; int* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Graph_map ------------------------
#define NAME     Graph_map
#define TEXTNAME "MPI_Graph_map"
#define CALLSIG  MPI_Comm arg0, int arg1, int* arg2, int* arg3, int* arg4
#define VARS     MPI_Comm arg0; int arg1; int* arg2; int* arg3; int* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Graph_neighbors ------------------------
#define NAME     Graph_neighbors
#define TEXTNAME "MPI_Graph_neighbors"
#define CALLSIG  MPI_Comm arg0, int arg1, int arg2, int* arg3
#define VARS     MPI_Comm arg0; int arg1; int arg2; int* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Graph_neighbors_count ------------------------
#define NAME     Graph_neighbors_count
#define TEXTNAME "MPI_Graph_neighbors_count"
#define CALLSIG  MPI_Comm arg0, int arg1, int* arg2
#define VARS     MPI_Comm arg0; int arg1; int* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Group_compare ------------------------
#define NAME     Group_compare
#define TEXTNAME "MPI_Group_compare"
#define CALLSIG  MPI_Group arg0, MPI_Group arg1, int* arg2
#define VARS     MPI_Group arg0; MPI_Group arg1; int* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Group_difference ------------------------
#define NAME     Group_difference
#define TEXTNAME "MPI_Group_difference"
#define CALLSIG  MPI_Group arg0, MPI_Group arg1, MPI_Group* arg2
#define VARS     MPI_Group arg0; MPI_Group arg1; MPI_Group* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Group_excl ------------------------
#define NAME     Group_excl
#define TEXTNAME "MPI_Group_excl"
#define CALLSIG  MPI_Group arg0, int arg1, int* arg2, MPI_Group* arg3
#define VARS     MPI_Group arg0; int arg1; int* arg2; MPI_Group* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Group_free ------------------------
#define NAME     Group_free
#define TEXTNAME "MPI_Group_free"
#define CALLSIG  MPI_Group* arg0
#define VARS     MPI_Group* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Group_incl ------------------------
#define NAME     Group_incl
#define TEXTNAME "MPI_Group_incl"
#define CALLSIG  MPI_Group arg0, int arg1, int* arg2, MPI_Group* arg3
#define VARS     MPI_Group arg0; int arg1; int* arg2; MPI_Group* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Group_intersection ------------------------
#define NAME     Group_intersection
#define TEXTNAME "MPI_Group_intersection"
#define CALLSIG  MPI_Group arg0, MPI_Group arg1, MPI_Group* arg2
#define VARS     MPI_Group arg0; MPI_Group arg1; MPI_Group* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Group_rank ------------------------
#define NAME     Group_rank
#define TEXTNAME "MPI_Group_rank"
#define CALLSIG  MPI_Group arg0, int* arg1
#define VARS     MPI_Group arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Group_size ------------------------
#define NAME     Group_size
#define TEXTNAME "MPI_Group_size"
#define CALLSIG  MPI_Group arg0, int* arg1
#define VARS     MPI_Group arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Group_translate_ranks ------------------------
#define NAME     Group_translate_ranks
#define TEXTNAME "MPI_Group_translate_ranks"
#define CALLSIG  MPI_Group arg0, int arg1, int* arg2, MPI_Group arg3, int* arg4
#define VARS     MPI_Group arg0; int arg1; int* arg2; MPI_Group arg3; int* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Group_union ------------------------
#define NAME     Group_union
#define TEXTNAME "MPI_Group_union"
#define CALLSIG  MPI_Group arg0, MPI_Group arg1, MPI_Group* arg2
#define VARS     MPI_Group arg0; MPI_Group arg1; MPI_Group* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Ibsend ------------------------
#define NAME     Ibsend
#define TEXTNAME "MPI_Ibsend"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5, MPI_Request* arg6
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; MPI_Request* arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Info_create ------------------------
#define NAME     Info_create
#define TEXTNAME "MPI_Info_create"
#define CALLSIG  MPI_Info* arg0
#define VARS     MPI_Info* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Info_delete ------------------------
#define NAME     Info_delete
#define TEXTNAME "MPI_Info_delete"
#define CALLSIG  MPI_Info arg0, char* arg1
#define VARS     MPI_Info arg0; char* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Info_dup ------------------------
#define NAME     Info_dup
#define TEXTNAME "MPI_Info_dup"
#define CALLSIG  MPI_Info arg0, MPI_Info* arg1
#define VARS     MPI_Info arg0; MPI_Info* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Info_free ------------------------
#define NAME     Info_free
#define TEXTNAME "MPI_Info_free"
#define CALLSIG  MPI_Info* arg0
#define VARS     MPI_Info* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Info_get ------------------------
#define NAME     Info_get
#define TEXTNAME "MPI_Info_get"
#define CALLSIG  MPI_Info arg0, char* arg1, int arg2, char* arg3, int* arg4
#define VARS     MPI_Info arg0; char* arg1; int arg2; char* arg3; int* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Info_get_nkeys ------------------------
#define NAME     Info_get_nkeys
#define TEXTNAME "MPI_Info_get_nkeys"
#define CALLSIG  MPI_Info arg0, int* arg1
#define VARS     MPI_Info arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Info_get_nthkey ------------------------
#define NAME     Info_get_nthkey
#define TEXTNAME "MPI_Info_get_nthkey"
#define CALLSIG  MPI_Info arg0, int arg1, char* arg2
#define VARS     MPI_Info arg0; int arg1; char* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Info_get_valuelen ------------------------
#define NAME     Info_get_valuelen
#define TEXTNAME "MPI_Info_get_valuelen"
#define CALLSIG  MPI_Info arg0, char* arg1, int* arg2, int* arg3
#define VARS     MPI_Info arg0; char* arg1; int* arg2; int* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Info_set ------------------------
#define NAME     Info_set
#define TEXTNAME "MPI_Info_set"
#define CALLSIG  MPI_Info arg0, char* arg1, char* arg2
#define VARS     MPI_Info arg0; char* arg1; char* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Init ------------------------
#define NAME     Init
#define TEXTNAME "MPI_Init"
#define CALLSIG  int* arg0, char*** arg1
#define VARS     int* arg0; char*** arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Initialized ------------------------
#define NAME     Initialized
#define TEXTNAME "MPI_Initialized"
#define CALLSIG  int* arg0
#define VARS     int* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Intercomm_create ------------------------
#define NAME     Intercomm_create
#define TEXTNAME "MPI_Intercomm_create"
#define CALLSIG  MPI_Comm arg0, int arg1, MPI_Comm arg2, int arg3, int arg4, MPI_Comm* arg5
#define VARS     MPI_Comm arg0; int arg1; MPI_Comm arg2; int arg3; int arg4; MPI_Comm* arg5; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Intercomm_merge ------------------------
#define NAME     Intercomm_merge
#define TEXTNAME "MPI_Intercomm_merge"
#define CALLSIG  MPI_Comm arg0, int arg1, MPI_Comm* arg2
#define VARS     MPI_Comm arg0; int arg1; MPI_Comm* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Iprobe ------------------------
#define NAME     Iprobe
#define TEXTNAME "MPI_Iprobe"
#define CALLSIG  int arg0, int arg1, MPI_Comm arg2, int* arg3, MPI_Status* arg4
#define VARS     int arg0; int arg1; MPI_Comm arg2; int* arg3; MPI_Status* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Irecv ------------------------
#define NAME     Irecv
#define TEXTNAME "MPI_Irecv"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5, MPI_Request* arg6
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; MPI_Request* arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Irsend ------------------------
#define NAME     Irsend
#define TEXTNAME "MPI_Irsend"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5, MPI_Request* arg6
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; MPI_Request* arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Isend ------------------------
#define NAME     Isend
#define TEXTNAME "MPI_Isend"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5, MPI_Request* arg6
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; MPI_Request* arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Issend ------------------------
#define NAME     Issend
#define TEXTNAME "MPI_Issend"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5, MPI_Request* arg6
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; MPI_Request* arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Keyval_create ------------------------
#define NAME     Keyval_create
#define TEXTNAME "MPI_Keyval_create"
#define CALLSIG  MPI_Copy_function* arg0, MPI_Delete_function* arg1, int* arg2, void* arg3
#define VARS     MPI_Copy_function* arg0; MPI_Delete_function* arg1; int* arg2; void* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Keyval_free ------------------------
#define NAME     Keyval_free
#define TEXTNAME "MPI_Keyval_free"
#define CALLSIG  int* arg0
#define VARS     int* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Op_create ------------------------
#define NAME     Op_create
#define TEXTNAME "MPI_Op_create"
#define CALLSIG  MPI_User_function* arg0, int arg1, MPI_Op* arg2
#define VARS     MPI_User_function* arg0; int arg1; MPI_Op* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Op_free ------------------------
#define NAME     Op_free
#define TEXTNAME "MPI_Op_free"
#define CALLSIG  MPI_Op* arg0
#define VARS     MPI_Op* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Pack ------------------------
#define NAME     Pack
#define TEXTNAME "MPI_Pack"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, void* arg3, int arg4, int* arg5, MPI_Comm arg6
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; void* arg3; int arg4; int* arg5; MPI_Comm arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Pack_size ------------------------
#define NAME     Pack_size
#define TEXTNAME "MPI_Pack_size"
#define CALLSIG  int arg0, MPI_Datatype arg1, MPI_Comm arg2, int* arg3
#define VARS     int arg0; MPI_Datatype arg1; MPI_Comm arg2; int* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Probe ------------------------
#define NAME     Probe
#define TEXTNAME "MPI_Probe"
#define CALLSIG  int arg0, int arg1, MPI_Comm arg2, MPI_Status* arg3
#define VARS     int arg0; int arg1; MPI_Comm arg2; MPI_Status* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Put ------------------------
#define NAME     Put
#define TEXTNAME "MPI_Put"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, MPI_Aint arg4, int arg5, MPI_Datatype arg6, MPI_Win arg7
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; MPI_Aint arg4; int arg5; MPI_Datatype arg6; MPI_Win arg7; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), arg7(arg7), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Recv ------------------------
#define NAME     Recv
#define TEXTNAME "MPI_Recv"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5, MPI_Status* arg6
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; MPI_Status* arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Recv_init ------------------------
#define NAME     Recv_init
#define TEXTNAME "MPI_Recv_init"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5, MPI_Request* arg6
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; MPI_Request* arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Reduce ------------------------
#define NAME     Reduce
#define TEXTNAME "MPI_Reduce"
#define CALLSIG  void* arg0, void* arg1, int arg2, MPI_Datatype arg3, MPI_Op arg4, int arg5, MPI_Comm arg6
#define VARS     void* arg0; void* arg1; int arg2; MPI_Datatype arg3; MPI_Op arg4; int arg5; MPI_Comm arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Reduce_scatter ------------------------
#define NAME     Reduce_scatter
#define TEXTNAME "MPI_Reduce_scatter"
#define CALLSIG  void* arg0, void* arg1, int* arg2, MPI_Datatype arg3, MPI_Op arg4, MPI_Comm arg5
#define VARS     void* arg0; void* arg1; int* arg2; MPI_Datatype arg3; MPI_Op arg4; MPI_Comm arg5; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Request_free ------------------------
#define NAME     Request_free
#define TEXTNAME "MPI_Request_free"
#define CALLSIG  MPI_Request* arg0
#define VARS     MPI_Request* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Rsend ------------------------
#define NAME     Rsend
#define TEXTNAME "MPI_Rsend"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Rsend_init ------------------------
#define NAME     Rsend_init
#define TEXTNAME "MPI_Rsend_init"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5, MPI_Request* arg6
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; MPI_Request* arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Scan ------------------------
#define NAME     Scan
#define TEXTNAME "MPI_Scan"
#define CALLSIG  void* arg0, void* arg1, int arg2, MPI_Datatype arg3, MPI_Op arg4, MPI_Comm arg5
#define VARS     void* arg0; void* arg1; int arg2; MPI_Datatype arg3; MPI_Op arg4; MPI_Comm arg5; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Scatter ------------------------
#define NAME     Scatter
#define TEXTNAME "MPI_Scatter"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, void* arg3, int arg4, MPI_Datatype arg5, int arg6, MPI_Comm arg7
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; void* arg3; int arg4; MPI_Datatype arg5; int arg6; MPI_Comm arg7; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), arg7(arg7), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Scatterv ------------------------
#define NAME     Scatterv
#define TEXTNAME "MPI_Scatterv"
#define CALLSIG  void* arg0, int* arg1, int* arg2, MPI_Datatype arg3, void* arg4, int arg5, MPI_Datatype arg6, int arg7, MPI_Comm arg8
#define VARS     void* arg0; int* arg1; int* arg2; MPI_Datatype arg3; void* arg4; int arg5; MPI_Datatype arg6; int arg7; MPI_Comm arg8; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), arg7(arg7), arg8(arg8), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Send ------------------------
#define NAME     Send
#define TEXTNAME "MPI_Send"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Send_init ------------------------
#define NAME     Send_init
#define TEXTNAME "MPI_Send_init"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5, MPI_Request* arg6
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; MPI_Request* arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Sendrecv ------------------------
#define NAME     Sendrecv
#define TEXTNAME "MPI_Sendrecv"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, void* arg5, int arg6, MPI_Datatype arg7, int arg8, int arg9, MPI_Comm arg10, MPI_Status* arg11
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; void* arg5; int arg6; MPI_Datatype arg7; int arg8; int arg9; MPI_Comm arg10; MPI_Status* arg11; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), arg7(arg7), arg8(arg8), arg9(arg9), arg10(arg10), arg11(arg11), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Sendrecv_replace ------------------------
#define NAME     Sendrecv_replace
#define TEXTNAME "MPI_Sendrecv_replace"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, int arg5, int arg6, MPI_Comm arg7, MPI_Status* arg8
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; int arg5; int arg6; MPI_Comm arg7; MPI_Status* arg8; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), arg7(arg7), arg8(arg8), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Ssend ------------------------
#define NAME     Ssend
#define TEXTNAME "MPI_Ssend"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Ssend_init ------------------------
#define NAME     Ssend_init
#define TEXTNAME "MPI_Ssend_init"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, int arg4, MPI_Comm arg5, MPI_Request* arg6
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; int arg4; MPI_Comm arg5; MPI_Request* arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Startall ------------------------
#define NAME     Startall
#define TEXTNAME "MPI_Startall"
#define CALLSIG  int arg0, MPI_Request* arg1
#define VARS     int arg0; MPI_Request* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Start ------------------------
#define NAME     Start
#define TEXTNAME "MPI_Start"
#define CALLSIG  MPI_Request* arg0
#define VARS     MPI_Request* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Testall ------------------------
#define NAME     Testall
#define TEXTNAME "MPI_Testall"
#define CALLSIG  int arg0, MPI_Request* arg1, int* arg2, MPI_Status* arg3
#define VARS     int arg0; MPI_Request* arg1; int* arg2; MPI_Status* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Testany ------------------------
#define NAME     Testany
#define TEXTNAME "MPI_Testany"
#define CALLSIG  int arg0, MPI_Request* arg1, int* arg2, int* arg3, MPI_Status* arg4
#define VARS     int arg0; MPI_Request* arg1; int* arg2; int* arg3; MPI_Status* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Test ------------------------
#define NAME     Test
#define TEXTNAME "MPI_Test"
#define CALLSIG  MPI_Request* arg0, int* arg1, MPI_Status* arg2
#define VARS     MPI_Request* arg0; int* arg1; MPI_Status* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Test_cancelled ------------------------
#define NAME     Test_cancelled
#define TEXTNAME "MPI_Test_cancelled"
#define CALLSIG  MPI_Status* arg0, int* arg1
#define VARS     MPI_Status* arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Testsome ------------------------
#define NAME     Testsome
#define TEXTNAME "MPI_Testsome"
#define CALLSIG  int arg0, MPI_Request* arg1, int* arg2, int* arg3, MPI_Status* arg4
#define VARS     int arg0; MPI_Request* arg1; int* arg2; int* arg3; MPI_Status* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Topo_test ------------------------
#define NAME     Topo_test
#define TEXTNAME "MPI_Topo_test"
#define CALLSIG  MPI_Comm arg0, int* arg1
#define VARS     MPI_Comm arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_commit ------------------------
#define NAME     Type_commit
#define TEXTNAME "MPI_Type_commit"
#define CALLSIG  MPI_Datatype* arg0
#define VARS     MPI_Datatype* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_contiguous ------------------------
#define NAME     Type_contiguous
#define TEXTNAME "MPI_Type_contiguous"
#define CALLSIG  int arg0, MPI_Datatype arg1, MPI_Datatype* arg2
#define VARS     int arg0; MPI_Datatype arg1; MPI_Datatype* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_create_darray ------------------------
#define NAME     Type_create_darray
#define TEXTNAME "MPI_Type_create_darray"
#define CALLSIG  int arg0, int arg1, int arg2, int* arg3, int* arg4, int* arg5, int* arg6, int arg7, MPI_Datatype arg8, MPI_Datatype* arg9
#define VARS     int arg0; int arg1; int arg2; int* arg3; int* arg4; int* arg5; int* arg6; int arg7; MPI_Datatype arg8; MPI_Datatype* arg9; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), arg7(arg7), arg8(arg8), arg9(arg9), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_create_subarray ------------------------
#define NAME     Type_create_subarray
#define TEXTNAME "MPI_Type_create_subarray"
#define CALLSIG  int arg0, int* arg1, int* arg2, int* arg3, int arg4, MPI_Datatype arg5, MPI_Datatype* arg6
#define VARS     int arg0; int* arg1; int* arg2; int* arg3; int arg4; MPI_Datatype arg5; MPI_Datatype* arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_extent ------------------------
#define NAME     Type_extent
#define TEXTNAME "MPI_Type_extent"
#define CALLSIG  MPI_Datatype arg0, MPI_Aint* arg1
#define VARS     MPI_Datatype arg0; MPI_Aint* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_free ------------------------
#define NAME     Type_free
#define TEXTNAME "MPI_Type_free"
#define CALLSIG  MPI_Datatype* arg0
#define VARS     MPI_Datatype* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_get_contents ------------------------
#define NAME     Type_get_contents
#define TEXTNAME "MPI_Type_get_contents"
#define CALLSIG  MPI_Datatype arg0, int arg1, int arg2, int arg3, int* arg4, MPI_Aint* arg5, MPI_Datatype* arg6
#define VARS     MPI_Datatype arg0; int arg1; int arg2; int arg3; int* arg4; MPI_Aint* arg5; MPI_Datatype* arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_get_envelope ------------------------
#define NAME     Type_get_envelope
#define TEXTNAME "MPI_Type_get_envelope"
#define CALLSIG  MPI_Datatype arg0, int* arg1, int* arg2, int* arg3, int* arg4
#define VARS     MPI_Datatype arg0; int* arg1; int* arg2; int* arg3; int* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_hindexed ------------------------
#define NAME     Type_hindexed
#define TEXTNAME "MPI_Type_hindexed"
#define CALLSIG  int arg0, int* arg1, MPI_Aint* arg2, MPI_Datatype arg3, MPI_Datatype* arg4
#define VARS     int arg0; int* arg1; MPI_Aint* arg2; MPI_Datatype arg3; MPI_Datatype* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_hvector ------------------------
#define NAME     Type_hvector
#define TEXTNAME "MPI_Type_hvector"
#define CALLSIG  int arg0, int arg1, MPI_Aint arg2, MPI_Datatype arg3, MPI_Datatype* arg4
#define VARS     int arg0; int arg1; MPI_Aint arg2; MPI_Datatype arg3; MPI_Datatype* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_indexed ------------------------
#define NAME     Type_indexed
#define TEXTNAME "MPI_Type_indexed"
#define CALLSIG  int arg0, int* arg1, int* arg2, MPI_Datatype arg3, MPI_Datatype* arg4
#define VARS     int arg0; int* arg1; int* arg2; MPI_Datatype arg3; MPI_Datatype* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_lb ------------------------
#define NAME     Type_lb
#define TEXTNAME "MPI_Type_lb"
#define CALLSIG  MPI_Datatype arg0, MPI_Aint* arg1
#define VARS     MPI_Datatype arg0; MPI_Aint* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_size ------------------------
#define NAME     Type_size
#define TEXTNAME "MPI_Type_size"
#define CALLSIG  MPI_Datatype arg0, int* arg1
#define VARS     MPI_Datatype arg0; int* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_struct ------------------------
#define NAME     Type_struct
#define TEXTNAME "MPI_Type_struct"
#define CALLSIG  int arg0, int* arg1, MPI_Aint* arg2, MPI_Datatype* arg3, MPI_Datatype* arg4
#define VARS     int arg0; int* arg1; MPI_Aint* arg2; MPI_Datatype* arg3; MPI_Datatype* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_ub ------------------------
#define NAME     Type_ub
#define TEXTNAME "MPI_Type_ub"
#define CALLSIG  MPI_Datatype arg0, MPI_Aint* arg1
#define VARS     MPI_Datatype arg0; MPI_Aint* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_vector ------------------------
#define NAME     Type_vector
#define TEXTNAME "MPI_Type_vector"
#define CALLSIG  int arg0, int arg1, int arg2, MPI_Datatype arg3, MPI_Datatype* arg4
#define VARS     int arg0; int arg1; int arg2; MPI_Datatype arg3; MPI_Datatype* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Unpack ------------------------
#define NAME     Unpack
#define TEXTNAME "MPI_Unpack"
#define CALLSIG  void* arg0, int arg1, int* arg2, void* arg3, int arg4, MPI_Datatype arg5, MPI_Comm arg6
#define VARS     void* arg0; int arg1; int* arg2; void* arg3; int arg4; MPI_Datatype arg5; MPI_Comm arg6; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Waitall ------------------------
#define NAME     Waitall
#define TEXTNAME "MPI_Waitall"
#define CALLSIG  int arg0, MPI_Request* arg1, MPI_Status* arg2
#define VARS     int arg0; MPI_Request* arg1; MPI_Status* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Waitany ------------------------
#define NAME     Waitany
#define TEXTNAME "MPI_Waitany"
#define CALLSIG  int arg0, MPI_Request* arg1, int* arg2, MPI_Status* arg3
#define VARS     int arg0; MPI_Request* arg1; int* arg2; MPI_Status* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Wait ------------------------
#define NAME     Wait
#define TEXTNAME "MPI_Wait"
#define CALLSIG  MPI_Request* arg0, MPI_Status* arg1
#define VARS     MPI_Request* arg0; MPI_Status* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Waitsome ------------------------
#define NAME     Waitsome
#define TEXTNAME "MPI_Waitsome"
#define CALLSIG  int arg0, MPI_Request* arg1, int* arg2, int* arg3, MPI_Status* arg4
#define VARS     int arg0; MPI_Request* arg1; int* arg2; int* arg3; MPI_Status* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_complete ------------------------
#define NAME     Win_complete
#define TEXTNAME "MPI_Win_complete"
#define CALLSIG  MPI_Win arg0
#define VARS     MPI_Win arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_create ------------------------
#define NAME     Win_create
#define TEXTNAME "MPI_Win_create"
#define CALLSIG  void* arg0, MPI_Aint arg1, int arg2, MPI_Info arg3, MPI_Comm arg4, MPI_Win* arg5
#define VARS     void* arg0; MPI_Aint arg1; int arg2; MPI_Info arg3; MPI_Comm arg4; MPI_Win* arg5; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_fence ------------------------
#define NAME     Win_fence
#define TEXTNAME "MPI_Win_fence"
#define CALLSIG  int arg0, MPI_Win arg1
#define VARS     int arg0; MPI_Win arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_free ------------------------
#define NAME     Win_free
#define TEXTNAME "MPI_Win_free"
#define CALLSIG  MPI_Win* arg0
#define VARS     MPI_Win* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_get_group ------------------------
#define NAME     Win_get_group
#define TEXTNAME "MPI_Win_get_group"
#define CALLSIG  MPI_Win arg0, MPI_Group* arg1
#define VARS     MPI_Win arg0; MPI_Group* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_post ------------------------
#define NAME     Win_post
#define TEXTNAME "MPI_Win_post"
#define CALLSIG  MPI_Group arg0, int arg1, MPI_Win arg2
#define VARS     MPI_Group arg0; int arg1; MPI_Win arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_start ------------------------
#define NAME     Win_start
#define TEXTNAME "MPI_Win_start"
#define CALLSIG  MPI_Group arg0, int arg1, MPI_Win arg2
#define VARS     MPI_Group arg0; int arg1; MPI_Win arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_wait ------------------------
#define NAME     Win_wait
#define TEXTNAME "MPI_Win_wait"
#define CALLSIG  MPI_Win arg0
#define VARS     MPI_Win arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Status_f2c ------------------------
#define NAME     Status_f2c
#define TEXTNAME "MPI_Status_f2c"
#define CALLSIG  MPI_Fint* arg0, MPI_Status* arg1
#define VARS     MPI_Fint* arg0; MPI_Status* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Info_f2c ------------------------
#define NAME     Info_f2c
#define TEXTNAME "MPI_Info_f2c"
#define CALLSIG  MPI_Fint arg0
#define VARS     MPI_Fint arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Info
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Status_c2f ------------------------
#define NAME     Status_c2f
#define TEXTNAME "MPI_Status_c2f"
#define CALLSIG  MPI_Status* arg0, MPI_Fint* arg1
#define VARS     MPI_Status* arg0; MPI_Fint* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Info_c2f ------------------------
#define NAME     Info_c2f
#define TEXTNAME "MPI_Info_c2f"
#define CALLSIG  MPI_Info arg0
#define VARS     MPI_Info arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Fint
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

#ifdef LAM_MPI // LAM_MPI the following do not have a PMPI_ version on alc

//-------------------------- MPI_Close_port ------------------------
#define NAME     Close_port
#define TEXTNAME "MPI_Close_port"
#define CALLSIG  char* arg0
#define VARS     char* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_accept ------------------------
#define NAME     Comm_accept
#define TEXTNAME "MPI_Comm_accept"
#define CALLSIG  char* arg0, MPI_Info arg1, int arg2, MPI_Comm arg3, MPI_Comm* arg4
#define VARS     char* arg0; MPI_Info arg1; int arg2; MPI_Comm arg3; MPI_Comm* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_connect ------------------------
#define NAME     Comm_connect
#define TEXTNAME "MPI_Comm_connect"
#define CALLSIG  char* arg0, MPI_Info arg1, int arg2, MPI_Comm arg3, MPI_Comm* arg4
#define VARS     char* arg0; MPI_Info arg1; int arg2; MPI_Comm arg3; MPI_Comm* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_create_errhandler ------------------------
#define NAME     Comm_create_errhandler
#define TEXTNAME "MPI_Comm_create_errhandler"
#define CALLSIG  MPI_Comm_errhandler_fn* arg0, MPI_Errhandler* arg1
#define VARS     MPI_Comm_errhandler_fn* arg0; MPI_Errhandler* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_create_keyval ------------------------
#define NAME     Comm_create_keyval
#define TEXTNAME "MPI_Comm_create_keyval"
#define CALLSIG  MPI_Comm_copy_attr_function* arg0, MPI_Comm_delete_attr_function* arg1, int* arg2, void* arg3
#define VARS     MPI_Comm_copy_attr_function* arg0; MPI_Comm_delete_attr_function* arg1; int* arg2; void* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_delete_attr ------------------------
#define NAME     Comm_delete_attr
#define TEXTNAME "MPI_Comm_delete_attr"
#define CALLSIG  MPI_Comm arg0, int arg1
#define VARS     MPI_Comm arg0; int arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_disconnect ------------------------
#define NAME     Comm_disconnect
#define TEXTNAME "MPI_Comm_disconnect"
#define CALLSIG  MPI_Comm* arg0
#define VARS     MPI_Comm* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_free_keyval ------------------------
#define NAME     Comm_free_keyval
#define TEXTNAME "MPI_Comm_free_keyval"
#define CALLSIG  int* arg0
#define VARS     int* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_get_errhandler ------------------------
#define NAME     Comm_get_errhandler
#define TEXTNAME "MPI_Comm_get_errhandler"
#define CALLSIG  MPI_Comm arg0, MPI_Errhandler* arg1
#define VARS     MPI_Comm arg0; MPI_Errhandler* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_get_parent ------------------------
#define NAME     Comm_get_parent
#define TEXTNAME "MPI_Comm_get_parent"
#define CALLSIG  MPI_Comm* arg0
#define VARS     MPI_Comm* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_join ------------------------
#define NAME     Comm_join
#define TEXTNAME "MPI_Comm_join"
#define CALLSIG  int arg0, MPI_Comm* arg1
#define VARS     int arg0; MPI_Comm* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_set_attr ------------------------
#define NAME     Comm_set_attr
#define TEXTNAME "MPI_Comm_set_attr"
#define CALLSIG  MPI_Comm arg0, int arg1, void* arg2
#define VARS     MPI_Comm arg0; int arg1; void* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_set_errhandler ------------------------
#define NAME     Comm_set_errhandler
#define TEXTNAME "MPI_Comm_set_errhandler"
#define CALLSIG  MPI_Comm arg0, MPI_Errhandler arg1
#define VARS     MPI_Comm arg0; MPI_Errhandler arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_spawn ------------------------
#define NAME     Comm_spawn
#define TEXTNAME "MPI_Comm_spawn"
#define CALLSIG  char* arg0, char** arg1, int arg2, MPI_Info arg3, int arg4, MPI_Comm arg5, MPI_Comm* arg6, int* arg7
#define VARS     char* arg0; char** arg1; int arg2; MPI_Info arg3; int arg4; MPI_Comm arg5; MPI_Comm* arg6; int* arg7; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), arg7(arg7), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_spawn_multiple ------------------------
#define NAME     Comm_spawn_multiple
#define TEXTNAME "MPI_Comm_spawn_multiple"
#define CALLSIG  int arg0, char** arg1, char*** arg2, int* arg3, MPI_Info* arg4, int arg5, MPI_Comm arg6, MPI_Comm* arg7, int* arg8
#define VARS     int arg0; char** arg1; char*** arg2; int* arg3; MPI_Info* arg4; int arg5; MPI_Comm arg6; MPI_Comm* arg7; int* arg8; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5), arg6(arg6), arg7(arg7), arg8(arg8), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Is_thread_main ------------------------
#define NAME     Is_thread_main
#define TEXTNAME "MPI_Is_thread_main"
#define CALLSIG  int* arg0
#define VARS     int* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Lookup_name ------------------------
#define NAME     Lookup_name
#define TEXTNAME "MPI_Lookup_name"
#define CALLSIG  char* arg0, MPI_Info arg1, char* arg2
#define VARS     char* arg0; MPI_Info arg1; char* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Open_port ------------------------
#define NAME     Open_port
#define TEXTNAME "MPI_Open_port"
#define CALLSIG  MPI_Info arg0, char* arg1
#define VARS     MPI_Info arg0; char* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Publish_name ------------------------
#define NAME     Publish_name
#define TEXTNAME "MPI_Publish_name"
#define CALLSIG  char* arg0, MPI_Info arg1, char* arg2
#define VARS     char* arg0; MPI_Info arg1; char* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Query_thread ------------------------
#define NAME     Query_thread
#define TEXTNAME "MPI_Query_thread"
#define CALLSIG  int* arg0
#define VARS     int* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_create_hindexed ------------------------
#define NAME     Type_create_hindexed
#define TEXTNAME "MPI_Type_create_hindexed"
#define CALLSIG  int arg0, int* arg1, MPI_Aint* arg2, MPI_Datatype arg3, MPI_Datatype* arg4
#define VARS     int arg0; int* arg1; MPI_Aint* arg2; MPI_Datatype arg3; MPI_Datatype* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_create_hvector ------------------------
#define NAME     Type_create_hvector
#define TEXTNAME "MPI_Type_create_hvector"
#define CALLSIG  int arg0, int arg1, MPI_Aint arg2, MPI_Datatype arg3, MPI_Datatype* arg4
#define VARS     int arg0; int arg1; MPI_Aint arg2; MPI_Datatype arg3; MPI_Datatype* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_create_keyval ------------------------
#define NAME     Type_create_keyval
#define TEXTNAME "MPI_Type_create_keyval"
#define CALLSIG  MPI_Type_copy_attr_function* arg0, MPI_Type_delete_attr_function* arg1, int* arg2, void* arg3
#define VARS     MPI_Type_copy_attr_function* arg0; MPI_Type_delete_attr_function* arg1; int* arg2; void* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_create_resized ------------------------
#define NAME     Type_create_resized
#define TEXTNAME "MPI_Type_create_resized"
#define CALLSIG  MPI_Datatype arg0, MPI_Aint arg1, MPI_Aint arg2, MPI_Datatype* arg3
#define VARS     MPI_Datatype arg0; MPI_Aint arg1; MPI_Aint arg2; MPI_Datatype* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_create_struct ------------------------
#define NAME     Type_create_struct
#define TEXTNAME "MPI_Type_create_struct"
#define CALLSIG  int arg0, int* arg1, MPI_Aint* arg2, MPI_Datatype* arg3, MPI_Datatype* arg4
#define VARS     int arg0; int* arg1; MPI_Aint* arg2; MPI_Datatype* arg3; MPI_Datatype* arg4; 
#define CALLARGS arg0, arg1, arg2, arg3, arg4
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_delete_attr ------------------------
#define NAME     Type_delete_attr
#define TEXTNAME "MPI_Type_delete_attr"
#define CALLSIG  MPI_Datatype arg0, int arg1
#define VARS     MPI_Datatype arg0; int arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_dup ------------------------
#define NAME     Type_dup
#define TEXTNAME "MPI_Type_dup"
#define CALLSIG  MPI_Datatype arg0, MPI_Datatype* arg1
#define VARS     MPI_Datatype arg0; MPI_Datatype* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_free_keyval ------------------------
#define NAME     Type_free_keyval
#define TEXTNAME "MPI_Type_free_keyval"
#define CALLSIG  int* arg0
#define VARS     int* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_get_attr ------------------------
#define NAME     Type_get_attr
#define TEXTNAME "MPI_Type_get_attr"
#define CALLSIG  MPI_Datatype arg0, int arg1, void* arg2, int* arg3
#define VARS     MPI_Datatype arg0; int arg1; void* arg2; int* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_get_extent ------------------------
#define NAME     Type_get_extent
#define TEXTNAME "MPI_Type_get_extent"
#define CALLSIG  MPI_Datatype arg0, MPI_Aint* arg1, MPI_Aint* arg2
#define VARS     MPI_Datatype arg0; MPI_Aint* arg1; MPI_Aint* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_get_name ------------------------
#define NAME     Type_get_name
#define TEXTNAME "MPI_Type_get_name"
#define CALLSIG  MPI_Datatype arg0, char* arg1, int* arg2
#define VARS     MPI_Datatype arg0; char* arg1; int* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_get_true_extent ------------------------
#define NAME     Type_get_true_extent
#define TEXTNAME "MPI_Type_get_true_extent"
#define CALLSIG  MPI_Datatype arg0, MPI_Aint* arg1, MPI_Aint* arg2
#define VARS     MPI_Datatype arg0; MPI_Aint* arg1; MPI_Aint* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_set_attr ------------------------
#define NAME     Type_set_attr
#define TEXTNAME "MPI_Type_set_attr"
#define CALLSIG  MPI_Datatype arg0, int arg1, void* arg2
#define VARS     MPI_Datatype arg0; int arg1; void* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_set_name ------------------------
#define NAME     Type_set_name
#define TEXTNAME "MPI_Type_set_name"
#define CALLSIG  MPI_Datatype arg0, char* arg1
#define VARS     MPI_Datatype arg0; char* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Unpublish_name ------------------------
#define NAME     Unpublish_name
#define TEXTNAME "MPI_Unpublish_name"
#define CALLSIG  char* arg0, MPI_Info arg1, char* arg2
#define VARS     char* arg0; MPI_Info arg1; char* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_create_errhandler ------------------------
#define NAME     Win_create_errhandler
#define TEXTNAME "MPI_Win_create_errhandler"
#define CALLSIG  MPI_Win_errhandler_fn* arg0, MPI_Errhandler* arg1
#define VARS     MPI_Win_errhandler_fn* arg0; MPI_Errhandler* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_create_keyval ------------------------
#define NAME     Win_create_keyval
#define TEXTNAME "MPI_Win_create_keyval"
#define CALLSIG  MPI_Win_copy_attr_function* arg0, MPI_Win_delete_attr_function* arg1, int* arg2, void* arg3
#define VARS     MPI_Win_copy_attr_function* arg0; MPI_Win_delete_attr_function* arg1; int* arg2; void* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_delete_attr ------------------------
#define NAME     Win_delete_attr
#define TEXTNAME "MPI_Win_delete_attr"
#define CALLSIG  MPI_Win arg0, int arg1
#define VARS     MPI_Win arg0; int arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_free_keyval ------------------------
#define NAME     Win_free_keyval
#define TEXTNAME "MPI_Win_free_keyval"
#define CALLSIG  int* arg0
#define VARS     int* arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_get_attr ------------------------
#define NAME     Win_get_attr
#define TEXTNAME "MPI_Win_get_attr"
#define CALLSIG  MPI_Win arg0, int arg1, void* arg2, int* arg3
#define VARS     MPI_Win arg0; int arg1; void* arg2; int* arg3; 
#define CALLARGS arg0, arg1, arg2, arg3
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), arg3(arg3), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_get_errhandler ------------------------
#define NAME     Win_get_errhandler
#define TEXTNAME "MPI_Win_get_errhandler"
#define CALLSIG  MPI_Win arg0, MPI_Errhandler* arg1
#define VARS     MPI_Win arg0; MPI_Errhandler* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_get_name ------------------------
#define NAME     Win_get_name
#define TEXTNAME "MPI_Win_get_name"
#define CALLSIG  MPI_Win arg0, char* arg1, int* arg2
#define VARS     MPI_Win arg0; char* arg1; int* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_set_attr ------------------------
#define NAME     Win_set_attr
#define TEXTNAME "MPI_Win_set_attr"
#define CALLSIG  MPI_Win arg0, int arg1, void* arg2
#define VARS     MPI_Win arg0; int arg1; void* arg2; 
#define CALLARGS arg0, arg1, arg2
#define VAR_INIT arg0(arg0), arg1(arg1), arg2(arg2), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_set_errhandler ------------------------
#define NAME     Win_set_errhandler
#define TEXTNAME "MPI_Win_set_errhandler"
#define CALLSIG  MPI_Win arg0, MPI_Errhandler arg1
#define VARS     MPI_Win arg0; MPI_Errhandler arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_set_name ------------------------
#define NAME     Win_set_name
#define TEXTNAME "MPI_Win_set_name"
#define CALLSIG  MPI_Win arg0, char* arg1
#define VARS     MPI_Win arg0; char* arg1; 
#define CALLARGS arg0, arg1
#define VAR_INIT arg0(arg0), arg1(arg1), retval(retval)
#define RET_TYPE int
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_c2f ------------------------
#define NAME     Comm_c2f
#define TEXTNAME "MPI_Comm_c2f"
#define CALLSIG  MPI_Comm arg0
#define VARS     MPI_Comm arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Fint
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Errhandler_c2f ------------------------
#define NAME     Errhandler_c2f
#define TEXTNAME "MPI_Errhandler_c2f"
#define CALLSIG  MPI_Errhandler arg0
#define VARS     MPI_Errhandler arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Fint
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Group_c2f ------------------------
#define NAME     Group_c2f
#define TEXTNAME "MPI_Group_c2f"
#define CALLSIG  MPI_Group arg0
#define VARS     MPI_Group arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Fint
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Op_c2f ------------------------
#define NAME     Op_c2f
#define TEXTNAME "MPI_Op_c2f"
#define CALLSIG  MPI_Op arg0
#define VARS     MPI_Op arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Fint
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Request_c2f ------------------------
#define NAME     Request_c2f
#define TEXTNAME "MPI_Request_c2f"
#define CALLSIG  MPI_Request arg0
#define VARS     MPI_Request arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Fint
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_f2c ------------------------
#define NAME     Win_f2c
#define TEXTNAME "MPI_Win_f2c"
#define CALLSIG  MPI_Fint arg0
#define VARS     MPI_Fint arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Win
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Comm_f2c ------------------------
#define NAME     Comm_f2c
#define TEXTNAME "MPI_Comm_f2c"
#define CALLSIG  MPI_Fint arg0
#define VARS     MPI_Fint arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Comm
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Errhandler_f2c ------------------------
#define NAME     Errhandler_f2c
#define TEXTNAME "MPI_Errhandler_f2c"
#define CALLSIG  MPI_Fint arg0
#define VARS     MPI_Fint arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Errhandler
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Group_f2c ------------------------
#define NAME     Group_f2c
#define TEXTNAME "MPI_Group_f2c"
#define CALLSIG  MPI_Fint arg0
#define VARS     MPI_Fint arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Group
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_c2f ------------------------
#define NAME     Type_c2f
#define TEXTNAME "MPI_Type_c2f"
#define CALLSIG  MPI_Datatype arg0
#define VARS     MPI_Datatype arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Fint
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Win_c2f ------------------------
#define NAME     Win_c2f
#define TEXTNAME "MPI_Win_c2f"
#define CALLSIG  MPI_Win arg0
#define VARS     MPI_Win arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Fint
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Type_f2c ------------------------
#define NAME     Type_f2c
#define TEXTNAME "MPI_Type_f2c"
#define CALLSIG  MPI_Fint arg0
#define VARS     MPI_Fint arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Datatype
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

//-------------------------- MPI_Request_f2c ------------------------
#define NAME     Request_f2c
#define TEXTNAME "MPI_Request_f2c"
#define CALLSIG  MPI_Fint arg0
#define VARS     MPI_Fint arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI_Request
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

#endif //ifdef LAM_MPI

