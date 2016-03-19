/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

//-------------------------- MPI::Abort ------------------------
#define NAME     Abort
#define TEXTNAME "MPI::Abort"
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

//-------------------------- MPI::Accumulate ------------------------
#define NAME     Accumulate
#define TEXTNAME "MPI::Accumulate"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, MPI_Aint arg4, int arg5, MPI_Datatype arg6, MPI_Op arg7, MPI::Win arg8
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; MPI_Aint arg4; int arg5; MPI_Datatype arg6; MPI_Op arg7; MPI::Win arg8; 
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

//-------------------------- MPI::Address ------------------------
#define NAME     Address
#define TEXTNAME "MPI::Address"
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

//-------------------------- MPI::Allgather ------------------------
#define NAME     Allgather
#define TEXTNAME "MPI::Allgather"
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

//-------------------------- MPI::Allgatherv ------------------------
#define NAME     Allgatherv
#define TEXTNAME "MPI::Allgatherv"
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

//-------------------------- MPI::Alloc_mem ------------------------
#define NAME     Alloc_mem
#define TEXTNAME "MPI::Alloc_mem"
#define CALLSIG  MPI_Aint arg0, MPI::Info arg1, void* arg2
#define VARS     MPI_Aint arg0; MPI::Info arg1; void* arg2; 
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

//-------------------------- MPI::Allreduce ------------------------
#define NAME     Allreduce
#define TEXTNAME "MPI::Allreduce"
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

//-------------------------- MPI::Alltoall ------------------------
#define NAME     Alltoall
#define TEXTNAME "MPI::Alltoall"
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

//-------------------------- MPI::Alltoallv ------------------------
#define NAME     Alltoallv
#define TEXTNAME "MPI::Alltoallv"
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

//-------------------------- MPI::Attr_delete ------------------------
#define NAME     Attr_delete
#define TEXTNAME "MPI::Attr_delete"
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

//-------------------------- MPI::Attr_get ------------------------
#define NAME     Attr_get
#define TEXTNAME "MPI::Attr_get"
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

//-------------------------- MPI::Attr_put ------------------------
#define NAME     Attr_put
#define TEXTNAME "MPI::Attr_put"
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

//-------------------------- MPI::Barrier ------------------------
#define NAME     Barrier
#define TEXTNAME "MPI::Barrier"
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

//-------------------------- MPI::Bcast ------------------------
#define NAME     Bcast
#define TEXTNAME "MPI::Bcast"
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

//-------------------------- MPI::Bsend ------------------------
#define NAME     Bsend
#define TEXTNAME "MPI::Bsend"
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

//-------------------------- MPI::Bsend_init ------------------------
#define NAME     Bsend_init
#define TEXTNAME "MPI::Bsend_init"
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

//-------------------------- MPI::Buffer_attach ------------------------
#define NAME     Buffer_attach
#define TEXTNAME "MPI::Buffer_attach"
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

//-------------------------- MPI::Buffer_detach ------------------------
#define NAME     Buffer_detach
#define TEXTNAME "MPI::Buffer_detach"
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

//-------------------------- MPI::Cancel ------------------------
#define NAME     Cancel
#define TEXTNAME "MPI::Cancel"
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

//-------------------------- MPI::Cart_coords ------------------------
#define NAME     Cart_coords
#define TEXTNAME "MPI::Cart_coords"
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

//-------------------------- MPI::Cart_create ------------------------
#define NAME     Cart_create
#define TEXTNAME "MPI::Cart_create"
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

//-------------------------- MPI::Cartdim_get ------------------------
#define NAME     Cartdim_get
#define TEXTNAME "MPI::Cartdim_get"
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

//-------------------------- MPI::Cart_get ------------------------
#define NAME     Cart_get
#define TEXTNAME "MPI::Cart_get"
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

//-------------------------- MPI::Cart_map ------------------------
#define NAME     Cart_map
#define TEXTNAME "MPI::Cart_map"
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

//-------------------------- MPI::Cart_rank ------------------------
#define NAME     Cart_rank
#define TEXTNAME "MPI::Cart_rank"
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

//-------------------------- MPI::Cart_shift ------------------------
#define NAME     Cart_shift
#define TEXTNAME "MPI::Cart_shift"
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

//-------------------------- MPI::Cart_sub ------------------------
#define NAME     Cart_sub
#define TEXTNAME "MPI::Cart_sub"
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

//-------------------------- MPI::Comm_compare ------------------------
#define NAME     Comm_compare
#define TEXTNAME "MPI::Comm_compare"
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

//-------------------------- MPI::Comm_create ------------------------
#define NAME     Comm_create
#define TEXTNAME "MPI::Comm_create"
#define CALLSIG  MPI_Comm arg0, MPI::Group arg1, MPI_Comm* arg2
#define VARS     MPI_Comm arg0; MPI::Group arg1; MPI_Comm* arg2; 
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

//-------------------------- MPI::Comm_dup ------------------------
#define NAME     Comm_dup
#define TEXTNAME "MPI::Comm_dup"
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

//-------------------------- MPI::Comm_free ------------------------
#define NAME     Comm_free
#define TEXTNAME "MPI::Comm_free"
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

//-------------------------- MPI::Comm_get_attr ------------------------
#define NAME     Comm_get_attr
#define TEXTNAME "MPI::Comm_get_attr"
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

//-------------------------- MPI::Comm_get_name ------------------------
#define NAME     Comm_get_name
#define TEXTNAME "MPI::Comm_get_name"
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

//-------------------------- MPI::Comm_group ------------------------
#define NAME     Comm_group
#define TEXTNAME "MPI::Comm_group"
#define CALLSIG  MPI_Comm arg0, MPI::Group* arg1
#define VARS     MPI_Comm arg0; MPI::Group* arg1; 
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

//-------------------------- MPI::Comm_rank ------------------------
#define NAME     Comm_rank
#define TEXTNAME "MPI::Comm_rank"
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

//-------------------------- MPI::Comm_remote_group ------------------------
#define NAME     Comm_remote_group
#define TEXTNAME "MPI::Comm_remote_group"
#define CALLSIG  MPI_Comm arg0, MPI::Group* arg1
#define VARS     MPI_Comm arg0; MPI::Group* arg1; 
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

//-------------------------- MPI::Comm_remote_size ------------------------
#define NAME     Comm_remote_size
#define TEXTNAME "MPI::Comm_remote_size"
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

//-------------------------- MPI::Comm_set_name ------------------------
#define NAME     Comm_set_name
#define TEXTNAME "MPI::Comm_set_name"
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

//-------------------------- MPI::Comm_size ------------------------
#define NAME     Comm_size
#define TEXTNAME "MPI::Comm_size"
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

//-------------------------- MPI::Comm_split ------------------------
#define NAME     Comm_split
#define TEXTNAME "MPI::Comm_split"
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

//-------------------------- MPI::Comm_test_inter ------------------------
#define NAME     Comm_test_inter
#define TEXTNAME "MPI::Comm_test_inter"
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

//-------------------------- MPI::Dims_create ------------------------
#define NAME     Dims_create
#define TEXTNAME "MPI::Dims_create"
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

//-------------------------- MPI::Errhandler_create ------------------------
#define NAME     Errhandler_create
#define TEXTNAME "MPI::Errhandler_create"
#define CALLSIG  MPI::Handler_function* arg0, MPI::Errhandler* arg1
#define VARS     MPI::Handler_function* arg0; MPI::Errhandler* arg1; 
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

//-------------------------- MPI::Errhandler_free ------------------------
#define NAME     Errhandler_free
#define TEXTNAME "MPI::Errhandler_free"
#define CALLSIG  MPI::Errhandler* arg0
#define VARS     MPI::Errhandler* arg0; 
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

//-------------------------- MPI::Errhandler_get ------------------------
#define NAME     Errhandler_get
#define TEXTNAME "MPI::Errhandler_get"
#define CALLSIG  MPI_Comm arg0, MPI::Errhandler* arg1
#define VARS     MPI_Comm arg0; MPI::Errhandler* arg1; 
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

//-------------------------- MPI::Errhandler_set ------------------------
#define NAME     Errhandler_set
#define TEXTNAME "MPI::Errhandler_set"
#define CALLSIG  MPI_Comm arg0, MPI::Errhandler arg1
#define VARS     MPI_Comm arg0; MPI::Errhandler arg1; 
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

//-------------------------- MPI::Error_class ------------------------
#define NAME     Error_class
#define TEXTNAME "MPI::Error_class"
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

//-------------------------- MPI::Error_string ------------------------
#define NAME     Error_string
#define TEXTNAME "MPI::Error_string"
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

//-------------------------- MPI::Finalized ------------------------
#define NAME     Finalized
#define TEXTNAME "MPI::Finalized"
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

//-------------------------- MPI::Free_mem ------------------------
#define NAME     Free_mem
#define TEXTNAME "MPI::Free_mem"
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

//-------------------------- MPI::Gather ------------------------
#define NAME     Gather
#define TEXTNAME "MPI::Gather"
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

//-------------------------- MPI::Gatherv ------------------------
#define NAME     Gatherv
#define TEXTNAME "MPI::Gatherv"
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

//-------------------------- MPI::Get_address ------------------------
#define NAME     Get_address
#define TEXTNAME "MPI::Get_address"
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

//-------------------------- MPI::Get ------------------------
#define NAME     Get
#define TEXTNAME "MPI::Get"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, MPI_Aint arg4, int arg5, MPI_Datatype arg6, MPI::Win arg7
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; MPI_Aint arg4; int arg5; MPI_Datatype arg6; MPI::Win arg7; 
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

//-------------------------- MPI::Get_count ------------------------
#define NAME     Get_count
#define TEXTNAME "MPI::Get_count"
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

//-------------------------- MPI::Get_elements ------------------------
#define NAME     Get_elements
#define TEXTNAME "MPI::Get_elements"
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

//-------------------------- MPI::Get_processor_name ------------------------
#define NAME     Get_processor_name
#define TEXTNAME "MPI::Get_processor_name"
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

//-------------------------- MPI::Get_version ------------------------
#define NAME     Get_version
#define TEXTNAME "MPI::Get_version"
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

//-------------------------- MPI::Graph_create ------------------------
#define NAME     Graph_create
#define TEXTNAME "MPI::Graph_create"
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

//-------------------------- MPI::Graphdims_get ------------------------
#define NAME     Graphdims_get
#define TEXTNAME "MPI::Graphdims_get"
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

//-------------------------- MPI::Graph_get ------------------------
#define NAME     Graph_get
#define TEXTNAME "MPI::Graph_get"
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

//-------------------------- MPI::Graph_map ------------------------
#define NAME     Graph_map
#define TEXTNAME "MPI::Graph_map"
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

//-------------------------- MPI::Graph_neighbors ------------------------
#define NAME     Graph_neighbors
#define TEXTNAME "MPI::Graph_neighbors"
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

//-------------------------- MPI::Graph_neighbors_count ------------------------
#define NAME     Graph_neighbors_count
#define TEXTNAME "MPI::Graph_neighbors_count"
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

//-------------------------- MPI::Group_compare ------------------------
#define NAME     Group_compare
#define TEXTNAME "MPI::Group_compare"
#define CALLSIG  MPI::Group arg0, MPI::Group arg1, int* arg2
#define VARS     MPI::Group arg0; MPI::Group arg1; int* arg2; 
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

//-------------------------- MPI::Group_difference ------------------------
#define NAME     Group_difference
#define TEXTNAME "MPI::Group_difference"
#define CALLSIG  MPI::Group arg0, MPI::Group arg1, MPI::Group* arg2
#define VARS     MPI::Group arg0; MPI::Group arg1; MPI::Group* arg2; 
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

//-------------------------- MPI::Group_excl ------------------------
#define NAME     Group_excl
#define TEXTNAME "MPI::Group_excl"
#define CALLSIG  MPI::Group arg0, int arg1, int* arg2, MPI::Group* arg3
#define VARS     MPI::Group arg0; int arg1; int* arg2; MPI::Group* arg3; 
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

//-------------------------- MPI::Group_free ------------------------
#define NAME     Group_free
#define TEXTNAME "MPI::Group_free"
#define CALLSIG  MPI::Group* arg0
#define VARS     MPI::Group* arg0; 
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

//-------------------------- MPI::Group_incl ------------------------
#define NAME     Group_incl
#define TEXTNAME "MPI::Group_incl"
#define CALLSIG  MPI::Group arg0, int arg1, int* arg2, MPI::Group* arg3
#define VARS     MPI::Group arg0; int arg1; int* arg2; MPI::Group* arg3; 
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

//-------------------------- MPI::Group_intersection ------------------------
#define NAME     Group_intersection
#define TEXTNAME "MPI::Group_intersection"
#define CALLSIG  MPI::Group arg0, MPI::Group arg1, MPI::Group* arg2
#define VARS     MPI::Group arg0; MPI::Group arg1; MPI::Group* arg2; 
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

//-------------------------- MPI::Group_rank ------------------------
#define NAME     Group_rank
#define TEXTNAME "MPI::Group_rank"
#define CALLSIG  MPI::Group arg0, int* arg1
#define VARS     MPI::Group arg0; int* arg1; 
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

//-------------------------- MPI::Group_size ------------------------
#define NAME     Group_size
#define TEXTNAME "MPI::Group_size"
#define CALLSIG  MPI::Group arg0, int* arg1
#define VARS     MPI::Group arg0; int* arg1; 
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

//-------------------------- MPI::Group_translate_ranks ------------------------
#define NAME     Group_translate_ranks
#define TEXTNAME "MPI::Group_translate_ranks"
#define CALLSIG  MPI::Group arg0, int arg1, int* arg2, MPI::Group arg3, int* arg4
#define VARS     MPI::Group arg0; int arg1; int* arg2; MPI::Group arg3; int* arg4; 
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

//-------------------------- MPI::Group_union ------------------------
#define NAME     Group_union
#define TEXTNAME "MPI::Group_union"
#define CALLSIG  MPI::Group arg0, MPI::Group arg1, MPI::Group* arg2
#define VARS     MPI::Group arg0; MPI::Group arg1; MPI::Group* arg2; 
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

//-------------------------- MPI::Ibsend ------------------------
#define NAME     Ibsend
#define TEXTNAME "MPI::Ibsend"
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

//-------------------------- MPI::Info_create ------------------------
#define NAME     Info_create
#define TEXTNAME "MPI::Info_create"
#define CALLSIG  MPI::Info* arg0
#define VARS     MPI::Info* arg0; 
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

//-------------------------- MPI::Info_delete ------------------------
#define NAME     Info_delete
#define TEXTNAME "MPI::Info_delete"
#define CALLSIG  MPI::Info arg0, char* arg1
#define VARS     MPI::Info arg0; char* arg1; 
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

//-------------------------- MPI::Info_dup ------------------------
#define NAME     Info_dup
#define TEXTNAME "MPI::Info_dup"
#define CALLSIG  MPI::Info arg0, MPI::Info* arg1
#define VARS     MPI::Info arg0; MPI::Info* arg1; 
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

//-------------------------- MPI::Info_free ------------------------
#define NAME     Info_free
#define TEXTNAME "MPI::Info_free"
#define CALLSIG  MPI::Info* arg0
#define VARS     MPI::Info* arg0; 
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

//-------------------------- MPI::Info_get ------------------------
#define NAME     Info_get
#define TEXTNAME "MPI::Info_get"
#define CALLSIG  MPI::Info arg0, char* arg1, int arg2, char* arg3, int* arg4
#define VARS     MPI::Info arg0; char* arg1; int arg2; char* arg3; int* arg4; 
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

//-------------------------- MPI::Info_get_nkeys ------------------------
#define NAME     Info_get_nkeys
#define TEXTNAME "MPI::Info_get_nkeys"
#define CALLSIG  MPI::Info arg0, int* arg1
#define VARS     MPI::Info arg0; int* arg1; 
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

//-------------------------- MPI::Info_get_nthkey ------------------------
#define NAME     Info_get_nthkey
#define TEXTNAME "MPI::Info_get_nthkey"
#define CALLSIG  MPI::Info arg0, int arg1, char* arg2
#define VARS     MPI::Info arg0; int arg1; char* arg2; 
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

//-------------------------- MPI::Info_get_valuelen ------------------------
#define NAME     Info_get_valuelen
#define TEXTNAME "MPI::Info_get_valuelen"
#define CALLSIG  MPI::Info arg0, char* arg1, int* arg2, int* arg3
#define VARS     MPI::Info arg0; char* arg1; int* arg2; int* arg3; 
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

//-------------------------- MPI::Info_set ------------------------
#define NAME     Info_set
#define TEXTNAME "MPI::Info_set"
#define CALLSIG  MPI::Info arg0, char* arg1, char* arg2
#define VARS     MPI::Info arg0; char* arg1; char* arg2; 
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

//-------------------------- MPI::Init ------------------------
#define NAME     Init
#define TEXTNAME "MPI::Init"
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

//-------------------------- MPI::Initialized ------------------------
#define NAME     Initialized
#define TEXTNAME "MPI::Initialized"
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

//-------------------------- MPI::Intercomm_create ------------------------
#define NAME     Intercomm_create
#define TEXTNAME "MPI::Intercomm_create"
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

//-------------------------- MPI::Intercomm_merge ------------------------
#define NAME     Intercomm_merge
#define TEXTNAME "MPI::Intercomm_merge"
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

//-------------------------- MPI::Iprobe ------------------------
#define NAME     Iprobe
#define TEXTNAME "MPI::Iprobe"
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

//-------------------------- MPI::Irecv ------------------------
#define NAME     Irecv
#define TEXTNAME "MPI::Irecv"
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

//-------------------------- MPI::Irsend ------------------------
#define NAME     Irsend
#define TEXTNAME "MPI::Irsend"
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

//-------------------------- MPI::Isend ------------------------
#define NAME     Isend
#define TEXTNAME "MPI::Isend"
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

//-------------------------- MPI::Issend ------------------------
#define NAME     Issend
#define TEXTNAME "MPI::Issend"
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

//-------------------------- MPI::Keyval_create ------------------------
#define NAME     Keyval_create
#define TEXTNAME "MPI::Keyval_create"
#define CALLSIG  MPI::Copy_function* arg0, MPI::Delete_function* arg1, int* arg2, void* arg3
#define VARS     MPI::Copy_function* arg0; MPI::Delete_function* arg1; int* arg2; void* arg3; 
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

//-------------------------- MPI::Keyval_free ------------------------
#define NAME     Keyval_free
#define TEXTNAME "MPI::Keyval_free"
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

//-------------------------- MPI::Op_create ------------------------
#define NAME     Op_create
#define TEXTNAME "MPI::Op_create"
#define CALLSIG  MPI::User_function* arg0, int arg1, MPI_Op* arg2
#define VARS     MPI::User_function* arg0; int arg1; MPI_Op* arg2; 
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

//-------------------------- MPI::Op_free ------------------------
#define NAME     Op_free
#define TEXTNAME "MPI::Op_free"
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

//-------------------------- MPI::Pack ------------------------
#define NAME     Pack
#define TEXTNAME "MPI::Pack"
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

//-------------------------- MPI::Pack_size ------------------------
#define NAME     Pack_size
#define TEXTNAME "MPI::Pack_size"
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

//-------------------------- MPI::Probe ------------------------
#define NAME     Probe
#define TEXTNAME "MPI::Probe"
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

//-------------------------- MPI::Put ------------------------
#define NAME     Put
#define TEXTNAME "MPI::Put"
#define CALLSIG  void* arg0, int arg1, MPI_Datatype arg2, int arg3, MPI_Aint arg4, int arg5, MPI_Datatype arg6, MPI::Win arg7
#define VARS     void* arg0; int arg1; MPI_Datatype arg2; int arg3; MPI_Aint arg4; int arg5; MPI_Datatype arg6; MPI::Win arg7; 
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

//-------------------------- MPI::Recv ------------------------
#define NAME     Recv
#define TEXTNAME "MPI::Recv"
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

//-------------------------- MPI::Recv_init ------------------------
#define NAME     Recv_init
#define TEXTNAME "MPI::Recv_init"
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

//-------------------------- MPI::Reduce ------------------------
#define NAME     Reduce
#define TEXTNAME "MPI::Reduce"
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

//-------------------------- MPI::Reduce_scatter ------------------------
#define NAME     Reduce_scatter
#define TEXTNAME "MPI::Reduce_scatter"
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

//-------------------------- MPI::Request_free ------------------------
#define NAME     Request_free
#define TEXTNAME "MPI::Request_free"
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

//-------------------------- MPI::Rsend ------------------------
#define NAME     Rsend
#define TEXTNAME "MPI::Rsend"
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

//-------------------------- MPI::Rsend_init ------------------------
#define NAME     Rsend_init
#define TEXTNAME "MPI::Rsend_init"
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

//-------------------------- MPI::Scan ------------------------
#define NAME     Scan
#define TEXTNAME "MPI::Scan"
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

//-------------------------- MPI::Scatter ------------------------
#define NAME     Scatter
#define TEXTNAME "MPI::Scatter"
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

//-------------------------- MPI::Scatterv ------------------------
#define NAME     Scatterv
#define TEXTNAME "MPI::Scatterv"
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

//-------------------------- MPI::Send ------------------------
#define NAME     Send
#define TEXTNAME "MPI::Send"
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

//-------------------------- MPI::Send_init ------------------------
#define NAME     Send_init
#define TEXTNAME "MPI::Send_init"
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

//-------------------------- MPI::Sendrecv ------------------------
#define NAME     Sendrecv
#define TEXTNAME "MPI::Sendrecv"
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

//-------------------------- MPI::Sendrecv_replace ------------------------
#define NAME     Sendrecv_replace
#define TEXTNAME "MPI::Sendrecv_replace"
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

//-------------------------- MPI::Ssend ------------------------
#define NAME     Ssend
#define TEXTNAME "MPI::Ssend"
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

//-------------------------- MPI::Ssend_init ------------------------
#define NAME     Ssend_init
#define TEXTNAME "MPI::Ssend_init"
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

//-------------------------- MPI::Startall ------------------------
#define NAME     Startall
#define TEXTNAME "MPI::Startall"
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

//-------------------------- MPI::Start ------------------------
#define NAME     Start
#define TEXTNAME "MPI::Start"
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

//-------------------------- MPI::Testall ------------------------
#define NAME     Testall
#define TEXTNAME "MPI::Testall"
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

//-------------------------- MPI::Testany ------------------------
#define NAME     Testany
#define TEXTNAME "MPI::Testany"
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

//-------------------------- MPI::Test ------------------------
#define NAME     Test
#define TEXTNAME "MPI::Test"
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

//-------------------------- MPI::Test_cancelled ------------------------
#define NAME     Test_cancelled
#define TEXTNAME "MPI::Test_cancelled"
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

//-------------------------- MPI::Testsome ------------------------
#define NAME     Testsome
#define TEXTNAME "MPI::Testsome"
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

//-------------------------- MPI::Topo_test ------------------------
#define NAME     Topo_test
#define TEXTNAME "MPI::Topo_test"
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

//-------------------------- MPI::Type_commit ------------------------
#define NAME     Type_commit
#define TEXTNAME "MPI::Type_commit"
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

//-------------------------- MPI::Type_contiguous ------------------------
#define NAME     Type_contiguous
#define TEXTNAME "MPI::Type_contiguous"
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

//-------------------------- MPI::Type_create_darray ------------------------
#define NAME     Type_create_darray
#define TEXTNAME "MPI::Type_create_darray"
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

//-------------------------- MPI::Type_create_subarray ------------------------
#define NAME     Type_create_subarray
#define TEXTNAME "MPI::Type_create_subarray"
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

//-------------------------- MPI::Type_extent ------------------------
#define NAME     Type_extent
#define TEXTNAME "MPI::Type_extent"
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

//-------------------------- MPI::Type_free ------------------------
#define NAME     Type_free
#define TEXTNAME "MPI::Type_free"
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

//-------------------------- MPI::Type_get_contents ------------------------
#define NAME     Type_get_contents
#define TEXTNAME "MPI::Type_get_contents"
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

//-------------------------- MPI::Type_get_envelope ------------------------
#define NAME     Type_get_envelope
#define TEXTNAME "MPI::Type_get_envelope"
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

//-------------------------- MPI::Type_hindexed ------------------------
#define NAME     Type_hindexed
#define TEXTNAME "MPI::Type_hindexed"
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

//-------------------------- MPI::Type_hvector ------------------------
#define NAME     Type_hvector
#define TEXTNAME "MPI::Type_hvector"
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

//-------------------------- MPI::Type_indexed ------------------------
#define NAME     Type_indexed
#define TEXTNAME "MPI::Type_indexed"
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

//-------------------------- MPI::Type_lb ------------------------
#define NAME     Type_lb
#define TEXTNAME "MPI::Type_lb"
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

//-------------------------- MPI::Type_size ------------------------
#define NAME     Type_size
#define TEXTNAME "MPI::Type_size"
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

//-------------------------- MPI::Type_struct ------------------------
#define NAME     Type_struct
#define TEXTNAME "MPI::Type_struct"
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

//-------------------------- MPI::Type_ub ------------------------
#define NAME     Type_ub
#define TEXTNAME "MPI::Type_ub"
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

//-------------------------- MPI::Type_vector ------------------------
#define NAME     Type_vector
#define TEXTNAME "MPI::Type_vector"
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

//-------------------------- MPI::Unpack ------------------------
#define NAME     Unpack
#define TEXTNAME "MPI::Unpack"
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

//-------------------------- MPI::Waitall ------------------------
#define NAME     Waitall
#define TEXTNAME "MPI::Waitall"
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

//-------------------------- MPI::Waitany ------------------------
#define NAME     Waitany
#define TEXTNAME "MPI::Waitany"
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

//-------------------------- MPI::Wait ------------------------
#define NAME     Wait
#define TEXTNAME "MPI::Wait"
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

//-------------------------- MPI::Waitsome ------------------------
#define NAME     Waitsome
#define TEXTNAME "MPI::Waitsome"
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

//-------------------------- MPI::Win_complete ------------------------
#define NAME     Win_complete
#define TEXTNAME "MPI::Win_complete"
#define CALLSIG  MPI::Win arg0
#define VARS     MPI::Win arg0; 
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

//-------------------------- MPI::Win_create ------------------------
#define NAME     Win_create
#define TEXTNAME "MPI::Win_create"
#define CALLSIG  void* arg0, MPI_Aint arg1, int arg2, MPI::Info arg3, MPI_Comm arg4, MPI::Win* arg5
#define VARS     void* arg0; MPI_Aint arg1; int arg2; MPI::Info arg3; MPI_Comm arg4; MPI::Win* arg5; 
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

//-------------------------- MPI::Win_fence ------------------------
#define NAME     Win_fence
#define TEXTNAME "MPI::Win_fence"
#define CALLSIG  int arg0, MPI::Win arg1
#define VARS     int arg0; MPI::Win arg1; 
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

//-------------------------- MPI::Win_free ------------------------
#define NAME     Win_free
#define TEXTNAME "MPI::Win_free"
#define CALLSIG  MPI::Win* arg0
#define VARS     MPI::Win* arg0; 
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

//-------------------------- MPI::Win_get_group ------------------------
#define NAME     Win_get_group
#define TEXTNAME "MPI::Win_get_group"
#define CALLSIG  MPI::Win arg0, MPI::Group* arg1
#define VARS     MPI::Win arg0; MPI::Group* arg1; 
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

//-------------------------- MPI::Win_post ------------------------
#define NAME     Win_post
#define TEXTNAME "MPI::Win_post"
#define CALLSIG  MPI::Group arg0, int arg1, MPI::Win arg2
#define VARS     MPI::Group arg0; int arg1; MPI::Win arg2; 
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

//-------------------------- MPI::Win_start ------------------------
#define NAME     Win_start
#define TEXTNAME "MPI::Win_start"
#define CALLSIG  MPI::Group arg0, int arg1, MPI::Win arg2
#define VARS     MPI::Group arg0; int arg1; MPI::Win arg2; 
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

//-------------------------- MPI::Win_wait ------------------------
#define NAME     Win_wait
#define TEXTNAME "MPI::Win_wait"
#define CALLSIG  MPI::Win arg0
#define VARS     MPI::Win arg0; 
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
#define CALLSIG  MPI::Fint* arg0, MPI_Status* arg1
#define VARS     MPI::Fint* arg0; MPI_Status* arg1; 
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
#define CALLSIG  MPI::Fint arg0
#define VARS     MPI::Fint arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI::Info
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
#define CALLSIG  MPI_Status* arg0, MPI::Fint* arg1
#define VARS     MPI_Status* arg0; MPI::Fint* arg1; 
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
#define CALLSIG  MPI::Info arg0
#define VARS     MPI::Info arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI::Fint
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

#ifdef LAM_MPI // LAM_MPI the following do not have a PMPI_ version on alc

//-------------------------- MPI::Close_port ------------------------
#define NAME     Close_port
#define TEXTNAME "MPI::Close_port"
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

//-------------------------- MPI::Comm_accept ------------------------
#define NAME     Comm_accept
#define TEXTNAME "MPI::Comm_accept"
#define CALLSIG  char* arg0, MPI::Info arg1, int arg2, MPI_Comm arg3, MPI_Comm* arg4
#define VARS     char* arg0; MPI::Info arg1; int arg2; MPI_Comm arg3; MPI_Comm* arg4; 
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

//-------------------------- MPI::Comm_connect ------------------------
#define NAME     Comm_connect
#define TEXTNAME "MPI::Comm_connect"
#define CALLSIG  char* arg0, MPI::Info arg1, int arg2, MPI_Comm arg3, MPI_Comm* arg4
#define VARS     char* arg0; MPI::Info arg1; int arg2; MPI_Comm arg3; MPI_Comm* arg4; 
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

//-------------------------- MPI::Comm_create_errhandler ------------------------
#define NAME     Comm_create_errhandler
#define TEXTNAME "MPI::Comm_create_errhandler"
#define CALLSIG  MPI::Comm_errhandler_fn* arg0, MPI::Errhandler* arg1
#define VARS     MPI::Comm_errhandler_fn* arg0; MPI::Errhandler* arg1; 
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

//-------------------------- MPI::Comm_create_keyval ------------------------
#define NAME     Comm_create_keyval
#define TEXTNAME "MPI::Comm_create_keyval"
#define CALLSIG  MPI::Comm_copy_attr_function* arg0, MPI::Comm_delete_attr_function* arg1, int* arg2, void* arg3
#define VARS     MPI::Comm_copy_attr_function* arg0; MPI::Comm_delete_attr_function* arg1; int* arg2; void* arg3; 
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

//-------------------------- MPI::Comm_delete_attr ------------------------
#define NAME     Comm_delete_attr
#define TEXTNAME "MPI::Comm_delete_attr"
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

//-------------------------- MPI::Comm_disconnect ------------------------
#define NAME     Comm_disconnect
#define TEXTNAME "MPI::Comm_disconnect"
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

//-------------------------- MPI::Comm_free_keyval ------------------------
#define NAME     Comm_free_keyval
#define TEXTNAME "MPI::Comm_free_keyval"
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

//-------------------------- MPI::Comm_get_errhandler ------------------------
#define NAME     Comm_get_errhandler
#define TEXTNAME "MPI::Comm_get_errhandler"
#define CALLSIG  MPI_Comm arg0, MPI::Errhandler* arg1
#define VARS     MPI_Comm arg0; MPI::Errhandler* arg1; 
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

//-------------------------- MPI::Comm_get_parent ------------------------
#define NAME     Comm_get_parent
#define TEXTNAME "MPI::Comm_get_parent"
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

//-------------------------- MPI::Comm_join ------------------------
#define NAME     Comm_join
#define TEXTNAME "MPI::Comm_join"
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

//-------------------------- MPI::Comm_set_attr ------------------------
#define NAME     Comm_set_attr
#define TEXTNAME "MPI::Comm_set_attr"
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

//-------------------------- MPI::Comm_set_errhandler ------------------------
#define NAME     Comm_set_errhandler
#define TEXTNAME "MPI::Comm_set_errhandler"
#define CALLSIG  MPI_Comm arg0, MPI::Errhandler arg1
#define VARS     MPI_Comm arg0; MPI::Errhandler arg1; 
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

//-------------------------- MPI::Comm_spawn ------------------------
#define NAME     Comm_spawn
#define TEXTNAME "MPI::Comm_spawn"
#define CALLSIG  char* arg0, char** arg1, int arg2, MPI::Info arg3, int arg4, MPI_Comm arg5, MPI_Comm* arg6, int* arg7
#define VARS     char* arg0; char** arg1; int arg2; MPI::Info arg3; int arg4; MPI_Comm arg5; MPI_Comm* arg6; int* arg7; 
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

//-------------------------- MPI::Comm_spawn_multiple ------------------------
#define NAME     Comm_spawn_multiple
#define TEXTNAME "MPI::Comm_spawn_multiple"
#define CALLSIG  int arg0, char** arg1, char*** arg2, int* arg3, MPI::Info* arg4, int arg5, MPI_Comm arg6, MPI_Comm* arg7, int* arg8
#define VARS     int arg0; char** arg1; char*** arg2; int* arg3; MPI::Info* arg4; int arg5; MPI_Comm arg6; MPI_Comm* arg7; int* arg8; 
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

//-------------------------- MPI::Is_thread_main ------------------------
#define NAME     Is_thread_main
#define TEXTNAME "MPI::Is_thread_main"
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

//-------------------------- MPI::Lookup_name ------------------------
#define NAME     Lookup_name
#define TEXTNAME "MPI::Lookup_name"
#define CALLSIG  char* arg0, MPI::Info arg1, char* arg2
#define VARS     char* arg0; MPI::Info arg1; char* arg2; 
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

//-------------------------- MPI::Open_port ------------------------
#define NAME     Open_port
#define TEXTNAME "MPI::Open_port"
#define CALLSIG  MPI::Info arg0, char* arg1
#define VARS     MPI::Info arg0; char* arg1; 
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

//-------------------------- MPI::Publish_name ------------------------
#define NAME     Publish_name
#define TEXTNAME "MPI::Publish_name"
#define CALLSIG  char* arg0, MPI::Info arg1, char* arg2
#define VARS     char* arg0; MPI::Info arg1; char* arg2; 
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

//-------------------------- MPI::Query_thread ------------------------
#define NAME     Query_thread
#define TEXTNAME "MPI::Query_thread"
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

//-------------------------- MPI::Type_create_hindexed ------------------------
#define NAME     Type_create_hindexed
#define TEXTNAME "MPI::Type_create_hindexed"
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

//-------------------------- MPI::Type_create_hvector ------------------------
#define NAME     Type_create_hvector
#define TEXTNAME "MPI::Type_create_hvector"
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

//-------------------------- MPI::Type_create_keyval ------------------------
#define NAME     Type_create_keyval
#define TEXTNAME "MPI::Type_create_keyval"
#define CALLSIG  MPI::Type_copy_attr_function* arg0, MPI::Type_delete_attr_function* arg1, int* arg2, void* arg3
#define VARS     MPI::Type_copy_attr_function* arg0; MPI::Type_delete_attr_function* arg1; int* arg2; void* arg3; 
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

//-------------------------- MPI::Type_create_resized ------------------------
#define NAME     Type_create_resized
#define TEXTNAME "MPI::Type_create_resized"
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

//-------------------------- MPI::Type_create_struct ------------------------
#define NAME     Type_create_struct
#define TEXTNAME "MPI::Type_create_struct"
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

//-------------------------- MPI::Type_delete_attr ------------------------
#define NAME     Type_delete_attr
#define TEXTNAME "MPI::Type_delete_attr"
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

//-------------------------- MPI::Type_dup ------------------------
#define NAME     Type_dup
#define TEXTNAME "MPI::Type_dup"
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

//-------------------------- MPI::Type_free_keyval ------------------------
#define NAME     Type_free_keyval
#define TEXTNAME "MPI::Type_free_keyval"
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

//-------------------------- MPI::Type_get_attr ------------------------
#define NAME     Type_get_attr
#define TEXTNAME "MPI::Type_get_attr"
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

//-------------------------- MPI::Type_get_extent ------------------------
#define NAME     Type_get_extent
#define TEXTNAME "MPI::Type_get_extent"
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

//-------------------------- MPI::Type_get_name ------------------------
#define NAME     Type_get_name
#define TEXTNAME "MPI::Type_get_name"
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

//-------------------------- MPI::Type_get_true_extent ------------------------
#define NAME     Type_get_true_extent
#define TEXTNAME "MPI::Type_get_true_extent"
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

//-------------------------- MPI::Type_set_attr ------------------------
#define NAME     Type_set_attr
#define TEXTNAME "MPI::Type_set_attr"
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

//-------------------------- MPI::Type_set_name ------------------------
#define NAME     Type_set_name
#define TEXTNAME "MPI::Type_set_name"
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

//-------------------------- MPI::Unpublish_name ------------------------
#define NAME     Unpublish_name
#define TEXTNAME "MPI::Unpublish_name"
#define CALLSIG  char* arg0, MPI::Info arg1, char* arg2
#define VARS     char* arg0; MPI::Info arg1; char* arg2; 
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

//-------------------------- MPI::Win_create_errhandler ------------------------
#define NAME     Win_create_errhandler
#define TEXTNAME "MPI::Win_create_errhandler"
#define CALLSIG  MPI::Win_errhandler_fn* arg0, MPI::Errhandler* arg1
#define VARS     MPI::Win_errhandler_fn* arg0; MPI::Errhandler* arg1; 
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

//-------------------------- MPI::Win_create_keyval ------------------------
#define NAME     Win_create_keyval
#define TEXTNAME "MPI::Win_create_keyval"
#define CALLSIG  MPI::Win_copy_attr_function* arg0, MPI::Win_delete_attr_function* arg1, int* arg2, void* arg3
#define VARS     MPI::Win_copy_attr_function* arg0; MPI::Win_delete_attr_function* arg1; int* arg2; void* arg3; 
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

//-------------------------- MPI::Win_delete_attr ------------------------
#define NAME     Win_delete_attr
#define TEXTNAME "MPI::Win_delete_attr"
#define CALLSIG  MPI::Win arg0, int arg1
#define VARS     MPI::Win arg0; int arg1; 
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

//-------------------------- MPI::Win_free_keyval ------------------------
#define NAME     Win_free_keyval
#define TEXTNAME "MPI::Win_free_keyval"
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

//-------------------------- MPI::Win_get_attr ------------------------
#define NAME     Win_get_attr
#define TEXTNAME "MPI::Win_get_attr"
#define CALLSIG  MPI::Win arg0, int arg1, void* arg2, int* arg3
#define VARS     MPI::Win arg0; int arg1; void* arg2; int* arg3; 
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

//-------------------------- MPI::Win_get_errhandler ------------------------
#define NAME     Win_get_errhandler
#define TEXTNAME "MPI::Win_get_errhandler"
#define CALLSIG  MPI::Win arg0, MPI::Errhandler* arg1
#define VARS     MPI::Win arg0; MPI::Errhandler* arg1; 
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

//-------------------------- MPI::Win_get_name ------------------------
#define NAME     Win_get_name
#define TEXTNAME "MPI::Win_get_name"
#define CALLSIG  MPI::Win arg0, char* arg1, int* arg2
#define VARS     MPI::Win arg0; char* arg1; int* arg2; 
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

//-------------------------- MPI::Win_set_attr ------------------------
#define NAME     Win_set_attr
#define TEXTNAME "MPI::Win_set_attr"
#define CALLSIG  MPI::Win arg0, int arg1, void* arg2
#define VARS     MPI::Win arg0; int arg1; void* arg2; 
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

//-------------------------- MPI::Win_set_errhandler ------------------------
#define NAME     Win_set_errhandler
#define TEXTNAME "MPI::Win_set_errhandler"
#define CALLSIG  MPI::Win arg0, MPI::Errhandler arg1
#define VARS     MPI::Win arg0; MPI::Errhandler arg1; 
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

//-------------------------- MPI::Win_set_name ------------------------
#define NAME     Win_set_name
#define TEXTNAME "MPI::Win_set_name"
#define CALLSIG  MPI::Win arg0, char* arg1
#define VARS     MPI::Win arg0; char* arg1; 
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
#define RET_TYPE MPI::Fint
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
#define CALLSIG  MPI::Errhandler arg0
#define VARS     MPI::Errhandler arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI::Fint
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
#define CALLSIG  MPI::Group arg0
#define VARS     MPI::Group arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI::Fint
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
#define RET_TYPE MPI::Fint
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
#define RET_TYPE MPI::Fint
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
#define CALLSIG  MPI::Fint arg0
#define VARS     MPI::Fint arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI::Win
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
#define CALLSIG  MPI::Fint arg0
#define VARS     MPI::Fint arg0; 
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
#define CALLSIG  MPI::Fint arg0
#define VARS     MPI::Fint arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI::Errhandler
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
#define CALLSIG  MPI::Fint arg0
#define VARS     MPI::Fint arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI::Group
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
#define RET_TYPE MPI::Fint
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
#define CALLSIG  MPI::Win arg0
#define VARS     MPI::Win arg0; 
#define CALLARGS arg0
#define VAR_INIT arg0(arg0), retval(retval)
#define RET_TYPE MPI::Fint
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
#define CALLSIG  MPI::Fint arg0
#define VARS     MPI::Fint arg0; 
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
#define CALLSIG  MPI::Fint arg0
#define VARS     MPI::Fint arg0; 
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

