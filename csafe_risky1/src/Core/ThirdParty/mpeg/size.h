/*************************************************************
Copyright (C) 1990, 1991, 1993 Andy C. Hung, all rights reserved.
PUBLIC DOMAIN LICENSE: Stanford University Portable Video Research
Group. If you use this software, you agree to the following: This
program package is purely experimental, and is licensed "as is".
Permission is granted to use, modify, and distribute this program
without charge for any purpose, provided this license/ disclaimer
notice appears in the copies.  No warranty or maintenance is given,
either expressed or implied.  In no event shall the author(s) be
liable to you or a third party for any special, incidental,
consequential, or other damages, arising out of the use or inability
to use the program for any purpose (or the loss of data), even if we
have been advised of such possibilities.  Any public reference or
advertisement of this source code should refer to it as the Portable
Video Research Group (PVRG) code, and not by any author(s) (or
Stanford University) name.
*************************************************************/

extern BLOCK MakeBlock();
extern void CopyBlock();
extern void CopyMem();
extern ClearMem();
extern int ParityMem();
extern void SetPointerBlock();
extern MEM *MakeMem();
extern void FreeMem();
extern MEM *LoadMem();
extern void YUVLoadMem();
extern void YUVSaveMem();
extern MEM *SaveMem();
extern MEM *MakeSubMem();
extern MEM *MakeSuperMem();
extern MEM *SM0HDecimateMem();
extern MEM *SM0VDecimateMem();
extern MEM *JP0HDecimateMem();
extern MEM *JP0VDecimateMem();
extern MEM *XHInterpolateMem();
extern MEM *XVInterpolateMem();
extern MEM *SM0HInterpolateMem();
extern MEM *JVCHInterpolateMem();
extern MEM *CECASHInterpolateMem();
extern MEM *BellCoreHInterpolateMem();
extern MEM *SonyHInterpolateMem();
extern MEM *SM0VInterpolateMem();
extern MEM *JVCVInterpolateMem();
extern MEM *CECASVInterpolateMem();
extern MEM *BellCoreVInterpolateMem();
extern MEM *SonyVInterpolateMem();
