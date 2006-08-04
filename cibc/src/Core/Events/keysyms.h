//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : keysyms.h
//    Author : Martin Cole
//    Date   : Mon Jun  5 13:52:01 2006

// Based on the gdk keysysm
// http://www.gtk.org/
// which was based on X11/keysymdef.h, which is:
// Copyright 1987 by Digital Equipment Corporation, Maynard, Massachusetts



#ifndef __SCIRun_KEYSYMS_H__
#define __SCIRun_KEYSYMS_H__


#define SCIRun_VoidSymbol 0xFFFFFF
#define SCIRun_BackSpace 0xFF08
#define SCIRun_Tab 0xFF09
#define SCIRun_Linefeed 0xFF0A
#define SCIRun_Clear 0xFF0B
#define SCIRun_Return 0xFF0D
#define SCIRun_Pause 0xFF13
#define SCIRun_Scroll_Lock 0xFF14
#define SCIRun_Sys_Req 0xFF15
#define SCIRun_Escape 0xFF1B
#define SCIRun_Delete 0xFFFF
#define SCIRun_Multi_key 0xFF20
#define SCIRun_Home 0xFF50
#define SCIRun_Left 0xFF51
#define SCIRun_Up 0xFF52
#define SCIRun_Right 0xFF53
#define SCIRun_Down 0xFF54
#define SCIRun_Prior 0xFF55
#define SCIRun_Page_Up 0xFF55
#define SCIRun_Next 0xFF56
#define SCIRun_Page_Down 0xFF56
#define SCIRun_End 0xFF57
#define SCIRun_Begin 0xFF58
#define SCIRun_Select 0xFF60
#define SCIRun_Print 0xFF61
#define SCIRun_Execute 0xFF62
#define SCIRun_Insert 0xFF63
#define SCIRun_Undo 0xFF65
#define SCIRun_Redo 0xFF66
#define SCIRun_Menu 0xFF67
#define SCIRun_Find 0xFF68
#define SCIRun_Cancel 0xFF69
#define SCIRun_Help 0xFF6A
#define SCIRun_Break 0xFF6B
#define SCIRun_Num_Lock 0xFF7F
#define SCIRun_KP_Space 0xFF80
#define SCIRun_KP_Tab 0xFF89
#define SCIRun_KP_Enter 0xFF8D
#define SCIRun_KP_F1 0xFF91
#define SCIRun_KP_F2 0xFF92
#define SCIRun_KP_F3 0xFF93
#define SCIRun_KP_F4 0xFF94
#define SCIRun_KP_Home 0xFF95
#define SCIRun_KP_Left 0xFF96
#define SCIRun_KP_Up 0xFF97
#define SCIRun_KP_Right 0xFF98
#define SCIRun_KP_Down 0xFF99
#define SCIRun_KP_Prior 0xFF9A
#define SCIRun_KP_Page_Up 0xFF9A
#define SCIRun_KP_Next 0xFF9B
#define SCIRun_KP_Page_Down 0xFF9B
#define SCIRun_KP_End 0xFF9C
#define SCIRun_KP_Begin 0xFF9D
#define SCIRun_KP_Insert 0xFF9E
#define SCIRun_KP_Delete 0xFF9F
#define SCIRun_KP_Equal 0xFFBD
#define SCIRun_KP_Multiply 0xFFAA
#define SCIRun_KP_Add 0xFFAB
#define SCIRun_KP_Separator 0xFFAC
#define SCIRun_KP_Subtract 0xFFAD
#define SCIRun_KP_Decimal 0xFFAE
#define SCIRun_KP_Divide 0xFFAF
#define SCIRun_KP_0 0xFFB0
#define SCIRun_KP_1 0xFFB1
#define SCIRun_KP_2 0xFFB2
#define SCIRun_KP_3 0xFFB3
#define SCIRun_KP_4 0xFFB4
#define SCIRun_KP_5 0xFFB5
#define SCIRun_KP_6 0xFFB6
#define SCIRun_KP_7 0xFFB7
#define SCIRun_KP_8 0xFFB8
#define SCIRun_KP_9 0xFFB9
#define SCIRun_F1 0xFFBE
#define SCIRun_F2 0xFFBF
#define SCIRun_F3 0xFFC0
#define SCIRun_F4 0xFFC1
#define SCIRun_F5 0xFFC2
#define SCIRun_F6 0xFFC3
#define SCIRun_F7 0xFFC4
#define SCIRun_F8 0xFFC5
#define SCIRun_F9 0xFFC6
#define SCIRun_F10 0xFFC7
#define SCIRun_F11 0xFFC8
#define SCIRun_L1 0xFFC8
#define SCIRun_F12 0xFFC9
#define SCIRun_L2 0xFFC9
#define SCIRun_F13 0xFFCA
#define SCIRun_L3 0xFFCA
#define SCIRun_F14 0xFFCB
#define SCIRun_L4 0xFFCB
#define SCIRun_F15 0xFFCC
#define SCIRun_L5 0xFFCC
#define SCIRun_F16 0xFFCD
#define SCIRun_L6 0xFFCD
#define SCIRun_F17 0xFFCE
#define SCIRun_L7 0xFFCE
#define SCIRun_F18 0xFFCF
#define SCIRun_L8 0xFFCF
#define SCIRun_F19 0xFFD0
#define SCIRun_L9 0xFFD0
#define SCIRun_F20 0xFFD1
#define SCIRun_L10 0xFFD1
#define SCIRun_F21 0xFFD2
#define SCIRun_R1 0xFFD2
#define SCIRun_F22 0xFFD3
#define SCIRun_R2 0xFFD3
#define SCIRun_F23 0xFFD4
#define SCIRun_R3 0xFFD4
#define SCIRun_F24 0xFFD5
#define SCIRun_R4 0xFFD5
#define SCIRun_F25 0xFFD6
#define SCIRun_R5 0xFFD6
#define SCIRun_F26 0xFFD7
#define SCIRun_R6 0xFFD7
#define SCIRun_F27 0xFFD8
#define SCIRun_R7 0xFFD8
#define SCIRun_F28 0xFFD9
#define SCIRun_R8 0xFFD9
#define SCIRun_F29 0xFFDA
#define SCIRun_R9 0xFFDA
#define SCIRun_F30 0xFFDB
#define SCIRun_R10 0xFFDB
#define SCIRun_F31 0xFFDC
#define SCIRun_R11 0xFFDC
#define SCIRun_F32 0xFFDD
#define SCIRun_R12 0xFFDD
#define SCIRun_F33 0xFFDE
#define SCIRun_R13 0xFFDE
#define SCIRun_F34 0xFFDF
#define SCIRun_R14 0xFFDF
#define SCIRun_F35 0xFFE0
#define SCIRun_R15 0xFFE0
#define SCIRun_Shift_L 0xFFE1
#define SCIRun_Shift_R 0xFFE2
#define SCIRun_Control_L 0xFFE3
#define SCIRun_Control_R 0xFFE4
#define SCIRun_Caps_Lock 0xFFE5
#define SCIRun_Shift_Lock 0xFFE6
#define SCIRun_Meta_L 0xFFE7
#define SCIRun_Meta_R 0xFFE8
#define SCIRun_Alt_L 0xFFE9
#define SCIRun_Alt_R 0xFFEA
#define SCIRun_Pointer_Left 0xFEE0
#define SCIRun_Pointer_Right 0xFEE1
#define SCIRun_Pointer_Up 0xFEE2
#define SCIRun_Pointer_Down 0xFEE3
#define SCIRun_Pointer_UpLeft 0xFEE4
#define SCIRun_Pointer_UpRight 0xFEE5
#define SCIRun_Pointer_DownLeft 0xFEE6
#define SCIRun_Pointer_DownRight 0xFEE7
#define SCIRun_Pointer_Button_Dflt 0xFEE8
#define SCIRun_Pointer_Button1 0xFEE9
#define SCIRun_Pointer_Button2 0xFEEA
#define SCIRun_Pointer_Button3 0xFEEB
#define SCIRun_Pointer_Button4 0xFEEC
#define SCIRun_Pointer_Button5 0xFEED
#define SCIRun_Pointer_DblClick_Dflt 0xFEEE
#define SCIRun_Pointer_DblClick1 0xFEEF
#define SCIRun_Pointer_DblClick2 0xFEF0
#define SCIRun_Pointer_DblClick3 0xFEF1
#define SCIRun_Pointer_DblClick4 0xFEF2
#define SCIRun_Pointer_DblClick5 0xFEF3
#define SCIRun_Pointer_Drag_Dflt 0xFEF4
#define SCIRun_Pointer_Drag1 0xFEF5
#define SCIRun_Pointer_Drag2 0xFEF6
#define SCIRun_Pointer_Drag3 0xFEF7
#define SCIRun_Pointer_Drag4 0xFEF8
#define SCIRun_Pointer_Drag5 0xFEFD
#define SCIRun_Pointer_EnableKeys 0xFEF9
#define SCIRun_Pointer_Accelerate 0xFEFA
#define SCIRun_Pointer_DfltBtnNext 0xFEFB
#define SCIRun_Pointer_DfltBtnPrev 0xFEFC
#define SCIRun_space 0x020
#define SCIRun_exclam 0x021
#define SCIRun_quotedbl 0x022
#define SCIRun_numbersign 0x023
#define SCIRun_dollar 0x024
#define SCIRun_percent 0x025
#define SCIRun_ampersand 0x026
#define SCIRun_apostrophe 0x027
#define SCIRun_quoteright 0x027
#define SCIRun_parenleft 0x028
#define SCIRun_parenright 0x029
#define SCIRun_asterisk 0x02a
#define SCIRun_plus 0x02b
#define SCIRun_comma 0x02c
#define SCIRun_minus 0x02d
#define SCIRun_period 0x02e
#define SCIRun_slash 0x02f
#define SCIRun_0 0x030
#define SCIRun_1 0x031
#define SCIRun_2 0x032
#define SCIRun_3 0x033
#define SCIRun_4 0x034
#define SCIRun_5 0x035
#define SCIRun_6 0x036
#define SCIRun_7 0x037
#define SCIRun_8 0x038
#define SCIRun_9 0x039
#define SCIRun_colon 0x03a
#define SCIRun_semicolon 0x03b
#define SCIRun_less 0x03c
#define SCIRun_equal 0x03d
#define SCIRun_greater 0x03e
#define SCIRun_question 0x03f
#define SCIRun_at 0x040
#define SCIRun_A 0x041
#define SCIRun_B 0x042
#define SCIRun_C 0x043
#define SCIRun_D 0x044
#define SCIRun_E 0x045
#define SCIRun_F 0x046
#define SCIRun_G 0x047
#define SCIRun_H 0x048
#define SCIRun_I 0x049
#define SCIRun_J 0x04a
#define SCIRun_K 0x04b
#define SCIRun_L 0x04c
#define SCIRun_M 0x04d
#define SCIRun_N 0x04e
#define SCIRun_O 0x04f
#define SCIRun_P 0x050
#define SCIRun_Q 0x051
#define SCIRun_R 0x052
#define SCIRun_S 0x053
#define SCIRun_T 0x054
#define SCIRun_U 0x055
#define SCIRun_V 0x056
#define SCIRun_W 0x057
#define SCIRun_X 0x058
#define SCIRun_Y 0x059
#define SCIRun_Z 0x05a
#define SCIRun_bracketleft 0x05b
#define SCIRun_backslash 0x05c
#define SCIRun_bracketright 0x05d
#define SCIRun_asciicircum 0x05e
#define SCIRun_underscore 0x05f
#define SCIRun_grave 0x060
#define SCIRun_quoteleft 0x060
#define SCIRun_a 0x061
#define SCIRun_b 0x062
#define SCIRun_c 0x063
#define SCIRun_d 0x064
#define SCIRun_e 0x065
#define SCIRun_f 0x066
#define SCIRun_g 0x067
#define SCIRun_h 0x068
#define SCIRun_i 0x069
#define SCIRun_j 0x06a
#define SCIRun_k 0x06b
#define SCIRun_l 0x06c
#define SCIRun_m 0x06d
#define SCIRun_n 0x06e
#define SCIRun_o 0x06f
#define SCIRun_p 0x070
#define SCIRun_q 0x071
#define SCIRun_r 0x072
#define SCIRun_s 0x073
#define SCIRun_t 0x074
#define SCIRun_u 0x075
#define SCIRun_v 0x076
#define SCIRun_w 0x077
#define SCIRun_x 0x078
#define SCIRun_y 0x079
#define SCIRun_z 0x07a
#define SCIRun_braceleft 0x07b
#define SCIRun_bar 0x07c
#define SCIRun_braceright 0x07d
#define SCIRun_asciitilde 0x07e
#define SCIRun_nobreakspace 0x0a0
#define SCIRun_exclamdown 0x0a1
#define SCIRun_notsign 0x0ac
#define SCIRun_hyphen 0x0ad
#define SCIRun_function 0x8f6
#define SCIRun_leftarrow 0x8fb
#define SCIRun_uparrow 0x8fc
#define SCIRun_rightarrow 0x8fd
#define SCIRun_downarrow 0x8fe
#define SCIRun_blank 0x9df

#endif /* __SCIRun_KEYSYMS_H__ */
