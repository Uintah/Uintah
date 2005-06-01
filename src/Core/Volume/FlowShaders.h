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
//    File   : FlowShaders.h
//    Author : Kurt Zimmerman
//    Date   : May 2005

#ifndef FlowShaders_h
#define FlowShaders_h

#include <string>

namespace SCIRun {

static const string AdvInit = 
"!!ARBfp1.0 \n"
"PARAM c[3] = { { 0.5, 0.25 }, \n"
"                program.local[1..2] }; \n"
"#PARAM c[3] = { { 0.05, 0.25 }, \n"
"#               { 1.0, 1.0, 1.0, 1.0 }, \n"
"#               { 0.5, 0.5, 0.5, 0.5 } }; \n"
"TEMP R0; \n"
"ADD R0.xy, fragment.texcoord[0], c[1]; \n"
"TEX R0.x, R0, texture[2], 2D; \n"
"MAD result.color.xy, fragment.texcoord[0], c[0].x, c[0].y; \n"
"MUL result.color.z, R0.x, c[2].x; \n"
"MOV result.color.w, c[2].x; \n"
"END";

static const string AdvAccum =
"!!ARBfp1.0 \n"
"PARAM c[4] = { program.local[0], \n"
"                { 2, 0.5, 4 }, \n"
"                program.local[2..3] }; \n"
"TEMP R0; \n"
"TEMP R1; \n"
"TEMP R2; \n"
"TEX R0, fragment.texcoord[0], texture[1], 2D; \n"
"MAD R1.zw, R0.xyxy, c[1].x, -c[1].y; \n"
"TEX R1.xy, R1.zwzw, texture[0], 2D; \n"
"ADD R1.xy, R1, -c[1].y; \n"
"ADD R1.zw, R1, c[2].xyxy; \n"
"MUL R1.xy, c[0].x, R1; \n"
"TEX R2.x, R1.zwzw, texture[2], 2D; \n"
"MAD result.color.xy, R1, c[1].z, R0; \n"
"MAD result.color.z, R2.x, c[3].x, R0; \n"
"ADD result.color.w, R0, c[3].x; \n"
"END";

static const string AdvRewire = 
"!!ARBfp1.0 \n"
"PARAM c[1] = { { 1 } }; \n"
"TEMP R0; \n"
"TEX R0.zw, fragment.texcoord[1], texture[1], 2D; \n"
"RCP R0.x, R0.w; \n"
"MUL result.color.xyz, R0.z, R0.x; \n"
"MOV result.color.w, c[0].x; \n"
"END";



static const string ConvInit = 
"!!ARBfp1.0 \n"
"PARAM c[1] = { program.local[0] }; \n"
"TEMP R0; \n"
"TEX R0.x, fragment.texcoord[0], texture[0], 2D; \n"
"MOV result.color.xy, fragment.texcoord[0]; \n"
"MUL result.color.z, R0.x, c[0].x; \n"
"MOV result.color.w, c[0].x; \n"
"END";

static const string ConvAccum = 
"!!ARBfp1.0 \n"
"PARAM c[3] = { { 0.5 }, \n"
"                program.local[1..2] }; \n"
"TEMP R0; \n"
"TEMP R1; \n"
"TEX R0, fragment.texcoord[0], texture[1], 2D; \n"
"TEX R1.xy, R0, texture[0], 2D; \n"
"ADD R1.xy, R1, -c[0].x; \n"
"MUL R1.zw, R1.xyxy, R1.xyxy; \n"
"ADD R1.z, R1, R1.w; \n"
"RSQ R1.z, R1.z; \n"
"MUL R1.xy, R1.z, R1; \n"
"MAD R0.xy, R1, c[1].x, R0; \n"
"TEX R1.x, R0, texture[2], 2D; \n"
"ADD R0.w, R0, c[2].x; \n"
"MAD R0.z, R1.x, c[2].x, R0; \n"
"MOV result.color, R0; \n"
"END";

static const string ConvRewire = 
"!!ARBfp1.0 \n"
"PARAM c[1] = { { 1 } }; \n"
"TEMP R0; \n"
"TEX R0.zw, fragment.texcoord[0], texture[0], 2D; \n"
"RCP R0.x, R0.w; \n"
"MUL result.color.xyz, R0.z, R0.x; \n"
"MOV result.color.w, c[0].x; \n"
"END";



//  For Nvidia

static const string AdvInitRect = 
"!!ARBfp1.0 \n"
"PARAM c[3] = { { 0.5, 0.25 }, \n"
"                program.local[1..2] }; \n"
"TEMP R0; \n"
"ADD R0.xy, fragment.texcoord[0], c[1]; \n"
"TEX R0.x, R0, texture[0], RECT; \n"
"MAD result.color.xy, fragment.texcoord[0], c[0].x, c[0].y; \n"
"MUL result.color.z, R0.x, c[2].x; \n"
"MOV result.color.w, c[2].x; \n"
"END";

static const string AdvAccumRect =
"!!ARBfp1.0 \n"
"PARAM c[4] = { program.local[0], \n"
"                { 2, 0.5, 4 }, \n"
"                program.local[2..3] }; \n"
"TEMP R0; \n"
"TEMP R1; \n"
"TEMP R2; \n"
"TEX R0, fragment.texcoord[0], texture[1], RECT; \n"
"MAD R1.zw, R0.xyxy, c[1].x, -c[1].y; \n"
"TEX R1.xy, R1.zwzw, texture[0], RECT; \n"
"ADD R1.xy, R1, -c[1].y; \n"
"ADD R1.zw, R1, c[2].xyxy; \n"
"MUL R1.xy, c[0].x, R1; \n"
"TEX R2.x, R1.zwzw, texture[2], RECT; \n"
"MAD result.color.xy, R1, c[1].z, R0; \n"
"MAD result.color.z, R2.x, c[3].x, R0; \n"
"ADD result.color.w, R0, c[3].x; \n"
"END";

static const string AdvRewireRect = 
"!!ARBfp1.0 \n"
"PARAM c[1] = { { 1 } }; \n"
"TEMP R0; \n"
"TEX R0.zw, fragment.texcoord[0], texture[1], RECT; \n"
"RCP R0.x, R0.w; \n"
"MUL result.color.xyz, R0.z, R0.x; \n"
"MOV result.color.w, c[0].x; \n"
"END";



static const string ConvInitRect = 
"!!ARBfp1.0 \n"
"PARAM c[1] = { program.local[0] }; \n"
"TEMP R0; \n"
"TEX R0.x, fragment.texcoord[0], texture[0], RECT; \n"
"MOV result.color.xy, fragment.texcoord[0]; \n"
"MUL result.color.z, R0.x, c[0].x; \n"
"MOV result.color.w, c[0].x; \n"
"END";

static const string ConvAccumRect = 
"!!ARBfp1.0 \n"
"PARAM c[3] = { { 0.5 }, \n"
"                program.local[1..2] }; \n"
"TEMP R0; \n"
"TEMP R1; \n"
"TEX R0, fragment.texcoord[0], texture[1], RECT; \n"
"TEX R1.xy, R0, texture[0], RECT; \n"
"ADD R1.xy, R1, -c[0].x; \n"
"MUL R1.zw, R1.xyxy, R1.xyxy; \n"
"ADD R1.z, R1, R1.w; \n"
"RSQ R1.z, R1.z; \n"
"MUL R1.xy, R1.z, R1; \n"
"MAD R0.xy, R1, c[1].x, R0; \n"
"TEX R1.x, R0, texture[2], RECT; \n"
"ADD R0.w, R0, c[2].x; \n"
"MAD R0.z, R1.x, c[2].x, R0; \n"
"MOV result.color, R0; \n"
"END";

static const string ConvRewireRect = 
"!!ARBfp1.0 \n"
"PARAM c[1] = { { 1 } }; \n"
"TEMP R0; \n"
"TEX R0.zw, fragment.texcoord[0], texture[0], RECT; \n"
"RCP R0.x, R0.w; \n"
"MUL result.color.xyz, R0.z, R0.x; \n"
"MOV result.color.w, c[0].x; \n"
"END";

static const string XToColor =
"!!ARBfp1.0 \n"
"# bla bla bla \n"
"TEMP X; \n"
"TEMP C; \n"
"TEX X, fragment.texcoord[0], texture[0], 2D; \n"
"TEX C, X.x, texture[2], 1D; \n"
"MOV result.color, C; \n"
"END";

static const string YToColor = 
"!!ARBfp1.0 \n"
"TEMP Y; \n"
"TEMP C; \n"
"TEX Y, fragment.texcoord[0], texture[0], 2D; \n"
"TEX C, Y.y, texture[2], 1D; \n"
"MOV result.color, C; \n"
"END";

static const string DrawNoise =
"!!ARBfp1.0 \n"
"TEMP X; \n"
"TEMP C; \n"
"TEX X, fragment.texcoord[1], texture[1], 2D; \n"
"TEX C, X.z, texture[3], 1D; \n"
"MOV result.color, C; \n"
"END";


} // end namespace SCIRun

#endif // FlowShaders_h
