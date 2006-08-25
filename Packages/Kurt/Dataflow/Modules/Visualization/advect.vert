//
// Vertex shader for rendering particles in flow
// via a partcle system
//
// Author: Randi Rost
//
// Copyright (c) 2003-2004: 3Dlabs, Inc.
//
// See 3Dlabs-License.txt for license information
//

varying vec2 TexCoord0;
varying vec2 TexCoord1;
void main(void)
{
    vec4  vert = gl_Vertex;
    TexCoord0 = gl_MultiTexCoord0.st;
    TexCoord1 = gl_MultiTexCoord1.st;
    gl_Position  = gl_ModelViewProjectionMatrix * vert;
}