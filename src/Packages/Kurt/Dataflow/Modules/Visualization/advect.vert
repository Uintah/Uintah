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

varying vec2 TexCoord;
void main(void)
{
    vec4  vert = gl_Vertex;
    TexCoord = gl_MultiTexCoord0.st;
    gl_Position  = gl_ModelViewProjectionMatrix * vert;
}