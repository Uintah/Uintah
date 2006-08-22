//
// Fragment shader for rendering a "confetti cannon"
// via a partcle system
//
// Author: Randi Rost
//
// Copyright (c) 2003-2004: 3Dlabs, Inc.
//
// See 3Dlabs-License.txt for license information
//

uniform sampler2D StartPositions;
uniform sampler2D Positions;
uniform sampler3D Flow;

uniform mat4 ParticleTrans;
uniform mat4 MeshTrans;

uniform float Time;
uniform float Step;

varying vec2 TexCoord;

bool out_of_bounds( vec4 pos )
{
     
    vec4 p = MeshTrans * pos;
//    vec3 shift = vec3( 0.5 );
//    vec3 r = p.xyz + shift;
      vec3 r = p.xyz;

    if (r.x < 0.0 || r.y < 0.0 || r.z < 0.0 ||
        r.x >= 1.0 || r.y >= 1.0 || r.z >= 1.0 ) {
      return true;
    } else {
      return false;
    }
}

void main (void)
{
    // get the stored position value
    vec4 pos = texture2D(Positions, TexCoord);

    // transform this to get the flow texture index
    vec3 flow_pos = (MeshTrans * vec4(pos.xyz, 1.0)).xyz;
    vec4 dir = texture3D(Flow, flow_pos);
    pos += vec4(dir.xyz, Time)*Step;

    if (pos.w <= 0.0 || out_of_bounds( vec4( vec3(pos), 1.0) ) ) {   
        vec4 start_pos =  texture2D( StartPositions, TexCoord );
        pos = ParticleTrans * vec4(start_pos.xyz, 1.0);
        pos.w = start_pos.w;
    }
  
    gl_FragColor = pos;
}

