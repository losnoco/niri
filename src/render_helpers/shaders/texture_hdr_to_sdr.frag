#version 100

//_DEFINES_

#if defined(EXTERNAL)
#extension GL_OES_EGL_image_external : require
#endif

precision highp float;
#if defined(EXTERNAL)
uniform samplerExternalOES tex;
#else
uniform sampler2D tex;
#endif

uniform float alpha;
uniform float niri_ref_lum_scale;
varying vec2 v_coords;

#if defined(DEBUG_FLAGS)
uniform float tint;
#endif

vec3 niri_pq_eotf(vec3 pq) {
    const float pq_m1 = 0.1593017578125;
    const float pq_m2 = 78.84375;
    const float pq_c1 = 0.8359375;
    const float pq_c2 = 18.8515625;
    const float pq_c3 = 18.6875;

    vec3 p = pow(clamp(pq, 0.0, 1.0), vec3(1.0 / pq_m2));
    vec3 n = max(p - vec3(pq_c1), vec3(0.0));
    vec3 d = max(vec3(pq_c2) - pq_c3 * p, vec3(0.000001));
    return pow(n / d, vec3(1.0 / pq_m1));
}

// Premultiplied PQ/BT.2020 in, premultiplied electrical sRGB out.
vec4 niri_hdr_to_sdr(vec4 color) {
    float a = color.a;
    vec3 rgb = a > 0.0 ? color.rgb / a : color.rgb;

    rgb = niri_pq_eotf(rgb);

    // BT.2020 -> BT.709, linear light, D65 (column-major).
    const mat3 to_bt709 = mat3(
        1.660491, -0.124550, -0.018151,
       -0.587641,  1.132900, -0.100579,
       -0.072850, -0.008349,  1.118730);
    rgb = to_bt709 * rgb;

    // Convert absolute PQ luminance to the SDR reference white used by niri's HDR blend path.
    float ref_scale = niri_ref_lum_scale > 0.0 ? niri_ref_lum_scale : 0.0203;
    rgb = clamp(rgb / ref_scale, 0.0, 1.0);

    // Match niri_blend()'s 2.2 power decode counterpart.
    rgb = pow(rgb, vec3(1.0 / 2.2));
    return vec4(rgb * a, a);
}

void main() {
    vec4 color = texture2D(tex, v_coords);

#if defined(NO_ALPHA)
    color = vec4(color.rgb, 1.0) * alpha;
#else
    color = color * alpha;
#endif

    color = niri_hdr_to_sdr(color);

#if defined(DEBUG_FLAGS)
    if (tint == 1.0)
        color = vec4(0.0, 0.2, 0.0, 0.2) + color * 0.8;
#endif

    gl_FragColor = color;
}
