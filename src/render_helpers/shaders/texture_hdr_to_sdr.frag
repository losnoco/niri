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
// 1.0 = use niri_gamut (container primaries -> BT.709, identity when equal) instead of the
// built-in BT.2020 constant.
uniform float niri_use_gamut;
uniform mat3 niri_gamut;
// 1.0 = tone map the headroom above the reference white into the SDR range instead of
// clipping it (modified Reinhard on the ICtCp intensity, like KWin); see hdr.frag.
uniform float niri_tonemap;
uniform float niri_tm_v;
uniform float niri_tm_ref_scale;
uniform float niri_tm_out_scale;
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

vec3 niri_pq_inv_eotf(vec3 lin) {
    const float pq_m1 = 0.1593017578125;
    const float pq_m2 = 78.84375;
    const float pq_c1 = 0.8359375;
    const float pq_c2 = 18.8515625;
    const float pq_c3 = 18.6875;
    vec3 y = pow(clamp(lin, vec3(0.0), vec3(1.0)), vec3(pq_m1));
    return pow((pq_c1 + pq_c2 * y) / (1.0 + pq_c3 * y), vec3(pq_m2));
}

float niri_pq_inv_eotf_s(float lin) {
    const float pq_m1 = 0.1593017578125;
    const float pq_m2 = 78.84375;
    const float pq_c1 = 0.8359375;
    const float pq_c2 = 18.8515625;
    const float pq_c3 = 18.6875;
    float y = pow(clamp(lin, 0.0, 1.0), pq_m1);
    return pow((pq_c1 + pq_c2 * y) / (1.0 + pq_c3 * y), pq_m2);
}

float niri_pq_eotf_s(float pq) {
    const float pq_m1 = 0.1593017578125;
    const float pq_m2 = 78.84375;
    const float pq_c1 = 0.8359375;
    const float pq_c2 = 18.8515625;
    const float pq_c3 = 18.6875;
    float p = pow(clamp(pq, 0.0, 1.0), 1.0 / pq_m2);
    float n = max(p - pq_c1, 0.0);
    float d = max(pq_c2 - pq_c3 * p, 0.000001);
    return pow(n / d, 1.0 / pq_m1);
}

// Tone maps normalized linear-light BT.2020 into the SDR range; see hdr.frag for details.
vec3 niri_tonemap_apply(vec3 lin) {
    if (niri_tonemap < 0.5)
        return lin;

    // BT.2020 -> LMS and back (BT.2100 ICtCp definition, column-major).
    const mat3 to_lms = mat3(
        0.412109375,    0.166748046875, 0.024169921875,
        0.52392578125,  0.720458984375, 0.075439453125,
        0.06396484375,  0.11279296875,  0.900390625);
    const mat3 from_lms = mat3(
        3.436606694333, -0.791329555599, -0.025949899691,
       -2.506452118656,  1.983600451792, -0.098913714712,
        0.069845424323, -0.192270896193,  1.124863614402);
    // L'M'S' (PQ-encoded) -> ICtCp and back (column-major).
    const mat3 to_ictcp = mat3(
        0.5,  1.61376953125,   4.378173828125,
        0.5, -3.323486328125, -4.24560546875,
        0.0,  1.709716796875, -0.132568359375);
    const mat3 from_ictcp = mat3(
        1.0,             1.0,             1.0,
        0.008609037038, -0.008609037038,  0.560031335711,
        0.111029625003, -0.111029625003, -0.320627174987);

    vec3 ictcp = to_ictcp * niri_pq_inv_eotf(to_lms * lin);

    float luminance = niri_pq_eotf_s(ictcp.x);
    float relative = max(luminance / niri_tm_ref_scale, 0.0);
    relative = relative * (1.0 + relative * niri_tm_v) / (1.0 + relative);
    ictcp.x = niri_pq_inv_eotf_s(relative * niri_tm_ref_scale);

    lin = from_lms * niri_pq_eotf(from_ictcp * ictcp);
    return clamp(lin, vec3(0.0), vec3(niri_tm_out_scale));
}

// Premultiplied PQ/BT.2020 in, premultiplied electrical sRGB out.
vec4 niri_hdr_to_sdr(vec4 color) {
    float a = color.a;
    vec3 rgb = a > 0.0 ? color.rgb / a : color.rgb;

    rgb = niri_pq_eotf(rgb);

    // Compress the headroom above the reference white into the SDR range instead of
    // clipping it. (For non-BT.2020 containers this happens in container space, a close
    // approximation.)
    rgb = niri_tonemap_apply(rgb);

    // BT.2020 -> BT.709, linear light, D65 (column-major).
    const mat3 to_bt709 = mat3(
        1.660491, -0.124550, -0.018151,
       -0.587641,  1.132900, -0.100579,
       -0.072850, -0.008349,  1.118730);
    rgb = niri_use_gamut > 0.5 ? niri_gamut * rgb : to_bt709 * rgb;

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
