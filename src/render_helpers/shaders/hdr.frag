// Blend-space transform for HDR outputs: encodes electrical sRGB content into PQ/BT.2020.
//
// niri_hdr_pq = 1.0 enables the transform, 0.0 passes through (SDR outputs; uniforms
// default to 0). niri_ref_lum_scale = reference luminance / 10000 (PQ peak).

uniform float niri_hdr_pq;
uniform float niri_ref_lum_scale;

vec3 niri_pq_inv_eotf(vec3 lin) {
    const float pq_m1 = 0.1593017578125;
    const float pq_m2 = 78.84375;
    const float pq_c1 = 0.8359375;
    const float pq_c2 = 18.8515625;
    const float pq_c3 = 18.6875;
    vec3 y = pow(clamp(lin, 0.0, 1.0), vec3(pq_m1));
    return pow((pq_c1 + pq_c2 * y) / (1.0 + pq_c3 * y), vec3(pq_m2));
}

// Premultiplied in, premultiplied out.
vec4 niri_blend(vec4 color) {
    if (niri_hdr_pq < 0.5)
        return color;

    float a = color.a;
    vec3 rgb = a > 0.0 ? color.rgb / a : color.rgb;

    // Pure 2.2 power decode: matches how SDR displays actually respond to the signal
    // (the piecewise sRGB curve would lift shadows).
    rgb = pow(max(rgb, vec3(0.0)), vec3(2.2));

    // BT.709 -> BT.2020 primaries, linear light, D65 (column-major).
    const mat3 to_bt2020 = mat3(
        0.627404, 0.069097, 0.016391,
        0.329283, 0.919540, 0.088013,
        0.043313, 0.011362, 0.895595);
    rgb = to_bt2020 * rgb;

    rgb = niri_pq_inv_eotf(rgb * niri_ref_lum_scale);
    return vec4(rgb * a, a);
}
