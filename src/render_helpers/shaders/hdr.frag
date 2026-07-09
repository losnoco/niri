// Blend-space transform for HDR outputs: encodes electrical sRGB content into PQ/BT.2020.
//
// niri_hdr_pq = 1.0 enables the transform, 0.0 passes through (SDR outputs; uniforms
// default to 0). niri_ref_lum_scale = reference luminance / 10000 (PQ peak).
//
// niri_linear selects extended-linear content handling (Windows scRGB or a parametric
// ext_linear image description): 0 = off, 1 = BT.709/sRGB container primaries, 2 = BT.2020.
// Encoded 1.0 corresponds to max_lum cd/m²; niri_linear_scale = max_lum / 10000 and
// niri_linear_to_ref = max_lum / reference_lum. Unlike other content it is also transformed
// on SDR outputs, since its raw linear values are meaningless there.

uniform float niri_hdr_pq;
uniform float niri_ref_lum_scale;
uniform float niri_linear;
uniform float niri_linear_scale;
uniform float niri_linear_to_ref;
uniform float niri_hdr_to_sdr;

vec3 niri_pq_inv_eotf(vec3 lin) {
    const float pq_m1 = 0.1593017578125;
    const float pq_m2 = 78.84375;
    const float pq_c1 = 0.8359375;
    const float pq_c2 = 18.8515625;
    const float pq_c3 = 18.6875;
    vec3 y = pow(clamp(lin, 0.0, 1.0), vec3(pq_m1));
    return pow((pq_c1 + pq_c2 * y) / (1.0 + pq_c3 * y), vec3(pq_m2));
}

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

// Premultiplied in, premultiplied out.
vec4 niri_blend(vec4 color) {
    if (niri_hdr_to_sdr > 0.5) {
        float a = color.a;
        vec3 rgb = a > 0.0 ? color.rgb / a : color.rgb;

        rgb = niri_pq_eotf(rgb);

        // BT.2020 -> BT.709, linear light, D65 (column-major).
        const mat3 to_bt709 = mat3(
            1.660491, -0.124550, -0.018151,
           -0.587641,  1.132900, -0.100579,
           -0.072850, -0.008349,  1.118730);
        rgb = to_bt709 * rgb;

        float ref_scale = niri_ref_lum_scale > 0.0 ? niri_ref_lum_scale : 0.0203;
        rgb = clamp(rgb / ref_scale, 0.0, 1.0);
        rgb = pow(rgb, vec3(1.0 / 2.2));
        return vec4(rgb * a, a);
    }

    if (niri_hdr_pq < 0.5 && niri_linear < 0.5)
        return color;

    float a = color.a;
    vec3 rgb = a > 0.0 ? color.rgb / a : color.rgb;

    // BT.709 -> BT.2020 primaries, linear light, D65 (column-major).
    const mat3 to_bt2020 = mat3(
        0.627404, 0.069097, 0.016391,
        0.329283, 0.919540, 0.088013,
        0.043313, 0.011362, 0.895595);

    if (niri_linear > 0.5) {
        if (niri_hdr_pq > 0.5) {
            // Extended-linear content on an HDR output: already linear light; negative
            // values escape the container gamut. The mapping to PQ is absolute
            // (max_lum / 10000 per channel) and deliberately independent of the SDR
            // reference luminance: scRGB-style content is display-referred for a
            // BT.2100/PQ-mode screen and must never be tone mapped, only clamped to the
            // output volume (which niri_pq_inv_eotf does).
            if (niri_linear < 1.5)
                rgb = to_bt2020 * rgb;
            rgb = niri_pq_inv_eotf(rgb * niri_linear_scale);
        } else {
            // Extended-linear content on an SDR output: rendering the raw linear values
            // would blow out (any channel above 1.0 clamps to full-scale in the
            // framebuffer, turning bright colors into white). Anchor the reference white
            // to display white, clamp the HDR headroom away, and gamma-encode.
            //
            // BT.2020 -> BT.709 primaries, linear light, D65 (column-major).
            const mat3 to_bt709 = mat3(
                1.660491, -0.124550, -0.018151,
                -0.587641, 1.132900, -0.100579,
                -0.072850, -0.008349, 1.118730);
            if (niri_linear > 1.5)
                rgb = to_bt709 * rgb;
            rgb = pow(clamp(rgb * niri_linear_to_ref, 0.0, 1.0), vec3(1.0 / 2.2));
        }
        return vec4(rgb * a, a);
    }

    // Pure 2.2 power decode: matches how SDR displays actually respond to the signal
    // (the piecewise sRGB curve would lift shadows).
    rgb = pow(max(rgb, vec3(0.0)), vec3(2.2));

    rgb = to_bt2020 * rgb;

    rgb = niri_pq_inv_eotf(rgb * niri_ref_lum_scale);
    return vec4(rgb * a, a);
}
