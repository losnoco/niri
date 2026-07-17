// Blend-space transform for HDR outputs: encodes electrical sRGB content into PQ/BT.2020.
//
// niri_hdr_pq = 1.0 enables the transform, 0.0 passes through (SDR outputs; uniforms
// default to 0). niri_ref_lum_scale = reference luminance / 10000 (PQ peak).
//
// niri_linear selects extended-linear content handling (Windows scRGB or a parametric
// ext_linear image description): 0 = off, nonzero = on; the container gamut comes from
// niri_gamut.
// Encoded 1.0 corresponds to max_lum cd/m²; niri_linear_scale = max_lum / 10000 and
// niri_linear_to_ref = max_lum / reference_lum. Unlike other content it is also transformed
// on SDR outputs, since its raw linear values are meaningless there.

uniform float niri_hdr_pq;
uniform float niri_ref_lum_scale;
uniform float niri_linear;
uniform float niri_linear_scale;
uniform float niri_linear_to_ref;
uniform float niri_hdr_to_sdr;
// 1.0 = re-encode PQ content through niri_gamut (non-BT.2020 PQ containers on HDR outputs).
uniform float niri_pq_gamut;
// 1.0 = use niri_gamut (container primaries -> blend space, identity when equal) instead of
// the built-in constants. Set for every content-aware draw; the frame-wide default path
// (plain sRGB content) leaves it at 0 and uses the constants.
uniform float niri_use_gamut;
uniform mat3 niri_gamut;
// 1.0 = tone map PQ content brighter than the output peak (modified Reinhard on the ICtCp
// intensity, like KWin). niri_tm_v is the curve parameter derived on the CPU so that
// f(input_range) = output_range; niri_tm_ref_scale = reference luminance / 10000 and
// niri_tm_out_scale = output peak luminance / 10000.
uniform float niri_tonemap;
uniform float niri_tm_v;
uniform float niri_tm_ref_scale;
uniform float niri_tm_out_scale;

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

// Tone maps normalized linear-light BT.2020 (1.0 = 10,000 cd/m²) into the output's peak
// luminance: KWin's modified Reinhard on the intensity of ICtCp, which compresses the
// luminance while preserving hue and (PQ-encoded) chroma. The curve satisfies f(0) = 0,
// f(x) <= x and f(input_range) = output_range, and reduces the reference luminance by at
// most half.
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
    // Clip the (small) out-of-range remainder against the output volume, keeping the white
    // point.
    return clamp(lin, vec3(0.0), vec3(niri_tm_out_scale));
}

// Premultiplied in, premultiplied out.
vec4 niri_blend(vec4 color) {
    if (niri_hdr_to_sdr > 0.5) {
        float a = color.a;
        vec3 rgb = a > 0.0 ? color.rgb / a : color.rgb;

        rgb = niri_pq_eotf(rgb);

        // Compress the headroom above the reference white into the SDR range instead of
        // clipping it below. (For non-BT.2020 containers this happens in container space,
        // a close approximation.)
        rgb = niri_tonemap_apply(rgb);

        // BT.2020 -> BT.709, linear light, D65 (column-major).
        const mat3 to_bt709 = mat3(
            1.660491, -0.124550, -0.018151,
           -0.587641,  1.132900, -0.100579,
           -0.072850, -0.008349,  1.118730);
        rgb = niri_use_gamut > 0.5 ? niri_gamut * rgb : to_bt709 * rgb;

        float ref_scale = niri_ref_lum_scale > 0.0 ? niri_ref_lum_scale : 0.0203;
        rgb = clamp(rgb / ref_scale, 0.0, 1.0);
        rgb = pow(rgb, vec3(1.0 / 2.2));
        return vec4(rgb * a, a);
    }

    // PQ content whose container primaries differ from the BT.2020 blend space, or which is
    // brighter than the output peak: decode, convert in linear light (the matrix is
    // scale-invariant, so the normalized PQ linear range works as-is; identity for
    // tone-map-only draws), tone map into the output peak, re-encode.
    if (niri_pq_gamut > 0.5) {
        float a = color.a;
        vec3 rgb = a > 0.0 ? color.rgb / a : color.rgb;
        rgb = niri_tonemap_apply(niri_gamut * niri_pq_eotf(rgb));
        rgb = niri_pq_inv_eotf(rgb);
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
            rgb = niri_use_gamut > 0.5 ? niri_gamut * rgb : to_bt2020 * rgb;
            rgb = niri_pq_inv_eotf(rgb * niri_linear_scale);
        } else {
            // Extended-linear content on an SDR output: rendering the raw linear values
            // would blow out (any channel above 1.0 clamps to full-scale in the
            // framebuffer, turning bright colors into white). Anchor the reference white
            // to display white, clamp the HDR headroom away, and gamma-encode.
            //
            if (niri_use_gamut > 0.5)
                rgb = niri_gamut * rgb;
            rgb = pow(clamp(rgb * niri_linear_to_ref, 0.0, 1.0), vec3(1.0 / 2.2));
        }
        return vec4(rgb * a, a);
    }

    // Pure 2.2 power decode: matches how SDR displays actually respond to the signal
    // (the piecewise sRGB curve would lift shadows).
    rgb = pow(max(rgb, vec3(0.0)), vec3(2.2));

    rgb = niri_use_gamut > 0.5 ? niri_gamut * rgb : to_bt2020 * rgb;

    rgb = niri_pq_inv_eotf(rgb * niri_ref_lum_scale);
    return vec4(rgb * a, a);
}
